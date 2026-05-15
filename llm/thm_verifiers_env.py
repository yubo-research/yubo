from __future__ import annotations

import base64
import json
import logging
import os
import re
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

logger = logging.getLogger(__name__)

_LEAN_HEADER = (
    "import Mathlib",
    "import Aesop",
    "",
    "set_option maxHeartbeats 0",
    "open BigOperators Real Nat Topology Rat",
    "",
)

_FORMAL_PLACEHOLDER_RE = re.compile(r"\b(sorry|admit|admitted|axiom|oops)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LanguageConfig:
    name: str
    extension: str
    docker_image: str
    compile_cmd: str
    guard_begin: str
    guard_end: str
    workdir: str
    proof_path: str = "/tmp/proof"
    session_dir: str | None = None
    root_file: str | None = None

    def full_proof_path(self) -> str:
        proof_path = self.proof_path
        if proof_path.startswith("/"):
            proof_path = proof_path[1:]
        return f"{self.workdir}/{proof_path}.{self.extension}"


LANGUAGES = {
    "lean4": LanguageConfig(
        name="lean4",
        extension="lean",
        docker_image="yubo-lean4-mathlib:latest",
        compile_cmd="yubo-lean {path}",
        guard_begin="-- guard begin",
        guard_end="-- guard end",
        workdir="/workspace",
        proof_path="tmp/proof",
    ),
    "coq": LanguageConfig(
        name="coq",
        extension="v",
        docker_image="coqorg/coq:8.19",
        compile_cmd="coqc {path}",
        guard_begin="(* guard begin *)",
        guard_end="(* guard end *)",
        workdir="/home/coq",
    ),
    "isabelle": LanguageConfig(
        name="isabelle",
        extension="thy",
        docker_image="makarius/isabelle:latest",
        compile_cmd="isabelle build -D {session_dir}",
        guard_begin="(* guard begin *)",
        guard_end="(* guard end *)",
        workdir="/home/isabelle",
        proof_path="/yubo_proof/Proof",
        session_dir="/home/isabelle/yubo_proof",
        root_file="/home/isabelle/yubo_proof/ROOT",
    ),
}


def _shell_quote(path: str) -> str:
    return "'" + path.replace("'", "'\"'\"'") + "'"


def _isabelle_root_content(session_name: str) -> str:
    return "\n".join(
        [
            f"session {session_name} = Pure",
            '  theories "Proof"',
            "",
        ]
    )


def _statement_head(statement: str) -> str:
    return statement.split(":=", 1)[0].strip()


def _lean_with_header(code: str) -> str:
    stripped = code.strip()
    if "import Mathlib" in stripped:
        return stripped + "\n"
    return "\n".join([*_LEAN_HEADER, stripped, ""])


def _lean_scaffold(statement: str) -> str:
    stmt = statement.strip()
    if ":= sorry" in stmt:
        stmt = stmt.replace(":= sorry", ":= by\n  sorry", 1)
    elif ":= by" not in stmt and ":=" not in stmt:
        stmt = f"{stmt} := by\n  sorry"
    return _lean_with_header(stmt)


def _strip_markdown_fences(text: str) -> str:
    match = re.search(r"```(?:lean4?|Lean4?)\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def _lean_fenced_code(text: str) -> str | None:
    match = re.search(r"```(?:lean4?|Lean4?)\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match is None:
        return None
    return match.group(1).strip()


def _lean_candidate_from_text(text: str, statement: str) -> str | None:
    fenced_code = _lean_fenced_code(text)
    code = fenced_code if fenced_code is not None else text.strip()
    if not code:
        return None

    if "```" in code:
        return None

    if "theorem " in code or "lemma " in code or "example " in code:
        return _lean_with_header(code)

    head = _statement_head(statement)
    if not head:
        return None

    if fenced_code is None and not _looks_like_lean_proof_body(code):
        return None

    proof = code
    if proof.startswith("by"):
        replacement = f"{head} := {proof}"
    else:
        replacement = f"{head} := by\n{_indent_nonempty(proof, spaces=2)}"
    return _lean_scaffold(replacement)


def _looks_like_lean_proof_body(code: str) -> bool:
    stripped = _strip_formal_comments(code).strip()
    if not stripped:
        return False
    first = stripped.splitlines()[0].strip()
    if not first:
        return False
    tactic_heads = (
        "by",
        "rfl",
        "trivial",
        "simp",
        "simpa",
        "norm_num",
        "ring",
        "ring_nf",
        "omega",
        "linarith",
        "nlinarith",
        "aesop",
        "decide",
        "tauto",
        "assumption",
        "contradiction",
        "constructor",
        "intro",
        "intros",
        "exact",
        "apply",
        "refine",
        "have",
        "calc",
        "rw",
        "rewrite",
        "field_simp",
        "positivity",
        "all_goals",
        " ·",
    )
    return first.startswith(tactic_heads)


def _strip_formal_comments(text: str) -> str:
    text = re.sub(r"/-.*?-/", "", text, flags=re.DOTALL)
    text = re.sub(r"\(\*.*?\*\)", "", text, flags=re.DOTALL)
    text = re.sub(r"--.*", "", text)
    text = re.sub(r"#.*", "", text)
    return text


def _has_formal_placeholder(text: str) -> bool:
    return bool(_FORMAL_PLACEHOLDER_RE.search(_strip_formal_comments(text)))


def _contains_expected_statement(content: str, expected_statement: str) -> bool:
    expected_head = _statement_head(_strip_formal_comments(expected_statement))
    if not expected_head:
        return True
    content_without_comments = _strip_formal_comments(content)
    return _normalize_formal_text(expected_head) in _normalize_formal_text(content_without_comments)


def _normalize_formal_text(text: str) -> str:
    return " ".join(text.split())


def _truncate_feedback(text: str, *, limit: int = 6000) -> str:
    if len(text) <= limit:
        return text
    return text[-limit:]


def _indent_nonempty(text: str, *, spaces: int) -> str:
    prefix = " " * spaces
    return "\n".join(prefix + line if line.strip() else line for line in text.splitlines())


class FormalRubric:
    """Judge that uses a formal compiler to verify the current proof file."""

    def __init__(self, lang_cfg: LanguageConfig):
        self.lang_cfg = lang_cfg

    async def score_state(self, state: dict[str, Any]) -> float:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            return 0.0

        proof_path = self.lang_cfg.full_proof_path()

        try:
            res = await sandbox_client.execute_command(sandbox_id, f"cat {proof_path}")
            content = res.stdout or ""
            expected_stmt = state.get("expected_statement", "")

            if expected_stmt and not _contains_expected_statement(content, expected_stmt):
                logger.warning("Theorem statement was tampered with.")
                return 0.0
            if _has_formal_placeholder(content):
                logger.info("Formal proof still contains a placeholder.")
                return 0.0
        except Exception:
            pass

        try:
            cmd = _format_compile_cmd(self.lang_cfg, proof_path)
            res = await sandbox_client.execute_command(sandbox_id, f"cd {self.lang_cfg.workdir} && {cmd}", timeout=60)

            output = (res.stdout or "") + (res.stderr or "")
            has_placeholder = _has_formal_placeholder(output)

            if res.exit_code == 0 and not has_placeholder:
                return 1.0
        except Exception:
            pass

        return 0.0


class TheoremVerifierEnv:
    """Verifier-style theorem environment backed by prime sandboxes.

    This owns theorem-specific harness behavior while inference stays outside the
    environment through the injected local-vLLM client.
    """

    def __init__(self, lang_cfg: LanguageConfig, *, max_turns: int = 10) -> None:
        self.lang_cfg = lang_cfg
        self.max_turns = int(max_turns)
        self.rubric = FormalRubric(lang_cfg)
        self.tools = {
            lang_cfg.name: self.execute_tool,
            lang_cfg.extension: self.execute_tool,
            "bash": self.execute_tool,
        }

    @property
    def docker_image(self) -> str:
        env_var = {
            "lean4": "THM_LEAN4_DOCKER_IMAGE",
            "coq": "THM_COQ_DOCKER_IMAGE",
            "isabelle": "THM_ISABELLE_DOCKER_IMAGE",
        }.get(self.lang_cfg.name)
        if env_var is not None:
            return os.environ.get(env_var, self.lang_cfg.docker_image)
        return self.lang_cfg.docker_image

    def system_prompt(self) -> str:
        if self.lang_cfg.name == "lean4":
            return (
                "Complete Lean 4 proof code in Mathlib. Return Lean code only. "
                "Do not write prose, markdown explanation, sorry, admit, or axiom. "
                "Preserve the theorem statement."
            )
        return (
            f"You are a formal theorem prover for {self.lang_cfg.name}. Follow the ReAct pattern.\n"
            "You can interact with the system using markdown code blocks:\n"
            f"1. ```{self.lang_cfg.name}\n<code>\n```: Writes code to the proof file and runs the compiler.\n"
            "2. ```bash\n<command>\n```: Runs a bash command in the sandbox.\n"
            "Always start with a thought block, then use a tool call if needed. "
            "The output of the tool will be provided in the next turn. "
            "When you have finished the proof and verified it with the compiler, write 'QED'."
        )

    def user_prompt(self, prompt: str, answer: dict[str, Any]) -> str:
        if self.lang_cfg.name != "lean4":
            return prompt
        statement = str(answer.get("statement", "")).strip()
        return f"Complete the following Lean 4 code:\n```lean4\n{_lean_scaffold(statement).strip()}\n\n```"

    def format_messages_for_generation(self, messages: list[dict[str, Any]]) -> str | None:
        if self.lang_cfg.name != "lean4":
            return None

        user_content = ""
        assistant_content = ""
        tool_content = ""
        for message in messages:
            role = message.get("role")
            content = str(message.get("content", ""))
            if role == "user":
                user_content = content
            elif role == "assistant":
                assistant_content = content
            elif role == "tool":
                tool_content = content

        if not tool_content:
            return user_content.rstrip() + "\n"

        prompt = [
            user_content.rstrip(),
            "",
            "The previous Lean completion did not verify.",
        ]
        if assistant_content.strip():
            prompt.extend(
                [
                    "Previous completion:",
                    "```lean4",
                    _strip_markdown_fences(assistant_content).strip(),
                    "```",
                ]
            )
        prompt.extend(
            [
                "Lean feedback:",
                "```text",
                _truncate_feedback(tool_content).strip(),
                "```",
                "Return a corrected complete Lean file. Lean code only.",
                "",
            ]
        )
        return "\n".join(prompt)

    async def setup_initial_proof(self, sandbox_id: str, answer: dict[str, Any], sandbox_client: Any) -> None:
        stmt = answer["statement"]
        proof_path = self.lang_cfg.full_proof_path()
        if self.lang_cfg.name == "isabelle":
            initial_content = f"theory Proof\n  imports Main\nbegin\n\n{self.lang_cfg.guard_begin}\n{stmt}\n{self.lang_cfg.guard_end}\n  sorry\n\nend"
            await self._setup_isabelle_session(sandbox_id, sandbox_client)
        elif self.lang_cfg.name == "lean4":
            initial_content = _lean_scaffold(stmt)
        else:
            initial_content = f"{self.lang_cfg.guard_begin}\n{stmt}\n{self.lang_cfg.guard_end}\n  sorry"

        await self._write_file(sandbox_id, sandbox_client, proof_path, initial_content)

    def _lean_tool_call_from_text(self, text: str, answer: dict[str, Any]) -> Any | None:
        code = _lean_candidate_from_text(text, str(answer.get("statement", "")))
        if code is None:
            return None
        return SimpleNamespace(
            id="synthetic-lean4",
            name="lean4",
            arguments=json.dumps({"code": code}),
        )

    async def execute_tool(self, tool_call: Any, sandbox_id: str, sandbox_client: Any) -> str:
        t_name = tool_call.name
        try:
            args = json.loads(tool_call.arguments)
            code = args.get("code", "")
            if self.lang_cfg.name == "lean4" and (t_name == self.lang_cfg.name or t_name == self.lang_cfg.extension):
                code = _lean_candidate_from_text(code, code) or code

            if t_name == self.lang_cfg.name or t_name == self.lang_cfg.extension:
                proof_path = self.lang_cfg.full_proof_path()
                await self._write_file(sandbox_id, sandbox_client, proof_path, code)

                cmd = _format_compile_cmd(self.lang_cfg, proof_path)
                res = await sandbox_client.execute_command(sandbox_id, f"cd {self.lang_cfg.workdir} && {cmd}", timeout=60)
                output = (res.stdout or "") + (res.stderr or "")
            elif t_name == "bash":
                res = await sandbox_client.execute_command(sandbox_id, code, timeout=60)
                output = (res.stdout or "") + (res.stderr or "")
            else:
                output = f"Unknown tool: {t_name}. Please use ```{self.lang_cfg.name}``` or ```bash```."
        except Exception as e:
            output = f"Error executing tool: {str(e)}"
        return output

    async def _setup_isabelle_session(self, sandbox_id: str, sandbox_client: Any) -> None:
        if not self.lang_cfg.session_dir or not self.lang_cfg.root_file:
            raise ValueError("isabelle backend requires session_dir and root_file.")
        await sandbox_client.execute_command(sandbox_id, f"mkdir -p {_shell_quote(self.lang_cfg.session_dir)}")
        await self._write_file(
            sandbox_id,
            sandbox_client,
            self.lang_cfg.root_file,
            _isabelle_root_content("YuboProof"),
        )

    async def _write_file(self, sandbox_id: str, sandbox_client: Any, path: str, content: str) -> None:
        parent = path.rsplit("/", 1)[0]
        b64_content = base64.b64encode(content.encode()).decode()
        await sandbox_client.execute_command(sandbox_id, f"mkdir -p {_shell_quote(parent)}")
        await sandbox_client.execute_command(sandbox_id, f"echo {b64_content} | base64 -d > {_shell_quote(path)}")


def _format_compile_cmd(lang_cfg: LanguageConfig, proof_path: str) -> str:
    return lang_cfg.compile_cmd.format(
        path=proof_path,
        path_no_ext=proof_path.rsplit(".", 1)[0],
        session_dir=lang_cfg.session_dir or "",
    )


__all__ = ["FormalRubric", "LANGUAGES", "LanguageConfig", "TheoremVerifierEnv"]
