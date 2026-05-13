from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from llm.console_observer import UnifiedConsoleManager
from llm.tasks_verifiers import _VLLMRLMClient


logger = logging.getLogger(__name__)


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

    def full_proof_path(self) -> str:
        return f"{self.workdir}/{self.proof_path}.{self.extension}"


LANGUAGES = {
    "lean4": LanguageConfig(
        name="lean4",
        extension="lean",
        docker_image="leanprover/lean4:v4.7.0",
        compile_cmd="lake env lean {path}",
        guard_begin="-- guard begin",
        guard_end="-- guard end",
        workdir="/workspace",
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
        compile_cmd="isabelle build -D .",
        guard_begin="(* guard begin *)",
        guard_end="(* guard end *)",
        workdir="/home/isabelle",
    ),
}


class FormalRubric:
    """Judge that uses a formal compiler (Lean, Coq, Isabelle) to verify proofs."""

    def __init__(self, lang_cfg: LanguageConfig):
        self.lang_cfg = lang_cfg

    async def score_state(self, state: dict[str, Any]) -> float:
        sandbox_client = state.get("sandbox_client")
        sandbox_id = state.get("sandbox_id")
        if not sandbox_client or not sandbox_id:
            return 0.0

        proof_path = self.lang_cfg.full_proof_path()

        # 1. Verification of the Guard (Anti-Cheating)
        try:
            res = await sandbox_client.execute_command(sandbox_id, f"cat {proof_path}")
            content = res.stdout or ""
            expected_stmt = state.get("expected_statement", "")

            # Simple check: Is the original theorem statement still there?
            if expected_stmt and expected_stmt not in content:
                logger.warning("Theorem statement was tampered with!")
                return 0.0
        except Exception:
            pass

        # 2. Compilation check
        try:
            cmd = self.lang_cfg.compile_cmd.format(path=proof_path, path_no_ext=proof_path.rsplit(".", 1)[0])
            res = await sandbox_client.execute_command(sandbox_id, f"cd {self.lang_cfg.workdir} && {cmd}", timeout=60)

            # Success is usually exit code 0 and no 'sorry' / 'admit'
            output = (res.stdout or "") + (res.stderr or "")
            has_sorry = any(s in output.lower() for s in ["sorry", "admit", "axiom"])

            if res.exit_code == 0 and not has_sorry:
                return 1.0
        except Exception:
            pass

        return 0.0


class TheoremProvingTask:
    """Universal task for formal theorem proving (Lean, Coq, Isabelle)."""

    def __init__(
        self,
        language: str = "lean4",
        dataset_name: str = "cat-searcher/minif2f-lean4",
        dataset_split: str = "validation",
        batch_size: int = 1,
        seed: int = 0,
        tokenizer: Any | None = None,
        console: UnifiedConsoleManager | None = None,
    ):
        self.lang_cfg = LANGUAGES.get(language)
        if not self.lang_cfg:
            raise ValueError(f"Unsupported language: {language}")

        self.dataset_name = dataset_name
        self.dataset_split = dataset_split
        self.batch_size = batch_size
        self.seed = seed
        self.tokenizer = tokenizer
        self.console = console
        self.idx = 0
        self._dataset = None

    def __getstate__(self) -> dict[str, Any]:
        return {
            "lang_name": self.lang_cfg.name,
            "dataset_name": self.dataset_name,
            "dataset_split": self.dataset_split,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "idx": self.idx,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.lang_cfg = LANGUAGES[state["lang_name"]]
        self.dataset_name = state["dataset_name"]
        self.dataset_split = state["dataset_split"]
        self.batch_size = state["batch_size"]
        self.seed = state["seed"]
        self.idx = state["idx"]
        self.tokenizer = None
        self.console = None
        self._dataset = None

    def get_batch(self) -> tuple[list[str], list[Any]]:
        if self._dataset is None:
            from datasets import load_dataset

            self._dataset = load_dataset(self.dataset_name, split=self.dataset_split)

        indices = np.arange(self.idx, self.idx + self.batch_size) % len(self._dataset)
        self.idx += self.batch_size
        rows = [self._dataset[int(i)] for i in indices]

        prompts = []
        answers = []
        for row in rows:
            stmt = row.get("formal_statement") or row.get("statement") or ""
            prompt = f"Prove the following theorem in {self.lang_cfg.name}:\n\n{stmt}"
            prompts.append(prompt)
            answers.append({"statement": stmt, "row": row})

        return prompts, answers

    async def generate_and_score_async(
        self,
        llm: Any,
        prompts: list[str],
        sampling_params_kwargs: dict[str, Any],
        lora_request_specs: list[tuple[str, int, str]] | None,
        answers: list[Any],
        args: Any,
    ) -> tuple[list[float], dict[str, float], list[str]]:
        from prime_sandboxes import AsyncSandboxClient

        sandbox_client = AsyncSandboxClient()
        tasks = []
        for i in range(len(prompts)):
            tasks.append(
                self._run_single(
                    llm,
                    prompts[i],
                    sampling_params_kwargs,
                    lora_request_specs[i] if lora_request_specs else None,
                    answers[i],
                    sandbox_client,
                )
            )

        results = await asyncio.gather(*tasks)
        fitnesses = [r[0] for r in results]
        logs = [r[1] for r in results]
        self.last_logs = logs[:3]  # Store first few for observation

        return fitnesses, {}, logs

    async def _run_single(self, llm, prompt, sampling, lora_spec, answer, sandbox_client):
        from prime_sandboxes import CreateSandboxRequest

        # 1. Create Sandbox
        req = CreateSandboxRequest(
            docker_image=self.lang_cfg.docker_image,
            name=f"thm-{self.lang_cfg.name}-{base64.b32encode(np.random.bytes(5)).decode().lower()}",
        )
        sandbox = await sandbox_client.create(req)
        sandbox_id = sandbox.id

        try:
            await sandbox_client.wait_for_creation(sandbox_id)
            await self._setup_initial_proof(sandbox_id, answer, sandbox_client)

            # 3. ReAct Loop
            trajectory = await self._react_loop(llm, prompt, sampling, lora_spec, sandbox_id, sandbox_client)

            # 4. Final Scoring
            rubric = FormalRubric(self.lang_cfg)
            state = {
                "sandbox_client": sandbox_client,
                "sandbox_id": sandbox_id,
                "expected_statement": answer["statement"],
            }
            reward = await rubric.score_state(state)

            if self.console:
                await self.console.broadcast_reward(reward, {"status": "success" if reward > 0 else "failure"})

            log_str = f"PROMPT: {prompt}\nREWARD: {reward}\nTRAJECTORY:\n" + "\n".join(trajectory)
            return reward, log_str

        finally:
            await sandbox_client.delete(sandbox_id)

    async def _setup_initial_proof(self, sandbox_id, answer, sandbox_client):
        stmt = answer["statement"]
        proof_path = self.lang_cfg.full_proof_path()
        initial_content = f"{self.lang_cfg.guard_begin}\n{stmt}\n{self.lang_cfg.guard_end}\n  sorry"

        # Use base64 to avoid quoting issues in bash
        b64_content = base64.b64encode(initial_content.encode()).decode()
        await sandbox_client.execute_command(sandbox_id, f"echo {b64_content} | base64 -d > {proof_path}")

    async def _react_loop(self, llm, prompt, sampling, lora_spec, sandbox_id, sandbox_client):
        proof_path = self.lang_cfg.full_proof_path()
        client = _VLLMRLMClient(llm, lora_spec, sampling, tokenizer=self.tokenizer)
        system_msg = (
            f"You are a formal theorem prover for {self.lang_cfg.name}. Follow the ReAct pattern.\n"
            "You can interact with the system using markdown code blocks:\n"
            f"1. ```{self.lang_cfg.name}\n<code>\n```: Writes code to the proof file and runs the compiler.\n"
            "2. ```bash\n<command>\n```: Runs a bash command in the sandbox.\n"
            "Always start with a thought block, then use a tool call if needed. "
            "The output of the tool will be provided in the next turn. "
            "When you have finished the proof and verified it with the compiler, write 'QED'."
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ]
        trajectory = []
        max_turns = 10

        for turn_idx in range(max_turns):
            # 1. Assistant Turn
            res = await client.create(messages)
            msg = res.choices[0].message
            trajectory.append(f"Model: {msg.content}")

            if self.console:
                await self.console.broadcast_step(turn_idx, {"role": "assistant", "content": msg.content})

            if "QED" in msg.content:
                break

            messages.append({"role": "assistant", "content": msg.content})

            # 2. Tool Turn
            if not msg.tool_calls:
                break

            for tool_call in msg.tool_calls:
                output = await self._execute_tool(tool_call, sandbox_id, sandbox_client, proof_path)
                trajectory.append(f"Tool [{tool_call.name}]: {output}")
                messages.append({"role": "tool", "content": output, "tool_call_id": tool_call.id})

                if self.console:
                    await self.console.broadcast_step(
                        turn_idx,
                        {"role": "tool", "name": tool_call.name, "output": output},
                    )

        return trajectory

    async def _execute_tool(self, tool_call, sandbox_id, sandbox_client, proof_path):
        import json

        t_name = tool_call.name
        try:
            args = json.loads(tool_call.arguments)
            code = args.get("code", "")

            if t_name == self.lang_cfg.name or t_name == self.lang_cfg.extension:
                # Write content to file
                b64_c = base64.b64encode(code.encode()).decode()
                await sandbox_client.execute_command(sandbox_id, f"echo {b64_c} | base64 -d > {proof_path}")

                # Run compiler
                cmd = self.lang_cfg.compile_cmd.format(path=proof_path, path_no_ext=proof_path.rsplit(".", 1)[0])
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

    def score(self, generations, truncateds, answer, *, pass_at_k=False) -> tuple[float, tuple[Any, ...], np.ndarray]:
        # This task primarily uses generate_and_score_async
        return 0.0, (), np.asarray([])

    @property
    def task_name(self) -> str:
        return f"thm:{self.lang_cfg.name}"


__all__ = ["TheoremProvingTask", "LanguageConfig", "LANGUAGES", "FormalRubric"]
