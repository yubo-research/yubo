#!/usr/bin/env python3
"""Patch ennbo checkout so Rust TuRBO uses failure_tolerance_dim for the TR clock."""

from __future__ import annotations

import sys
from pathlib import Path

_MARKER = "failure_tolerance_dim: Option<f64>"


def _patch_config_rs(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if _MARKER in text:
        return
    text = text.replace(
        "    pub noise_aware: bool,\n",
        "    pub noise_aware: bool,\n    pub failure_tolerance_dim: Option<f64>,\n",
        1,
    )
    text = text.replace(
        "            noise_aware: false,\n        }\n    }\n}\n\n/// Surrogate type configuration.",
        "            noise_aware: false,\n            failure_tolerance_dim: None,\n        }\n    }\n}\n\n/// Surrogate type configuration.",
        1,
    )
    text = text.replace(
        "    pub noise_aware: Option<bool>,\n    pub trust_region_kind: Option<String>,\n",
        "    pub noise_aware: Option<bool>,\n    pub failure_tolerance_dim: Option<f64>,\n    pub trust_region_kind: Option<String>,\n",
        1,
    )
    insert = """        if let Some(d) = self.failure_tolerance_dim {
            config.failure_tolerance_dim = Some(d);
        }
"""
    text = text.replace(
        "        if let Some(kind) = &self.trust_region_kind {",
        insert + "        if let Some(kind) = &self.trust_region_kind {",
        1,
    )
    text = text.replace(
        "        noise_aware: false,\n    }\n}",
        "        noise_aware: false,\n        failure_tolerance_dim: None,\n    }\n}",
    )
    path.write_text(text, encoding="utf-8")


def _patch_trust_region_rs(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "failure_tolerance_dim: Option<f64>" in text.split("pub struct TurboTrustRegion")[1][:800]:
        return
    text = text.replace(
        "    /// Length configuration.\n    config: TRLengthConfig,\n}",
        "    /// Length configuration.\n    config: TRLengthConfig,\n    /// Optional effective dimension for failure tolerance (defaults to num_dim).\n    failure_tolerance_dim: Option<f64>,\n}",
        1,
    )
    text = text.replace(
        "            config,\n        }\n    }\n\n    /// Initialize or update batch size.",
        "            config,\n            failure_tolerance_dim: None,\n        }\n    }\n\n    /// Override the ambient dimension used for failure tolerance.\n    pub fn set_failure_tolerance_dim(&mut self, dim: f64) {\n        self.failure_tolerance_dim = Some(dim);\n        self.compute_failure_tolerance();\n    }\n\n    /// Initialize or update batch size.",
        1,
    )
    text = text.replace(
        "            let tolerance =\n                ((4.0 / num_arms as f64).max(self.num_dim as f64 / num_arms as f64)).ceil() as i32;",
        "            let eff_dim = self\n                .failure_tolerance_dim\n                .unwrap_or(self.num_dim as f64);\n            let tolerance =\n                ((4.0 / num_arms as f64).max(eff_dim / num_arms as f64)).ceil() as i32;",
        1,
    )
    path.write_text(text, encoding="utf-8")


def _patch_optimizer_mod_rs(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if "config.failure_tolerance_dim" in text:
        return
    text = text.replace(
        "        let tr_state = TrustRegionState::from_config(num_dim, &config.trust_region, rng)\n            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;\n",
        "        let mut tr_state = TrustRegionState::from_config(num_dim, &config.trust_region, rng)\n            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;\n        if let Some(dim) = config.failure_tolerance_dim {\n            if let TrustRegionState::Turbo(t) = &mut tr_state {\n                t.set_failure_tolerance_dim(dim);\n            }\n        }\n",
        1,
    )
    path.write_text(text, encoding="utf-8")


def _patch_py_optimizer_rs(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    if 'optional_f64(dict, "failure_tolerance_dim")' in text:
        return
    text = text.replace(
        '    overrides.scale_x = optional_bool(dict, "scale_x")?;\n    Ok(())\n}',
        '    overrides.scale_x = optional_bool(dict, "scale_x")?;\n    overrides.failure_tolerance_dim = optional_f64(dict, "failure_tolerance_dim")?;\n    Ok(())\n}',
        1,
    )
    path.write_text(text, encoding="utf-8")


def patch_enn_repo(root: Path) -> None:
    root = root.resolve()
    _patch_config_rs(root / "rust/crates/ennbo/src/config.rs")
    _patch_trust_region_rs(root / "rust/crates/ennbo/src/trust_region.rs")
    _patch_optimizer_mod_rs(root / "rust/crates/ennbo/src/optimizer/mod.rs")
    _patch_py_optimizer_rs(root / "rust/crates/enn-py/src/py_optimizer.rs")


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} ENN_REPO_ROOT", file=sys.stderr)
        return 2
    patch_enn_repo(Path(argv[1]))
    print("patched enn for failure_tolerance_dim")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
