"""Coverage for ennbo source patch used by the Pixi macOS ennbo build task."""

from __future__ import annotations

from admin.patch_enn_failure_tolerance_dim import _MARKER, patch_enn_repo


def test_patch_enn_failure_tolerance_dim_exports() -> None:
    assert "failure_tolerance_dim" in _MARKER
    assert callable(patch_enn_repo)


def test_patch_enn_failure_tolerance_dim_updates_current_upstream_shape(tmp_path) -> None:
    root = tmp_path
    config = root / "rust/crates/ennbo/src/config.rs"
    trust_region = root / "rust/crates/ennbo/src/trust_region.rs"
    optimizer = root / "rust/crates/ennbo/src/optimizer/mod.rs"
    py_optimizer = root / "rust/crates/enn-py/src/py_optimizer.rs"
    for path in (config, trust_region, optimizer, py_optimizer):
        path.parent.mkdir(parents=True, exist_ok=True)

    config.write_text(
        """
pub struct OptimizerConfig {
    pub noise_aware: bool,
}
impl Default for OptimizerConfig {
    fn default() -> Self {
        Self {
            noise_aware: false,
        }
    }
}

/// Surrogate type configuration.
pub struct ConfigOverrides {
    pub noise_aware: Option<bool>,
    pub enn_storage: Option<EnnStorage>,
    pub work_dir: Option<PathBuf>,
    pub trust_region_kind: Option<String>,
}

fn apply_enn_surrogate_fields() {}

impl ConfigOverrides {
    pub fn apply_to(&self, mut config: OptimizerConfig) -> OptimizerConfig {
        if let Some(na) = self.noise_aware {
            config.noise_aware = na;
        }
        config
    }
}

pub fn turbo_enn_config() -> OptimizerConfig {
    OptimizerConfig {
        noise_aware: false,
    }
}
""",
        encoding="utf-8",
    )
    trust_region.write_text(
        """
pub struct TurboTrustRegion {
    /// Length configuration.
    config: TRLengthConfig,
}

impl TurboTrustRegion {
    pub fn new() -> Self {
        Self {
            config,
        }
    }

    /// Initialize or update batch size.
    pub fn set_num_arms(&mut self) {}

    fn compute_failure_tolerance(&mut self) {
        if let Some(num_arms) = self.num_arms {
            let tolerance =
                ((4.0 / num_arms as f64).max(self.num_dim as f64 / num_arms as f64)).ceil() as i32;
        }
    }
}
""",
        encoding="utf-8",
    )
    optimizer.write_text(
        """
impl Optimizer {
    pub fn new_with_strategy() {
        let tr_state = TrustRegionState::from_config(num_dim, &config.trust_region, rng)
            .map_err(|e| ENNError::InvalidParameter(e.to_string()))?;
    }
}
""",
        encoding="utf-8",
    )
    py_optimizer.write_text(
        """
pub(crate) fn apply_scalar_overrides() -> PyResult<()> {
    overrides.scale_x = optional_bool(dict, "scale_x")?;
    Ok(())
}
""",
        encoding="utf-8",
    )

    patch_enn_repo(root)

    config_text = config.read_text(encoding="utf-8")
    assert "pub failure_tolerance_dim: Option<f64>," in config_text
    assert "config.failure_tolerance_dim = Some(d);" in config_text
    assert "failure_tolerance_dim: None," in config_text
    assert "set_failure_tolerance_dim" in trust_region.read_text(encoding="utf-8")
    assert "config.failure_tolerance_dim" in optimizer.read_text(encoding="utf-8")
    assert 'optional_f64(dict, "failure_tolerance_dim")' in py_optimizer.read_text(encoding="utf-8")
