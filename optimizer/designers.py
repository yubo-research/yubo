from .designer_errors import NoSuchDesignerError
from .designer_registry import _DESIGNER_DISPATCH, _DESIGNER_OPTION_SPECS, _SIMPLE_DISPATCH, _SimpleContext
from .designer_spec import DesignerCatalogEntry, DesignerOptionSpec, DesignerSpec, parse_designer_spec

__all__ = [
    "Designers",
    "NoSuchDesignerError",
    "DesignerCatalogEntry",
    "DesignerOptionSpec",
    "DesignerSpec",
]


class Designers:
    def __init__(self, policy, num_arms):
        self._policy = policy
        self._num_arms = num_arms

    def is_valid(self, designer_name):
        # Must be cheap: used for validation / CLI help.
        try:
            spec = parse_designer_spec(designer_name)
        except Exception:
            return False
        return spec.base in _DESIGNER_DISPATCH

    def _bt_designer(
        self,
        acq_factory,
        acq_kwargs=None,
        init_sobol=1,
        opt_sequential=False,
        num_restarts=10,
        raw_samples=10,
        start_at_max=False,
        num_keep=None,
        keep_style=None,
        model_spec=None,
        sample_around_best=False,
    ):
        from .bt_designer import BTDesigner

        return BTDesigner(
            self._policy,
            acq_factory,
            acq_kwargs=acq_kwargs,
            num_keep=num_keep,
            keep_style=keep_style,
            model_spec=model_spec,
            init_sobol=init_sobol,
            opt_sequential=opt_sequential,
            optimizer_options={
                "batch_limit": 10,
                "maxiter": 1000,
                "sample_around_best": sample_around_best,
            },
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            start_at_max=start_at_max,
        )

    def create(self, designer_name):
        spec = parse_designer_spec(designer_name)
        handler = _DESIGNER_DISPATCH.get(spec.base)
        if handler is None:
            raise NoSuchDesignerError(spec.base)

        num_keep = spec.general["num_keep"]
        keep_style = spec.general["keep_style"]
        model_spec = spec.general["model_spec"]
        sample_around_best = spec.general["sample_around_best"]

        init_ax_default = max(5, 2 * self._policy.num_params())
        init_yubo_default = self._num_arms
        default_num_X_samples = max(64, 10 * self._num_arms)

        def bt(*args, **kw):
            return self._bt_designer(
                *args,
                num_keep=num_keep,
                keep_style=keep_style,
                model_spec=model_spec,
                sample_around_best=sample_around_best,
                **kw,
            )

        num_keep_val = num_keep if keep_style == "trailing" else None
        ctx = _SimpleContext(
            self._policy,
            self._num_arms,
            bt,
            num_keep=num_keep,
            keep_style=keep_style,
            num_keep_val=num_keep_val,
            init_yubo_default=init_yubo_default,
            init_ax_default=init_ax_default,
            default_num_X_samples=default_num_X_samples,
        )
        return handler(ctx, spec.specific)

    def catalog(self) -> list[DesignerCatalogEntry]:
        """
        Return a catalog of supported designer base names.

        Notes:
        - This lists *designer-specific* options only (not general BT options like num_keep).
        - Dispatch functions are the underlying builders (e.g. `_h_sobol`, `_d_ts_sweep`).
        """
        entries: list[DesignerCatalogEntry] = []

        # "No opts" designers expose their underlying _h_* dispatchers.
        for base_name, fn in _SIMPLE_DISPATCH.items():
            entries.append(
                DesignerCatalogEntry(
                    base_name=base_name,
                    options=_DESIGNER_OPTION_SPECS.get(base_name, []),
                    dispatch=fn,
                )
            )

        for base_name, fn in _DESIGNER_DISPATCH.items():
            if base_name in _SIMPLE_DISPATCH:
                continue
            entries.append(
                DesignerCatalogEntry(
                    base_name=base_name,
                    options=_DESIGNER_OPTION_SPECS.get(base_name, []),
                    dispatch=fn,
                )
            )

        entries.sort(key=lambda e: e.base_name)
        return entries
