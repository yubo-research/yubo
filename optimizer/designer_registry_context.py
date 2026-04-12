class _SimpleContext:
    def __init__(
        self,
        policy,
        num_arms,
        bt,
        *,
        num_keep,
        keep_style,
        num_keep_val,
        init_yubo_default,
        init_ax_default,
        default_num_X_samples,
    ):
        self.policy = policy
        self.num_arms = num_arms
        self.bt = bt
        self.num_keep = num_keep
        self.keep_style = keep_style
        self.num_keep_val = num_keep_val
        self.init_yubo_default = init_yubo_default
        self.init_ax_default = init_ax_default
        self.default_num_X_samples = default_num_X_samples
