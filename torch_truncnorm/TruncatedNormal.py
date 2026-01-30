import math
from numbers import Number
from typing import NamedTuple

import torch
from torch.distributions import Distribution, constraints
from torch.distributions.utils import broadcast_all

CONST_SQRT_2 = math.sqrt(2)
CONST_INV_SQRT_2PI = 1 / math.sqrt(2 * math.pi)
CONST_INV_SQRT_2 = 1 / math.sqrt(2)
CONST_LOG_INV_SQRT_2PI = math.log(CONST_INV_SQRT_2PI)
CONST_LOG_SQRT_2PI_E = 0.5 * math.log(2 * math.pi * math.e)


class _PiAndX(NamedTuple):
    pi: torch.Tensor
    X: torch.Tensor


class TruncatedStandardNormal(Distribution):
    """
    Truncated Standard Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    arg_constraints = {
        "a": constraints.real,
        "b": constraints.real,
    }
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        assert a.device == b.device, (a.device, b.device)
        assert a.dtype == b.dtype, (a.dtype, b.dtype)
        self.device = a.device
        self.dtype = a.dtype
        self.a, self.b = broadcast_all(a, b)
        if isinstance(a, Number) and isinstance(b, Number):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super(TruncatedStandardNormal, self).__init__(
            batch_shape, validate_args=validate_args
        )
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        if any(
            (self.a >= self.b)
            .view(
                -1,
            )
            .tolist()
        ):
            raise ValueError("Incorrect truncation range")
        eps = torch.finfo(self.a.dtype).eps
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        little_phi_coeff_a = torch.nan_to_num(self.a, nan=math.nan)
        little_phi_coeff_b = torch.nan_to_num(self.b, nan=math.nan)
        self._lpbb_m_lpaa_d_Z = (
            self._little_phi_b * little_phi_coeff_b
            - self._little_phi_a * little_phi_coeff_a
        ) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._variance = (
            1
            - self._lpbb_m_lpaa_d_Z
            - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        )
        self._entropy = CONST_LOG_SQRT_2PI_E + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @constraints.dependent_property
    def support(self):
        return constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    @property
    def auc(self):
        return self._Z

    @staticmethod
    def _little_phi(x):
        return (-(x**2) * 0.5).exp() * CONST_INV_SQRT_2PI

    @staticmethod
    def _big_phi(x):
        return 0.5 * (1 + (x * CONST_INV_SQRT_2).erf())

    @staticmethod
    def _inv_big_phi(x):
        return CONST_SQRT_2 * (2 * x - 1).erfinv()

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return CONST_LOG_INV_SQRT_2PI - self._log_Z - (value**2) * 0.5

    def rsample(self, sample_shape=torch.Size()):
        return self.p_and_sample(sample_shape)[1]

    def p_and_sample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        # TODO: Try Sobol instead of uniform
        p = torch.empty(shape, device=self.device, dtype=self.dtype).uniform_(
            self._dtype_min_gt_0, self._dtype_max_lt_1
        )
        return _PiAndX(p, self.icdf(p))


class TruncatedNormal:
    """
    Truncated Normal distribution
    https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    """

    has_rsample = True

    # TODO: Contain, DRY

    def __init__(self, loc, scale, a, b, validate_args=None, rng=None):
        self.loc, self.scale, a, b = broadcast_all(loc, scale, a, b)
        a = (a - self.loc) / self.scale
        b = (b - self.loc) / self.scale
        self._tn = TruncatedStandardNormal(a, b, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._tn._mean * self.scale + self.loc
        self._variance = self._tn._variance * self.scale**2
        self._entropy = self._tn._entropy + self._log_scale
        self._rng = rng

    def _to_std_rv(self, value):
        return (value - self.loc) / self.scale

    def _from_std_rv(self, value):
        return value * self.scale + self.loc

    def cdf(self, value):
        return self._tn.cdf(self._to_std_rv(value))

    def icdf(self, value):
        return self._from_std_rv(self._tn.icdf(value))

    def log_prob(self, value):
        return self._tn.log_prob(self._to_std_rv(value)) - self._log_scale

    def rsample(self, sample_shape=torch.Size()):
        return self.p_and_sample(sample_shape).X

    def p_and_sample(self, sample_shape=torch.Size()):
        shape = self._tn._extended_shape(sample_shape)
        # TODO: Try Sobol instead of uniform
        p = torch.empty(shape, device=self._tn.device, dtype=self._tn.dtype).uniform_(
            self._tn._dtype_min_gt_0,
            self._tn._dtype_max_lt_1,
            generator=self._rng,
        )
        return _PiAndX(p, self.icdf(p))
