"""Tests for factor_lab.distributions — Student-t variance standardization.

The `student_t` sampler standardizes to UNIT VARIANCE by default so a nominal
scale/vol equals the realized standard deviation (raw `standard_t(df)` has
variance df/(df-2)). These tests pin that contract: unit variance by default,
the raw form on opt-out, loc honored on BOTH paths, df<=2 rejected when
standardizing, and a warning (not silent discard) when a scale is supplied
under standardize=True.
"""

import numpy as np
import pytest
from loguru import logger

from factor_lab.distributions import create_sampler

SEED = 20240621
# Sized for a tight-but-fast variance estimate. The binding case is the raw
# (un-standardized) df=5 draw: kurtosis 9 → Var(sample_var) ≈ 25/N, so std ≈ 0.005
# at N=1e6, leaving ~10σ against the rtol=0.03 (~0.05 abs) tolerance below. Smaller
# would still pass but trims margin; 4e6 (the original) just burned 4× the draws.
N = 1_000_000


def _sample(df, **kw):
    rng = np.random.default_rng(SEED)
    return create_sampler("student_t", rng, df=df, **kw)(N)


def _capture_warnings(fn):
    """Run ``fn`` and return the list of WARNING-level loguru records it emitted."""
    captured = []
    sink = logger.add(captured.append, level="WARNING")
    try:
        fn()
    finally:
        logger.remove(sink)
    return captured


class TestStudentTStandardization:

    # df=3 is intentionally excluded here: the standardized t with df=3 has
    # infinite 4th moment, so the sample-variance estimator itself has infinite
    # variance and the assertion is technically flaky even at large N. df>4 keeps
    # the estimator well-behaved; the df=3 unit-variance contract is still exercised
    # indirectly (it is a valid df>2 input that does not raise).
    @pytest.mark.parametrize("df", [5, 6, 30])
    def test_default_is_unit_variance(self, df):
        """standardize defaults to True: realized variance ~= 1 for df>4 (finite kurtosis)."""
        assert np.isclose(_sample(df).var(), 1.0, atol=0.02)

    def test_df3_standardizes_without_raising(self):
        """df=3 is a valid (df>2) standardize input — it must build/draw, not raise.

        Its variance is finite (=1 in distribution) but the *sample* variance is
        too heavy-tailed to assert tightly, so we only check the draw succeeds.
        """
        x = _sample(3)
        assert x.shape == (N,) and np.isfinite(x).all()

    @pytest.mark.parametrize("df", [5, 6, 30])
    def test_disabled_gives_raw_variance(self, df):
        """standardize=False reproduces the raw standard_t variance df/(df-2)."""
        assert np.isclose(_sample(df, standardize=False).var(),
                          df / (df - 2.0), rtol=0.03)

    def test_scale_is_overwritten_when_standardizing(self):
        """A user scale is overwritten in standardize mode — variance stays 1."""
        assert np.isclose(_sample(6, scale=10.0).var(), 1.0, atol=0.02)

    def test_standardizing_with_scale_warns(self):
        """The overwrite is loud, not silent: a scale under standardize=True logs a warning."""
        msgs = _capture_warnings(
            lambda: create_sampler("student_t", np.random.default_rng(SEED), df=6, scale=10.0)
        )
        assert any("ignores the provided scale" in str(m) for m in msgs)

    def test_no_warning_when_scale_absent(self):
        """No spurious warning when the caller does not pass a scale."""
        msgs = _capture_warnings(
            lambda: create_sampler("student_t", np.random.default_rng(SEED), df=6)
        )
        assert msgs == []

    def test_scale_applies_when_not_standardizing(self):
        """With standardize=False, scale multiplies: var = scale**2 * df/(df-2)."""
        df, scale = 6, 3.0
        assert np.isclose(_sample(df, scale=scale, standardize=False).var(),
                          scale ** 2 * df / (df - 2.0), rtol=0.03)

    def test_loc_shifts_mean_without_touching_variance(self):
        """loc is honored under standardization (old code dropped loc entirely)."""
        x = _sample(10, loc=5.0)
        assert np.isclose(x.mean(), 5.0, atol=0.02)
        assert np.isclose(x.var(), 1.0, atol=0.02)

    def test_loc_applies_on_raw_path_too(self):
        """loc is honored on the standardize=False path as well."""
        x = _sample(10, loc=5.0, standardize=False)
        assert np.isclose(x.mean(), 5.0, atol=0.02)

    @pytest.mark.parametrize("df", [1, 2, 1.5])
    def test_df_le_2_raises_when_standardizing(self, df):
        """Infinite variance cannot be normalized — fail loud."""
        with pytest.raises(ValueError, match="df > 2"):
            create_sampler("student_t", np.random.default_rng(SEED), df=df)

    def test_df_le_2_allowed_when_not_standardizing(self):
        """The raw heavy-variance draw is still reachable for df<=2."""
        x = _sample(2, standardize=False)
        assert x.shape == (N,)
