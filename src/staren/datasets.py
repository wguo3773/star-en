"""Synthetic source/target datasets; no patient records are distributed."""
import numpy as np
from scipy.special import expit
from .outcomes import survival_outcome


def _design(n_source, n_target, n_test, n_features, block_size,
            within_correlation, between_correlation, sigma2_tl, random_state):
    for n in (n_source, n_target, n_test, n_features, block_size):
        if not isinstance(n, (int, np.integer)) or n < 2:
            raise ValueError("Sample sizes, n_features and block_size must be integers >=2")
    if not 0 <= between_correlation <= within_correlation < 1:
        raise ValueError("Require 0 <= between_correlation <= within_correlation < 1")
    if not np.isfinite(sigma2_tl) or sigma2_tl < 0:
        raise ValueError("sigma2_tl is a nonnegative coefficient-shift variance")
    rng = np.random.default_rng(random_state)
    blocks = np.arange(n_features) // block_size
    def draw(n):
        shared = rng.normal(size=(n, 1))
        block = rng.normal(size=(n, blocks.max()+1))[:, blocks]
        return (np.sqrt(between_correlation)*shared
                + np.sqrt(within_correlation-between_correlation)*block
                + np.sqrt(1-within_correlation)*rng.normal(size=(n, n_features)))
    xs, xt, xv = draw(n_source), draw(n_target), draw(n_test)
    effects = np.r_[np.ones(3), np.full(3, .7), np.full(4, .4)]
    bs = np.zeros(n_features)
    bs[:min(10, n_features)] = effects[:min(10, n_features)]
    bt = bs + rng.normal(scale=np.sqrt(sigma2_tl), size=n_features)
    return rng, xs, xt, xv, bs, bt, np.array([f"layer{g}" for g in blocks])


def make_binary_transfer(n_source=300, n_target=80, n_test=250, n_features=500,
                         block_size=10, within_correlation=.50,
                         between_correlation=.05, sigma2_tl=.25, random_state=42):
    """Block-correlated logistic simulation with beta_T-beta_S ~ N(0,sigma2_tl).

    Default coefficients/correlation follow the manuscript's stated binary
    generating design, but random streams and fitted pipelines are not a
    reproduction of the saved manuscript estimates. y=1 is the positive class.
    """
    rng, xs, xt, xv, bs, bt, groups = _design(
        n_source, n_target, n_test, n_features, block_size,
        within_correlation, between_correlation, sigma2_tl, random_state)
    return dict(X_source=xs, y_source=rng.binomial(1, expit(xs @ bs)),
                X_target=xt, y_target=rng.binomial(1, expit(xt @ bt)),
                X_test=xv, y_test=rng.binomial(1, expit(xv @ bt)),
                groups=groups, beta_source=bs, beta_target=bt)


def make_survival_transfer(n_source=180, n_target=100, n_test=100, n_features=30,
                           block_size=10, within_correlation=.50,
                           between_correlation=.05, sigma2_tl=.10,
                           baseline_rate=.15, censoring_rate=.06, random_state=42):
    """Illustrative exponential proportional-hazards data, with times in years.

    This is a software demonstration, not a manuscript Cox simulation result.
    """
    if not np.isfinite(baseline_rate) or baseline_rate <= 0 or not np.isfinite(censoring_rate) or censoring_rate <= 0:
        raise ValueError("baseline_rate and censoring_rate must be positive")
    rng, xs, xt, xv, bs, bt, groups = _design(
        n_source, n_target, n_test, n_features, block_size,
        within_correlation, between_correlation, sigma2_tl, random_state)
    def outcome(X, beta):
        event_time = rng.exponential(size=len(X)) / (baseline_rate*np.exp(np.clip(X @ beta, -30, 30)))
        censor = rng.exponential(1/censoring_rate, len(X))
        return survival_outcome(np.minimum(event_time, censor), event_time <= censor)
    return dict(X_source=xs, y_source=outcome(xs, bs),
                X_target=xt, y_target=outcome(xt, bt),
                X_test=xv, y_test=outcome(xv, bt),
                groups=groups, beta_source=bs, beta_target=bt)

