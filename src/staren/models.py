"""Binary and Cox estimators with explicit source and target inputs."""
from collections.abc import Mapping
from itertools import product
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.validation import check_is_fitted
from ._loss import solve, loss_gradient
from ._preprocess import matrix, Preprocessor
from .outcomes import validate_outcome


def _nonnegative(value, name):
    if not np.isscalar(value) or not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be a finite nonnegative scalar")


def _tau_vector(tau, groups):
    if isinstance(tau, Mapping):
        missing = set(groups) - set(tau)
        if missing:
            raise ValueError(f"Missing tau values for layers: {missing}")
        values = np.array([tau[g] for g in groups], dtype=float)
    else:
        _nonnegative(tau, "tau")
        values = np.full(len(groups), tau, dtype=float)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("All tau values must be finite and nonnegative")
    return values


class _Model(BaseEstimator):
    """Shared fixed-parameter model. Use a CV subclass to select parameters."""
    def __init__(self, lam=.05, alpha=.5, tau=1.0, source_lam=.05,
                 source_alpha=.5, strategy="lsat",
                 max_iter=2000, tol=1e-6):
        self.lam, self.alpha, self.tau = lam, alpha, tau
        self.source_lam, self.source_alpha = source_lam, source_alpha
        self.strategy = strategy
        self.max_iter, self.tol = max_iter, tol

    def _input(self, X_source, y_source, X_target, y_target, groups, feature_names):
        if self.strategy not in ("lsat", "pooled", "target"):
            raise ValueError("strategy must be lsat, pooled or target")
        if hasattr(X_source, "columns") or hasattr(X_target, "columns"):
            if not (hasattr(X_source, "columns") and hasattr(X_target, "columns")):
                raise ValueError("Use DataFrames for both cohorts or arrays for both")
            if list(X_source.columns) != list(X_target.columns):
                raise ValueError("Source and target feature columns must match in order")
            if feature_names is None:
                feature_names = list(X_source.columns)
        xs, xt = matrix(X_source), matrix(X_target)
        ys, yt = validate_outcome(y_source, self.endpoint), validate_outcome(y_target, self.endpoint)
        if xs.shape[1] != xt.shape[1] or len(ys) != len(xs) or len(yt) != len(xt):
            raise ValueError("Cohort dimensions and outcomes must match")
        groups = np.asarray(groups, dtype=str)
        if groups.shape != (xs.shape[1],) or np.any(groups == ""):
            raise ValueError("groups must contain one nonempty layer label per feature")
        names = np.asarray(feature_names if feature_names is not None else
                           [f"x{i}" for i in range(xs.shape[1])], dtype=str)
        if names.shape != groups.shape or len(set(names)) != len(names):
            raise ValueError("feature_names must be unique, one per predictor")
        for value, label in [(self.source_lam, "source_lam"), (self.tol, "tol")]:
            _nonnegative(value, label)
        if not 0 <= self.source_alpha <= 1 or self.tol == 0:
            raise ValueError("source_alpha must be in [0,1] and tol positive")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        return xs, ys, xt, yt, groups, names

    def _prepare(self, xs, ys, xt, yt, groups):
        training_x = xt if self.strategy == "target" else np.vstack([xs, xt])
        pre = Preprocessor().fit(training_x)
        selected = np.arange(xs.shape[1])
        xss, xts = pre.transform(xs)[:, selected], pre.transform(xt)[:, selected]
        anchor = np.zeros(len(selected))
        source_result = None
        if self.strategy == "lsat":
            anchor, _, source_result = solve(
                xss, ys, self.endpoint, self.source_lam, self.source_alpha,
                np.zeros(len(selected)), anchor, self.max_iter, self.tol)
        return pre, selected, xss, xts, anchor, source_result

    def _fit_prepared(self, prepared, ys, yt, groups, params):
        _, selected, xss, xts, anchor, _ = prepared
        _nonnegative(params["lam"], "lam")
        if not np.isfinite(params["alpha"]) or not 0 <= params["alpha"] <= 1:
            raise ValueError("alpha must be in [0,1]")
        tau = (_tau_vector(params["tau"], groups[selected]) if self.strategy == "lsat"
               else np.zeros(len(selected)))
        X = np.vstack([xss, xts]) if self.strategy == "pooled" else xts
        y = np.concatenate([ys, yt]) if self.strategy == "pooled" else yt
        return solve(X, y, self.endpoint, params["lam"], params["alpha"],
                     tau, anchor, self.max_iter, self.tol)

    def _finalize(self, xs, ys, xt, yt, groups, names, params):
        prepared = self._prepare(xs, ys, xt, yt, groups)
        self.preprocessor_, selected, _, xts, anchor, src_result = prepared
        beta, intercept, result = self._fit_prepared(prepared, ys, yt, groups, params)
        self.n_features_in_ = xs.shape[1]
        self.feature_names_in_, self.groups_ = names, groups
        self.selected_indices_ = selected
        self.coef_standardized_ = np.zeros(self.n_features_in_)
        self.coef_standardized_[selected] = beta
        self.anchor_coef_standardized_ = np.zeros(self.n_features_in_)
        self.anchor_coef_standardized_[selected] = anchor
        self.coef_ = self.coef_standardized_ / self.preprocessor_.scales_
        self._intercept_standardized_ = intercept
        self.intercept_ = float(intercept - self.preprocessor_.means_ @ self.coef_)
        self.best_params_ = dict(params)
        self.n_nonzero_ = int(np.sum(np.abs(self.coef_standardized_) > 1e-6))
        self.optimization_result_ = result
        self.source_optimization_result_ = src_result
        self.converged_ = bool(result.success and (src_result is None or src_result.success))
        if not self.converged_:
            warnings.warn("An optimizer did not converge; inspect optimization_result_ and increase max_iter",
                          ConvergenceWarning, stacklevel=2)
        if self.endpoint == "binary":
            self.classes_ = np.array([0, 1])
        else:
            # Target-cohort calibration; source baseline risk is not transferred.
            eta = xts @ beta
            self.event_times_ = np.unique(yt["time"][yt["event"]])
            increments = [np.log(np.sum(yt["event"] & (yt["time"] == t)))
                          - logsumexp(eta[yt["time"] >= t]) for t in self.event_times_]
            self.log_baseline_cumulative_hazard_ = np.logaddexp.accumulate(increments)
        return self

    def fit(self, X_source, y_source, X_target, y_target, *, groups, feature_names=None):
        """Fit on source and target-training data; never supply held-out test data."""
        data = self._input(X_source, y_source, X_target, y_target, groups, feature_names)
        params = {"lam": self.lam, "alpha": self.alpha, "tau": self.tau}
        return self._finalize(*data, params)

    def decision_function(self, X):
        """Log odds of class 1 (binary), or relative log hazard (Cox)."""
        check_is_fitted(self, "coef_standardized_")
        if hasattr(X, "columns") and list(map(str, X.columns)) != list(self.feature_names_in_):
            raise ValueError("Prediction columns must match training order")
        X = matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count differs from training")
        return self.preprocessor_.transform(X) @ self.coef_standardized_ + self._intercept_standardized_

    def feature_table(self):
        """Coefficients, source anchors and layer weights for every supplied feature."""
        check_is_fitted(self, "coef_standardized_")
        tau = (_tau_vector(self.best_params_["tau"], self.groups_) if self.strategy == "lsat"
               else np.zeros(self.n_features_in_))
        return pd.DataFrame({"feature": self.feature_names_in_, "layer": self.groups_,
                             "nonzero": np.abs(self.coef_standardized_) > 1e-6,
                             "coefficient": self.coef_, "standardized_coefficient": self.coef_standardized_,
                             "source_anchor": self.anchor_coef_standardized_, "tau": tau})


class LSATBinary(_Model):
    """Logistic LSAT-EN with fixed settings and an unpenalized target intercept.

    tau: nonnegative scalar or layer->value mapping. All supplied features
    enter fitting; there is no separate screening step. strategy='lsat'
    fits the target likelihood with a source-only coefficient anchor;
    'pooled' and 'target' fit the corresponding unanchored EN baselines.
    """
    endpoint = "binary"

    def predict_proba(self, X):
        """Two columns: P(y=0), P(y=1). For survival labels, y=1 means alive."""
        p = expit(self.decision_function(X))
        return np.column_stack([1-p, p])

    def predict(self, X):
        return (self.decision_function(X) >= 0).astype(int)


class LSATCox(_Model):
    """Cox LSAT-EN with Breslow ties, event-normalized likelihood and no intercept."""
    endpoint = "cox"

    def predict(self, X):
        """Relative log hazard: higher means higher event risk."""
        return self.decision_function(X)

    def predict_survival_function(self, X, times):
        """Return n_patients x n_times survival probabilities on input time units.

        Baseline hazard is estimated from target training outcomes. Beyond the
        last training event the estimate is flat, not validated extrapolation.
        """
        eta = self.decision_function(X)
        times = np.asarray(times, dtype=float)
        if times.ndim != 1 or not len(times) or np.any(~np.isfinite(times)) or np.any(times < 0):
            raise ValueError("times must be a nonempty vector of finite nonnegative values")
        idx = np.searchsorted(self.event_times_, times, side="right") - 1
        log_hazard = np.full(len(times), -np.inf)
        valid = idx >= 0
        log_hazard[valid] = self.log_baseline_cumulative_hazard_[idx[valid]]
        return np.exp(-np.exp(np.minimum(eta[:, None] + log_hazard[None, :], 700)))


class _CV:
    def __init__(self, lambdas=(.01, .1), alphas=(.5,), tau_grid=(0., .1, 1.),
                 cv=5,
                 tau_search="coordinate", coordinate_passes=2, max_candidates=10000,
                 source_lam=.05, source_alpha=.5, strategy="lsat",
                 max_iter=2000, tol=1e-6, random_state=42):
        super().__init__(source_lam=source_lam, source_alpha=source_alpha,
                         strategy=strategy, max_iter=max_iter, tol=tol)
        self.lambdas, self.alphas, self.tau_grid = lambdas, alphas, tau_grid
        self.cv = cv
        self.tau_search, self.coordinate_passes = tau_search, coordinate_passes
        self.max_candidates, self.random_state = max_candidates, random_state

    def fit(self, X_source, y_source, X_target, y_target, *, groups, feature_names=None):
        """Select settings using target-fold validation; refit training data only.

        All predictors are retained. Imputation and scaling exclude the inner
        validation fold. Source anchors are fitted in that fold's coordinate
        system, then held fixed across target-parameter candidates.
        """
        data = self._input(X_source, y_source, X_target, y_target, groups, feature_names)
        xs, ys, xt, yt, groups, names = data
        strata = yt if self.endpoint == "binary" else yt["event"].astype(int)
        counts = np.bincount(strata.astype(int), minlength=2)
        if not isinstance(self.cv, (int, np.integer)) or self.cv < 2 or np.min(counts) < self.cv:
            raise ValueError("cv requires at least cv members of both classes/event strata; reduce cv explicitly")
        if self.tau_search not in ("coordinate", "grid"):
            raise ValueError("tau_search must be coordinate or grid")
        if not isinstance(self.coordinate_passes, int) or self.coordinate_passes < 1:
            raise ValueError("coordinate_passes must be a positive integer")
        if not isinstance(self.max_candidates, int) or self.max_candidates < 1:
            raise ValueError("max_candidates must be a positive integer")
        lambdas, alphas = list(self.lambdas), list(self.alphas)
        if not lambdas or not alphas:
            raise ValueError("Parameter grids cannot be empty")
        for lam in lambdas:
            _nonnegative(lam, "lambda grid")
        if any(not np.isfinite(a) or a < 0 or a > 1 for a in alphas):
            raise ValueError("alphas must lie in [0,1]")
        layers = list(dict.fromkeys(groups))
        if isinstance(self.tau_grid, Mapping) and set(self.tau_grid) != set(layers):
            raise ValueError("tau_grid map must specify every layer exactly")
        tau_options = {g: list(self.tau_grid[g] if isinstance(self.tau_grid, Mapping)
                               else self.tau_grid) for g in layers}
        for values in tau_options.values():
            if not values:
                raise ValueError("Tau grids cannot be empty")
            for t in values:
                _nonnegative(t, "tau grid")
        folds = list(StratifiedKFold(self.cv, shuffle=True, random_state=self.random_state).split(xt, strata))
        self.cv_splits_ = [(a.copy(), b.copy()) for a, b in folds]
        records, best = [], None
        self.fold_preprocessing_ = []
        prepared = [self._prepare(xs, ys, xt[a], yt[a], groups) for a, _ in folds]
        self.fold_preprocessing_ = [
            {"medians": p[0].medians_.copy(), "means": p[0].means_.copy(),
             "scales": p[0].scales_.copy()} for p in prepared]
        for lam, alpha in product(lambdas, alphas):
                cache = {}

                def evaluate(tau):
                    nonlocal best
                    key = tuple(tau[g] for g in layers)
                    if key in cache:
                        return cache[key]
                    if len(records) >= self.max_candidates:
                        raise ValueError("Search exceeds max_candidates; reduce the grid or use coordinate search")
                    losses, converged = [], True
                    params = {"lam": float(lam), "alpha": float(alpha), "tau": dict(tau)}
                    for (a, b), p in zip(folds, prepared):
                        beta, intercept, result = self._fit_prepared(p, ys, yt[a], groups, params)
                        val = p[0].transform(xt[b])[:, p[1]]
                        losses.append(loss_gradient(val, yt[b], beta, intercept, self.endpoint)[0])
                        converged &= bool(result.success and (p[5] is None or p[5].success))
                    mean_loss = float(np.mean(losses))
                    record = dict(params, mean_validation_loss=mean_loss,
                                  fold_losses=tuple(losses), converged=converged)
                    records.append(record)
                    cache[key] = mean_loss if converged else np.inf
                    if converged and np.isfinite(mean_loss) and (best is None or mean_loss < best[0]):
                        best = (mean_loss, params)
                    return mean_loss if converged else np.inf

                if self.strategy != "lsat":
                    evaluate(dict.fromkeys(layers, 0.0))
                elif self.tau_search == "grid":
                    if np.prod([len(tau_options[g]) for g in layers], dtype=float) > self.max_candidates:
                        raise ValueError("Cartesian tau grid exceeds max_candidates; use coordinate search")
                    for values in product(*(tau_options[g] for g in layers)):
                        evaluate(dict(zip(layers, values)))
                else:
                    current = {g: float(tau_options[g][0]) for g in layers}
                    evaluate(current)
                    for _ in range(self.coordinate_passes):
                        previous = dict(current)
                        for g in layers:
                            options = [(evaluate(dict(current, **{g: float(t)})), float(t))
                                       for t in tau_options[g]]
                            current[g] = min(options, key=lambda pair: pair[0])[1]
                        if current == previous:
                            break
        self.cv_results_ = pd.DataFrame(records)
        if best is None:
            raise RuntimeError("No converged CV candidate; increase max_iter or regularization")
        self.best_validation_loss_ = best[0]
        return self._finalize(*data, best[1])


class LSATBinaryCV(_CV, LSATBinary):
    """Training-only target-fold CV for binary LSAT-EN and EN baselines.

    See docs/api.md for all inputs. Coordinate tau search is the default;
    tau_search='grid' searches the full Cartesian product (potentially costly).
    All supplied predictors enter each fold; no screening or budget search.
    """


class LSATCoxCV(_CV, LSATCox):
    """Target-fold CV using event-normalized negative Cox partial likelihood.

    Event-stratified folds must contain both event strata. No validation
    outcomes participate in preprocessing, screening or source-anchor fitting.
    """

