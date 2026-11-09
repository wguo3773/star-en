"""Source-anchored EN: elastic-net regularization of target deviations."""
import numpy as np
from scipy.optimize import minimize
from .models import LSATBinary, LSATCox, LSATBinaryCV, LSATCoxCV, _nonnegative
from ._loss import loss_gradient


class _Deviation:
    def _fit_prepared(self, prepared, ys, yt, groups, params):
        _, _, _, X, anchor, _ = prepared
        lam, alpha = params['lam'], params['alpha']
        _nonnegative(lam, 'lam')
        if not np.isfinite(alpha) or not 0 <= alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        p, binary = X.shape[1], self.endpoint == 'binary'
        initial = np.zeros(2*p + int(binary))
        if binary:
            prevalence = np.clip(np.mean(yt), 1e-8, 1-1e-8)
            initial[-1] = np.log(prevalence/(1-prevalence)) - np.mean(X @ anchor)

        def objective(v):
            delta = v[:p] - v[p:2*p]
            loss, grad, grad0 = loss_gradient(X, yt, anchor+delta,
                v[-1] if binary else 0., self.endpoint)
            loss += lam*alpha*np.sum(v[:2*p]) + lam*(1-alpha)*(delta@delta)/2
            grad += lam*(1-alpha)*delta
            full = np.r_[grad+lam*alpha, -grad+lam*alpha]
            return float(loss), np.r_[full, grad0] if binary else full

        result = minimize(objective, initial, jac=True, method='L-BFGS-B',
            bounds=[(0,None)]*(2*p) + ([(None,None)] if binary else []),
            options={'maxiter':self.max_iter, 'gtol':self.tol, 'ftol':1e-12, 'maxls':50})
        return anchor+result.x[:p]-result.x[p:2*p], float(result.x[-1]) if binary else 0., result

    def _finalize(self, *args):
        super()._finalize(*args)
        self.best_params_.pop('tau', None)
        if hasattr(self, 'cv_results_'):
            self.cv_results_ = self.cv_results_.drop(columns='tau', errors='ignore')
        self.deviation_standardized_ = self.coef_standardized_ - self.anchor_coef_standardized_
        return self

    def feature_table(self):
        # Parent table uses tau internally, which is absent from SAT's public settings.
        import pandas as pd
        from sklearn.utils.validation import check_is_fitted
        check_is_fitted(self, 'deviation_standardized_')
        return pd.DataFrame({'feature':self.feature_names_in_, 'layer':self.groups_,
            'coefficient':self.coef_, 'standardized_coefficient':self.coef_standardized_,
            'source_anchor':self.anchor_coef_standardized_,
            'standardized_deviation':self.deviation_standardized_,
            'nonzero':np.abs(self.coef_standardized_)>1e-6,
            'nonzero_deviation':np.abs(self.deviation_standardized_)>1e-6})


class SATBinary(_Deviation, LSATBinary):
    """Logistic SAT-EN: minimize target loss + EN(beta - source_anchor).

    No tau parameter. Sparsity is in deviations, not necessarily final beta.
    All supplied features enter fitting; groups annotate molecular layers.
    """
    def __init__(self, lam=.05, alpha=.5, source_lam=.05, source_alpha=.5,
                 max_iter=2000, tol=1e-6):
        super().__init__(lam=lam, alpha=alpha, tau=0., source_lam=source_lam,
                         source_alpha=source_alpha, max_iter=max_iter, tol=tol)


class SATCox(_Deviation, LSATCox):
    """Cox SAT-EN with EN-penalized deviations and target baseline hazard."""
    def __init__(self, lam=.05, alpha=.5, source_lam=.05, source_alpha=.5,
                 max_iter=2000, tol=1e-6):
        super().__init__(lam=lam, alpha=alpha, tau=0., source_lam=source_lam,
                         source_alpha=source_alpha, max_iter=max_iter, tol=tol)


class SATBinaryCV(_Deviation, LSATBinaryCV):
    """Target-fold CV of lambda/alpha for deviation-penalized logistic SAT-EN."""
    def __init__(self, lambdas=(.01,.1), alphas=(.5,), cv=5,
                 source_lam=.05, source_alpha=.5, max_iter=2000, tol=1e-6,
                 random_state=42):
        super().__init__(lambdas=lambdas, alphas=alphas, tau_grid=(0.,), cv=cv,
            source_lam=source_lam, source_alpha=source_alpha, max_iter=max_iter,
            tol=tol, random_state=random_state)


class SATCoxCV(_Deviation, LSATCoxCV):
    """Target-fold CV of lambda/alpha for deviation-penalized Cox SAT-EN."""
    def __init__(self, lambdas=(.01,.1), alphas=(.5,), cv=5,
                 source_lam=.05, source_alpha=.5, max_iter=2000, tol=1e-6,
                 random_state=42):
        super().__init__(lambdas=lambdas, alphas=alphas, tau_grid=(0.,), cv=cv,
            source_lam=source_lam, source_alpha=source_alpha, max_iter=max_iter,
            tol=tol, random_state=random_state)
