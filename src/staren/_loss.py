"""Convex likelihoods and an L-BFGS-B elastic-net solver."""
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, logsumexp


def logistic_loss_gradient(X, y, beta, intercept=0.0):
    eta = X @ beta + intercept
    residual = expit(eta) - y
    return (np.mean(np.logaddexp(0, eta) - y * eta),
            X.T @ residual / len(y), float(np.mean(residual)))


def cox_loss_gradient(X, y, beta):
    """Breslow tied-event partial likelihood, normalized by event count."""
    eta, time, event = X @ beta, y["time"], y["event"]
    count = np.sum(event)
    if count == 0:
        raise ValueError("Cox loss requires at least one observed event")
    loss, grad = 0.0, np.zeros(X.shape[1])
    for t in np.unique(time[event]):
        deaths, risk = event & (time == t), time >= t
        d = int(np.sum(deaths))
        log_denom = logsumexp(eta[risk])
        weights = np.exp(eta[risk] - log_denom)
        loss += d * log_denom - np.sum(eta[deaths])
        grad += d * (weights @ X[risk]) - np.sum(X[deaths], axis=0)
    return float(loss / count), grad / count, 0.0


def loss_gradient(X, y, beta, intercept, endpoint):
    fun = logistic_loss_gradient if endpoint == "binary" else cox_loss_gradient
    return fun(X, y, beta, intercept) if endpoint == "binary" else fun(X, y, beta)


def solve(X, y, endpoint, lam, alpha, tau, anchor, max_iter, tol):
    """Represent beta=u-v, u,v>=0, and solve using SciPy's bounded optimizer.

    This expresses the L1 penalty exactly without smoothing or rounding.
    The binary intercept is unpenalized; Cox has no intercept.
    """
    p = X.shape[1]
    binary = endpoint == "binary"
    initial = np.r_[np.maximum(anchor, 0), np.maximum(-anchor, 0)]
    if binary:
        prevalence = np.clip(np.mean(y), 1e-8, 1 - 1e-8)
        initial = np.r_[initial, np.log(prevalence / (1 - prevalence))]

    def objective(v):
        beta, intercept = v[:p] - v[p:2*p], v[-1] if binary else 0.0
        value, grad, grad0 = loss_gradient(X, y, beta, intercept, endpoint)
        diff = beta - anchor
        value += (lam * alpha * np.sum(v[:2*p])
                  + 0.5 * lam * (1-alpha) * (beta @ beta)
                  + 0.5 * np.sum(tau * diff**2))
        grad = grad + lam * (1-alpha) * beta + tau * diff
        full_grad = np.r_[grad + lam*alpha, -grad + lam*alpha]
        if binary:
            full_grad = np.r_[full_grad, grad0]
        return float(value), full_grad

    result = minimize(objective, initial, jac=True, method="L-BFGS-B",
                      bounds=[(0, None)]*(2*p) + ([(None, None)] if binary else []),
                      options={"maxiter": max_iter, "gtol": tol, "ftol": 1e-12,
                               "maxls": 50})
    return (result.x[:p] - result.x[p:2*p],
            float(result.x[-1]) if binary else 0.0, result)

