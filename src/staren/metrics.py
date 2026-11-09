"""Evaluation via established scikit-learn and scikit-survival routines."""
import numpy as np
from sklearn.metrics import roc_auc_score, log_loss
from .outcomes import validate_outcome


def binary_metrics(y, probability):
    """AUC and log loss for P(y=1); positive class must be stated by the user."""
    y = validate_outcome(y, "binary")
    p = np.asarray(probability, dtype=float)
    if p.shape != y.shape or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError("probability must match y and lie in [0,1]")
    return {"auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p))}


def survival_metrics(y_train, y_test, risk, times=(2., 3., 4., 5.), on_error="raise"):
    """Harrell C-index and horizon IPCW AUC using target-training censoring data.

    Higher risk means higher event risk. Time units must match the outcome.
    Support/censoring failures raise by default. on_error='nan' records each
    reason explicitly rather than clipping censoring weights or inventing AUCs.
    Requires the optional 'survival' extra.
    """
    try:
        from sksurv.metrics import concordance_index_censored, cumulative_dynamic_auc
    except ImportError as exc:
        raise ImportError("Install lsat-en[survival] for C-index and IPCW AUC") from exc
    if on_error not in ("raise", "nan"):
        raise ValueError("on_error must be raise or nan")
    train = validate_outcome(y_train, "cox", require_events=False)
    test = validate_outcome(y_test, "cox", require_events=False)
    risk = np.asarray(risk, dtype=float)
    times = np.asarray(times, dtype=float)
    if risk.shape != (len(test),) or np.any(~np.isfinite(risk)):
        raise ValueError("risk must be finite with one value per test patient")
    if times.ndim != 1 or not len(times) or np.any(~np.isfinite(times)) or np.any(times <= 0) or len(np.unique(times)) != len(times):
        raise ValueError("times must be distinct positive finite values")
    result = {"c_index": np.nan, "ipcw_auc": {}, "errors": {}}
    try:
        result["c_index"] = float(concordance_index_censored(test["event"], test["time"], risk)[0])
    except ValueError as exc:
        if on_error == "raise":
            raise
        result["errors"]["c_index"] = str(exc)
    for t in times:
        try:
            result["ipcw_auc"][float(t)] = float(cumulative_dynamic_auc(train, test, risk, [t])[0][0])
        except ValueError as exc:
            if on_error == "raise":
                raise
            result["ipcw_auc"][float(t)] = np.nan
            result["errors"][float(t)] = str(exc)
    return result

