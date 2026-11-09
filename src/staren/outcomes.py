"""Validated outcomes with explicit event and horizon conventions."""
import numpy as np


def survival_outcome(time, event):
    """Return structured (event, time) data; event=1 means observed event."""
    time, event = np.asarray(time, dtype=float), np.asarray(event)
    if time.ndim != 1 or event.shape != time.shape or len(time) < 2:
        raise ValueError("time and event must be matching vectors of length >=2")
    if not np.all(np.isfinite(time)) or np.any(time <= 0):
        raise ValueError("Follow-up times must be positive and finite")
    if not np.all(np.isin(event, [0, 1])):
        raise ValueError("event must be 0/1, with 1 denoting an observed event")
    out = np.empty(len(time), dtype=[("event", "?"), ("time", "f8")])
    out["event"], out["time"] = event.astype(bool), time
    return out


def validate_outcome(y, endpoint, require_events=True):
    y = np.asarray(y)
    if endpoint == "binary":
        if y.ndim != 1 or not np.all(np.isin(y, [0, 1])) or len(np.unique(y)) != 2:
            raise ValueError("Binary outcomes must contain both 0 and 1")
        return y.astype(float)
    if y.dtype.names is None or not {"time", "event"}.issubset(y.dtype.names):
        raise ValueError("Use survival_outcome(time, event) for Cox outcomes")
    y = survival_outcome(y["time"], y["event"])
    if require_events and not np.any(y["event"]):
        raise ValueError("Cox fitting or validation requires at least one event")
    return y


def binary_at_horizon(y, horizon):
    """Return (known-outcome mask, binary labels); 1 means survival through h.

    Events at or before h are 0. Event-free censoring at h or later is known
    survival through h. Censoring before h is excluded, never labeled alive.
    """
    y = validate_outcome(y, "cox", require_events=False)
    if not np.isfinite(horizon) or horizon <= 0:
        raise ValueError("horizon must be positive and use the same units as time")
    dead = y["event"] & (y["time"] <= horizon)
    known = dead | (y["time"] >= horizon)
    return known, (~dead[known]).astype(int)

