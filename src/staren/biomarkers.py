"""Descriptive repeated-fit stability and priority-set comparisons."""
import numpy as np
import pandas as pd


def stable_features(models, min_frequency=.8, min_sign_consistency=.8, threshold=1e-6):
    """Prioritize by |median nonzero standardized coefficient| x frequency.

    models must have identical original feature names and layer assignments.
    All predictors are supplied to fitting. This is descriptive stability across
    refits, not a p-value or proof of biological/clinical validity.
    """
    if not models or not 0 <= min_frequency <= 1 or not 0 <= min_sign_consistency <= 1 or threshold < 0:
        raise ValueError("Supply fitted models and valid stability thresholds")
    names, groups = models[0].feature_names_in_, models[0].groups_
    if any(not np.array_equal(m.feature_names_in_, names) or not np.array_equal(m.groups_, groups) for m in models):
        raise ValueError("All refits must have matching feature names and layers")
    betas = np.array([m.coef_standardized_ for m in models])
    rows = []
    for j, name in enumerate(names):
        nonzero = betas[:, j][np.abs(betas[:, j]) > threshold]
        frequency = len(nonzero) / len(models)
        sign = max(np.mean(nonzero > 0), np.mean(nonzero < 0)) if len(nonzero) else 0.0
        median = float(np.median(nonzero)) if len(nonzero) else 0.0
        rows.append({"feature": name, "layer": groups[j], "selection_frequency": frequency,
                     "sign_consistency": sign, "median_standardized_coefficient": median,
                     "priority": abs(median)*frequency,
                     "stable": frequency >= min_frequency and sign >= min_sign_consistency})
    return pd.DataFrame(rows).sort_values("priority", ascending=False, kind="stable").reset_index(drop=True)


def priority_overlap(tables, top_n=50):
    """Boolean membership in each method's top stable set, suitable for UpSet.

    Absence is absence from this priority list, not necessarily a zero fitted
    coefficient. Different fitted models may produce different priority lists.
    """
    if not tables or not isinstance(top_n, int) or top_n < 1:
        raise ValueError("Supply named stability tables and positive top_n")
    sets = {name: set(table.loc[table["stable"]].sort_values("priority", ascending=False).head(top_n)["feature"])
            for name, table in tables.items()}
    names = sorted(set().union(*sets.values()))
    return pd.DataFrame({name: [f in features for f in names] for name, features in sets.items()},
                        index=pd.Index(names, name="feature"))

