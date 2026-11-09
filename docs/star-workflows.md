# STAR-EN: Four Workflows

```bash
python -m pip install "star-en[analysis,survival] @ git+https://github.com/wguo3773/LSAT-EN.git"
```

STAR-EN is the distribution; `staren` is the import. It contains SAT-EN and
LSAT-EN, not a separately downloaded tutorial adapter. Existing `lsaten`
imports remain supported.

## 1. SAT-EN Prediction

```python
from staren import SATBinaryCV, SATCoxCV

# Source/target matrices must have identical columns in the same order.
# y_binary: 0/1 labels. y_survival: survival_outcome(time, event).
binary = SATBinaryCV(cv=5, lambdas=(.01, .1), alphas=(.5, 1.))
binary.fit(X_source, y_source_binary, X_train, y_train_binary, groups=layers)
probability_class1 = binary.predict_proba(X_test)[:, 1]

survival = SATCoxCV(cv=5, lambdas=(.01, .1), alphas=(.5, 1.))
survival.fit(X_source, y_source_survival, X_train, y_train_survival, groups=layers)
log_hazard = survival.predict(X_test)
survival_probability = survival.predict_survival_function(X_test, times=[2, 3, 4, 5])
```

SAT-EN minimizes target likelihood plus an EN penalty on
`delta = beta_target - beta_source`. There is **no tau**. Zero delta means
retaining a source coefficient, not removing that predictor. Source and target
intercepts/baseline hazards are not transferred.

## 2. LSAT-EN Prediction

```python
from staren import LSATBinaryCV, LSATCoxCV

binary = LSATBinaryCV(cv=5, lambdas=(.01,.1), alphas=(.5,), tau_grid=(0.,.1,1.))
binary.fit(X_source, y_source_binary, X_train, y_train_binary, groups=layers)
probability_class1 = binary.predict_proba(X_test)[:,1]

survival = LSATCoxCV(cv=5, lambdas=(.01,.1), alphas=(.5,), tau_grid=(0.,.1,1.))
survival.fit(X_source, y_source_survival, X_train, y_train_survival, groups=layers)
log_hazard = survival.predict(X_test)
```

LSAT-EN penalizes final target coefficients with EN and separately anchors
them to source coefficients using layer-specific tau. It is not the same
objective as SAT-EN. Both methods use all supplied predictors without SIS.

## 3. SAT-EN Risk Stratification and Both Figures

```python
import numpy as np
from staren import (risk_scores, stratify_risk, plot_risk_stratification,
    pathway_scores, correlate_programs, plot_molecular_correlations)

# Use the fitted SAT Cox model above; only held-out target predictions below.
risk = risk_scores(survival, X_test)
cutoff = float(np.median(risk_scores(survival, X_train)))
result = stratify_risk(risk, y_test_survival, horizon=5, cutoff=cutoff,
    subgroups={"Female": female_mask}, n_bootstrap=1000, random_state=42)
print(result["composition"])
print(result["rmst"])
plot_risk_stratification(result).savefig("sat_risk_rmst.pdf", bbox_inches="tight")

# Expression rows must align exactly with X_test. Gene sets map names to genes.
scores = pathway_scores(expression_test, hallmark_gene_sets, reference=expression_train)
pathways = correlate_programs(risk, scores)
mirnas = correlate_programs(risk, mirna_test)
pathways.to_csv("sat_pathways.csv", index=False)
mirnas.to_csv("sat_mirnas.csv", index=False)
plot_molecular_correlations(pathways, mirnas, threshold=.05, significance="p").savefig(
    "sat_pathways_mirna.pdf", bbox_inches="tight")
```

For a **binary** model, substitute:

```python
# When binary_at_horizon encoded event=0 and event-free=1:
risk = risk_scores(binary, X_test, event_class=0)
cutoff = .5
# The same stratify_risk and molecular functions then apply.
# Keep original y_test_survival for RMST, including censoring information.
```

## 4. LSAT-EN Risk Stratification and Both Figures

Use exactly the same functions, with the fitted LSAT model from Case 2:

```python
risk = risk_scores(survival, X_test)
cutoff = float(np.median(risk_scores(survival, X_train)))
result = stratify_risk(risk, y_test_survival, horizon=5, cutoff=cutoff, n_bootstrap=1000)
plot_risk_stratification(result).savefig("lsat_risk_rmst.pdf", bbox_inches="tight")
pathways = correlate_programs(risk, scores)
mirnas = correlate_programs(risk, mirna_test)
plot_molecular_correlations(pathways, mirnas, significance="q", threshold=.05).savefig(
    "lsat_pathways_mirna.pdf", bbox_inches="tight")
# Binary LSAT alternative: risk_scores(binary, X_test, event_class=0), cutoff=.5.
```

## Run a Complete Example

The [runnable example](../examples/star_workflows.py) defines every input and
executes **both methods x both endpoints**, saving predictions, coefficients,
composition, RMST, correlation tables, and two PDF/PNG figures per combination:

```bash
python examples/star_workflows.py --output star_outputs
```

These are small synthetic data and toy gene sets, not clinical study results.
Replace them with harmonized, aligned cohort matrices and real gene sets.

## Interpretation

- Binary risk direction must explicitly specify the event class. Cox risk is
  relative log hazard, not an absolute probability unless horizon is supplied.
- Training-derived cutoffs prevent test-outcome threshold selection. A test
  median split is descriptive, not validation of a deployable cutoff.
- RMST differences use Kaplan-Meier curves, so the downstream contrast does
  not require proportional hazards. Positive low-minus-high means longer
  restricted mean event-free time for Low. OS instead measures survival time.
- RMST CIs are conditional percentile patient bootstraps, not refitted-model
  CIs. Unsupported small groups/follow-up are NE. Bootstrap failures are counted.
- Donut fractions describe observed events, not censoring-adjusted incidence.
- Pathway scores are means of training-reference gene z-scores, not GSVA.
  Correlations are descriptive and can be partly induced by shared predictors.
  Both p and BH q values are exported; marker association is not causality.
- Across repeats, use `aggregate_heldout_predictions` on held-out-only records
  with comparable score scales. Each patient is represented once downstream.
