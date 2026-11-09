# STAR-EN

### Source-Anchored Transfer and Adaptive Regularization with Elastic Net

**Multi-omics prediction and risk stratification for underrepresented populations.**

[**Tutorial website**](https://source-anchored-transfer-tutorial.prmjppx7jh.chatgpt.site/) |
[**Installation**](#installation) |
[**Four worked examples**](https://source-anchored-transfer-tutorial.prmjppx7jh.chatgpt.site/#examples) |
[API reference](docs/api.md) |
[Download source](dist/star_en-0.2.0.tar.gz)

## Overview

![STAR-EN main workflow: SAT-EN and LSAT-EN prediction and risk stratification](assets/star-en-main-workflow.svg)

STAR-EN is a Python package for prediction and risk stratification in
underrepresented populations with limited target-cohort data. It combines
**SAT-EN**, which uses elastic-net regularization on target deviations from a
source coefficient anchor, and **LSAT-EN**, which separately adapts borrowing
strength across molecular layers. Both methods support **binary logistic**
and **Cox time-to-event** prediction. Their held-out target predictions can
then be used for low- and high-risk grouping, restricted mean survival or
event-free time (RMST) comparisons, and correlations with pathway scores and
miRNA abundance. The package provides functions to export the underlying
tables and generate both risk/RMST and molecular-correlation figures.

The main figure is the same figure shown on the
[STAR-EN tutorial website](https://source-anchored-transfer-tutorial.prmjppx7jh.chatgpt.site/#overview).
Its embedded mini-plots are saved Overall RSF illustrations of downstream
outputs; they are not LSAT-EN fitted results.

## Installation

Python 3.10 or later is required. Install the complete **STAR-EN** package:

```bash
python -m pip install "star-en[analysis,survival] @ git+https://github.com/wguo3773/LSAT-EN.git"
```

The repository retains its **LSAT-EN** URL, but the distribution is
`star-en` and the public import is **`staren`**. Both SAT-EN and LSAT-EN are
included; no separate tutorial adapter is needed. Existing `lsaten` imports
remain compatible.

Verify all four estimator imports:

```python
from staren import SATBinaryCV, SATCoxCV, LSATBinaryCV, LSATCoxCV
```

The `analysis` extra supplies lifelines and matplotlib for RMST and plotting.
The `survival` extra supplies scikit-survival for C-index and IPCW AUC.
NumPy, SciPy, pandas and scikit-learn support the core estimators.

## Two Parts, Both Methods

| Task | SAT-EN | LSAT-EN |
|---|---|---|
| Binary prediction | `SATBinary`, `SATBinaryCV` | `LSATBinary`, `LSATBinaryCV` |
| Survival prediction | `SATCox`, `SATCoxCV` | `LSATCox`, `LSATCoxCV` |
| Risk groups and RMST figure | `risk_scores`, `stratify_risk`, `plot_risk_stratification` | Same functions |
| Pathway and miRNA figure | `pathway_scores`, `correlate_programs`, `plot_molecular_correlations` | Same functions |

Both downstream workflows accept binary event probabilities or Cox risk
scores. **RMST requires original follow-up times and event indicators**, even
when the model is binary. Pathway correlations require patient-aligned
expression or precomputed pathway scores.

Follow the [online examples](https://source-anchored-transfer-tutorial.prmjppx7jh.chatgpt.site/#examples)
for prediction and risk stratification with each method, or read the
[four-case Python guide](docs/star-workflows.md). The runnable demonstration
generates predictions, tables, and both figure types for both methods and
both outcomes:

```bash
git clone https://github.com/wguo3773/LSAT-EN.git
cd LSAT-EN
python -m pip install -e ".[analysis,survival]"
python examples/star_workflows.py --output star_outputs
```

The demonstration uses small synthetic data and toy gene sets, not patient
study results. The [tutorial's saved clinical figures](https://source-anchored-transfer-tutorial.prmjppx7jh.chatgpt.site/#examples)
are shown with their model and analysis provenance.

**All supplied features enter fitting; there is no separate SIS or feature-budget
selection step.** LSAT-EN can set final coefficients to zero. SAT-EN sparsity
applies to deviations from the source, not necessarily final coefficients.
No held-out target patients enter fitting, inner-fold preprocessing, or
source-anchor estimation. The API is not limited to six layers, breast
cancer, or age-defined cohorts.

## 1. Binary Prediction

```python
from staren import LSATBinaryCV, make_binary_transfer, binary_metrics

data = make_binary_transfer(
    n_source=180, n_target=90, n_test=150,
    n_features=36, block_size=6, sigma2_tl=0.25, random_state=42,
)
model = LSATBinaryCV(
    lambdas=(0.02, 0.1), alphas=(0.5,), tau_grid=(0.0, 0.5, 2.0),
    cv=3, coordinate_passes=1,
)
model.fit(
    data["X_source"], data["y_source"],
    data["X_target"], data["y_target"], groups=data["groups"],
)
probability = model.predict_proba(data["X_test"])[:, 1]
print(binary_metrics(data["y_test"], probability))
print(model.best_params_)
print(model.feature_table().head())
```

All 36 predictors are retained as model inputs. `groups` assigns one layer
label to every feature. Real datasets can use labels such as `mRNA`, `miRNA`,
`methylation`, `CNV`, `protein`, and `mutation`.

`predict_proba(X)[:,1]` always predicts **label 1**. For fixed-horizon
survival, encode 1 as survival through the horizon and 0 as death by it.
Patients censored before that horizon must be excluded from binary fitting;
[`binary_at_horizon()`](docs/api.md) provides this conversion.

## 2. Cox Survival Prediction

```python
from staren import LSATCoxCV, make_survival_transfer, survival_metrics

data = make_survival_transfer(
    n_source=140, n_target=90, n_test=100,
    n_features=36, block_size=6, random_state=43,
)
model = LSATCoxCV(
    lambdas=(0.05, 0.15), alphas=(0.5,), tau_grid=(0.0, 0.5),
    cv=3, coordinate_passes=1,
)
model.fit(
    data["X_source"], data["y_source"],
    data["X_target"], data["y_target"], groups=data["groups"],
)
risk = model.predict(data["X_test"])  # higher = higher event risk
print(survival_metrics(
    data["y_target"], data["y_test"], risk,
    times=(2.0, 3.0, 4.0, 5.0), on_error="nan",
))
survival_probability = model.predict_survival_function(
    data["X_test"], times=[2.0, 3.0, 4.0, 5.0],
)
```

Synthetic follow-up is in **years**. For actual data measured in days, use
matching day-valued evaluation times. `survival_outcome(time,event)` creates
the required structured labels, with event=1 for an observed death/event.
Target-training outcomes estimate the baseline hazard and censoring weights.
Unsupported IPCW evaluations raise by default, or return NaN **with the
reason** when `on_error="nan"`; missing AUC is never replaced with zero.

## Inputs and Outputs

| Input | Meaning |
|---|---|
| `X_source`, `X_target` | Numeric matrices or matching DataFrames; rows are patients, columns are all supplied features |
| `y_source`, `y_target` | Binary 0/1, or structured event/time outcomes |
| `groups` | One layer label per feature, in column order |
| `feature_names` | Optional unique feature names; inferred from DataFrame columns |
| `lambdas`, `alphas` | Candidate elastic-net strengths and L1 fractions |
| `tau_grid` | LSAT-EN candidate layer weights, shared sequence or layer-to-sequence dictionary |
| `source_lam`, `source_alpha` | Fixed source-anchor regularization settings |
| `cv`, `random_state` | Target inner folds and fold reproducibility |

| Output | Meaning |
|---|---|
| `best_params_` | Selected lambda and alpha; layer-specific tau for LSAT-EN |
| `cv_results_` | Tried settings, fold losses, mean validation loss and convergence |
| `feature_table()` | Every input feature, coefficients and source anchor; SAT deviations or LSAT tau |
| `n_features_in_`, `n_nonzero_` | Number supplied and number with absolute standardized coefficient >1e-6 |
| `coef_`, `coef_standardized_` | Coefficients in input and standardized units |
| `predict_proba()` | Binary class probabilities |
| `predict()` | Binary class or Cox relative log hazard |
| `predict_survival_function()` | Cox survival probabilities, patients x requested times |
| `converged_` | Source/target optimizer convergence status |

**Default CV settings:** lambdas=(0.01,0.1), alphas=(0.5,),
tau_grid=(0,0.1,1), cv=5, source_lam=0.05, source_alpha=0.5.
The default coordinate tau search is practical but not a guaranteed global
grid optimum. Set `tau_search="grid"` for an exhaustive Cartesian search
when feasible. Nonconverged candidates cannot win CV. Use fixed-parameter
`LSATBinary` / `LSATCox` when tuning is not needed. [Full API](docs/api.md)

## 3. Fair Comparisons and Repeated Target Splits

```python
overall = LSATBinaryCV(strategy="pooled", cv=3)
individual = LSATBinaryCV(strategy="target", cv=3)
```

Fit each method on the same source/target-training inputs and use identical
outer test patients. Each baseline can select its own lambda and alpha from
the same grids using target validation folds. The target-only strategy also
uses target-only preprocessing. No feature screening occurs in any strategy.

[`examples/repeated_target_splits.py`](examples/repeated_target_splits.py)
shows 10 matched 70/30 outer target splits with inner CV. Do not choose
parameters or report only the most favorable outer split. Repeated splits
share patients: their SD describes split variability, not independent-cohort
uncertainty. Source/target definitions, end points and time units are user inputs.

## 4. Biomarker Prioritization

```python
from staren import stable_features, priority_overlap

# repeated_models: independently refitted models with the same input columns
stability = stable_features(
    repeated_models, min_frequency=0.8, min_sign_consistency=0.8,
)
top_markers = stability.query("stable").head(50)
# compare = priority_overlap({"LSAT": stability, "Overall": overall_stability})
```

Priority is absolute median nonzero standardized coefficient x selection
frequency. Positive binary coefficients increase model-predicted odds of
label 1; positive Cox coefficients increase event hazard, conditional on
other predictors. Priority-list absence is not necessarily a zero coefficient.
Markers remain candidates, not validated mechanisms or treatment predictors.
The package does not bundle miRDB data, infer regulatory interactions from
coefficient signs, or automatically establish pathway enrichment.

## Example Outputs

These are **actual synthetic-data results** from
[`examples/quickstart.py`](examples/quickstart.py), using all 36 predictors
in each method. They illustrate outputs, not expected superiority in every
dataset and not a reproduction of the manuscript tables.

| Binary method | Test AUC | Test log loss | Nonzero coefficients |
|---|---:|---:|---:|
| LSAT-EN | 0.9287 | 0.3381 | 33 |
| Overall EN | 0.9177 | 0.3639 | 25 |
| Individual EN | 0.9408 | 0.3182 | 30 |

| Cox method | C-index | 2-year AUC | 3-year AUC | 4-year AUC | 5-year AUC |
|---|---:|---:|---:|---:|---:|
| LSAT-Cox-EN | 0.9033 | 0.9697 | 0.9636 | 0.9699 | 0.9749 |
| Overall Cox-EN | 0.9033 | 0.9577 | 0.9491 | 0.9657 | 0.9715 |
| Individual Cox-EN | 0.9048 | 0.9711 | 0.9660 | 0.9722 | 0.9782 |

[Binary metrics CSV](docs/example-output/binary-metrics.csv) |
[Cox metrics CSV](docs/example-output/cox-metrics.csv) |
[Binary coefficients](docs/example-output/binary-features.csv) |
[Cox coefficients](docs/example-output/cox-features.csv) |
[Selected binary parameters](docs/example-output/binary-best-parameters.json).

## Simulation Figure and Settings

![Manuscript illustration: binary simulations across source-target heterogeneity](assets/manuscript-binary-simulation.png)

The figure summarizes the manuscript's binary simulation benchmark. Each
panel varies the source-target coefficient-shift **variance**, not its SD:
A, 0; B, 0.25; C, 1; D, 4.

## Documentation and Validation

- [Tutorial: real input layouts, fitting and prediction](docs/tutorial.md).
- [API reference](docs/api.md) and [objective/CV specification](docs/methods.md).
- [Validation summary](docs/validation.md) and [test suite](tests/test_lsaten.py).
- [Contributing](CONTRIBUTING.md), [license](LICENSE), [citation metadata](CITATION.cff).
- [scikit-survival IPCW AUC](https://scikit-survival.readthedocs.io/en/stable/api/generated/sksurv.metrics.cumulative_dynamic_auc.html).

The package checks analytic gradients, limiting cases, fold-local preprocessing
and predictor retention. Its unanchored ridge cases are compared against
scikit-learn logistic regression and scikit-survival Cox regression.
Clinical use requires independent validation and appropriate study oversight.

## Citation

Please cite the method references listed on the [STAR-EN tutorial website](https://source-anchored-transfer-tutorial.prmjppx7jh.chatgpt.site/):

1. Guo, W., & Tatsuoka, C. (2026). Transfer learning improves multi-omics survival prediction for black patients with pediatric AML using white target-AML patients as the source cohort. *Blood* supplement (forthcoming). ASH 2026, Publication No. 2581.

2. Guo, W., Wang, B., & Tatsuoka, C. (2026). Layer-Specific Source-Anchored Transfer Learning for Multi-Omics Survival Prediction in Younger Breast Cancer Patients. *bioRxiv*. [https://doi.org/10.64898/2026.11.01.763494](https://doi.org/10.64898/2026.11.01.763494).
