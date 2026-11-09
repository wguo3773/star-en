// Each displayed example uses the same public API as the downloadable script.
(() => {
  const encode = s => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
  const values={sat:{binary:['0.9970','0.1083'],cox:['not a binary AUC','0.1083']},
                lsat:{binary:['0.9673','0.1128'],cox:['not a binary AUC','0.1177']}};
  function code(method, endpoint, task) {
    const prefix=method==='sat'?'SAT':'LSAT';
    const modelClass=prefix+(endpoint==='binary'?'BinaryCV':'CoxCV');
    let text=`import numpy as np
import pandas as pd
from staren import (${modelClass}, make_survival_transfer,
    binary_at_horizon, binary_metrics, risk_scores, stratify_risk,
    pathway_scores, correlate_programs, plot_risk_stratification,
    plot_molecular_correlations)

# 1. Reproducible synthetic input; replace with your aligned cohort matrices.
data = make_survival_transfer(n_source=70, n_target=50, n_test=40,
    n_features=12, block_size=4, random_state=42)
xs, xt, xe = (data[k] for k in ['X_source', 'X_target', 'X_test'])
ys, yt, ye = (data[k] for k in ['y_source', 'y_target', 'y_test'])
horizon = 0.25  # synthetic time units; use supported clinical horizons

# 2. Fit using source and target training ONLY.
model = ${modelClass}(lambdas=(.05, .15), alphas=(.5,), cv=2${method==='lsat'?',\n    tau_grid=(0., .5), coordinate_passes=1':''})
`;
    if(endpoint==='binary') text+=`source_known, source_labels = binary_at_horizon(ys, horizon)
target_known, target_labels = binary_at_horizon(yt, horizon)
model.fit(xs[source_known], source_labels, xt[target_known], target_labels,
          groups=data['groups'])

# 3. Event risk is P(class 0), since class 1 means event-free.
risk = risk_scores(model, xe, event_class=0)
known, labels = binary_at_horizon(ye, horizon)
print(binary_metrics(labels, 1-risk[known])['auc'])
cutoff = float(np.median(risk_scores(model, xt[target_known], event_class=0)))
`;
    else text+=`model.fit(xs, ys, xt, yt, groups=data['groups'])

# 3. Cox risk: higher relative log hazard means greater event risk.
risk = risk_scores(model, xe)
survival_probabilities = model.predict_survival_function(xe, [horizon])
cutoff = float(np.median(risk_scores(model, xt)))
`;
    text+=`if not model.converged_:
    raise RuntimeError('Inspect convergence before interpreting outputs')
model.feature_table().to_csv('${method}_${endpoint}_coefficients.csv', index=False)
pd.DataFrame({'risk': risk}).to_csv('${method}_${endpoint}_test_risk.csv', index=False)
`;
    if(task==='risk') text+=`
# 4. Risk grouping + RMST: retain original time/event records for BOTH endpoints.
# Cutoff is a training median, not selected using test outcomes.
result = stratify_risk(risk, ye, horizon, cutoff=cutoff, n_bootstrap=30,
    subgroups={'Synthetic subgroup': np.arange(len(ye)) % 2 == 0})
print(result['composition'])
print(result['rmst'])
plot_risk_stratification(result).savefig('${method}_${endpoint}_risk_rmst.pdf',
    bbox_inches='tight')

# 5. Pathway + miRNA-style correlations. Demonstration sets are NOT Hallmarks.
# Replace with real gene sets and aligned expression/miRNA abundance matrices.
columns = [f'gene{i}' for i in range(xs.shape[1])]
training = pd.DataFrame(np.vstack([xs, xt]), columns=columns)
expression = pd.DataFrame(xe, columns=columns)
scores = pathway_scores(expression,
    {'Demonstration set A': columns[:4], 'Demonstration set B': columns[4:8]}, reference=training)
pathways = correlate_programs(risk, scores)
mirnas = correlate_programs(risk, expression[columns[8:]])
print(pathways[['program', 'rho', 'p', 'q']])
pathways.to_csv('${method}_${endpoint}_pathways.csv', index=False)
plot_molecular_correlations(pathways, mirnas, significance='p', threshold=1.).savefig(
    '${method}_${endpoint}_molecular.pdf', bbox_inches='tight')
# threshold=1 shows all demonstration programs; use a prespecified p/q threshold for studies.
`;
    return text;
  }
  document.querySelectorAll('[data-case]').forEach(container => {
    const [method, task]=container.dataset.case.split('-');
    let endpoint=task==='risk'?'cox':'binary';
    function render() {
      const source=code(method,endpoint,task);
      const result=values[method][endpoint];
      container.innerHTML=`<h4>${task==='risk'?'Risk grouping, RMST and molecular figures':'Fit and predict'} · ${method.toUpperCase()}-EN</h4>
        <div class="segmented" aria-label="${method} ${task} outcome">
          <button data-outcome="binary" class="${endpoint==='binary'?'selected':''}">Binary</button>
          <button data-outcome="cox" class="${endpoint==='cox'?'selected':''}">Time to event</button>
        </div><pre><code>${encode(source)}</code></pre>
        <button class="copy-example" title="Copy the complete Python example">Copy code</button>
        <p><b>Verified synthetic output:</b> ${endpoint==='binary'?'known-status AUC = '+result[0]+'. ':''}
        ${task==='risk'?'RMST(Low) − RMST(High) = '+result[1]+' at horizon 0.25. ':''}40 held-out target predictions. These are synthetic-data outputs, not the clinical figures.</p>
        ${task==='risk'?`<div class="actions"><a href="downloads/example-output/${method}_${endpoint}/risk_rmst.pdf">Actual example RMST figure ↗</a><a href="downloads/example-output/${method}_${endpoint}/molecular.pdf">Actual example molecular figure ↗</a><a href="downloads/example-output/${method}_${endpoint}/rmst.csv" download>RMST table ↓</a><a href="downloads/example-output/${method}_${endpoint}/pathway_correlations.csv" download>Correlation table ↓</a></div>`:''}`;
      container.querySelectorAll('[data-outcome]').forEach(b=>b.onclick=()=>{endpoint=b.dataset.outcome;render();});
      container.querySelector('.copy-example').onclick=async e=>{
        try {await navigator.clipboard.writeText(source);e.target.textContent='Copied';}
        catch {e.target.textContent='Select code above to copy';}
      };
    }
    render();
  });
})();
