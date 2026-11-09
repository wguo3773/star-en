const dialog = document.querySelector('#detail');
const detail = document.querySelector('#detail-body');
const escapeHTML = s => s.replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');
const common = `model.fit(data["X_source"], data["y_source"],
          data["X_target"], data["y_target"], groups=data["groups"])`;
const snippets = {
binary: `from lsaten import LSATBinaryCV, make_binary_transfer, binary_metrics

data = make_binary_transfer(
    n_source=180, n_target=90, n_test=150,
    n_features=36, block_size=6, sigma2_tl=0.25, random_state=42)
model = LSATBinaryCV(lambdas=(0.02, 0.1), alphas=(0.5,),
    tau_grid=(0.0, 0.5, 2.0), cv=3, coordinate_passes=1)
${common}
p_label1 = model.predict_proba(data["X_test"])[:, 1]
print(binary_metrics(data["y_test"], p_label1))
print(model.best_params_)
print(model.feature_table().head())

# For real horizon data: label 1 = event-free beyond the horizon.
# Event risk is then 1 - p_label1. Do not apply this interpretation
# to arbitrary synthetic class labels without checking their encoding.`,
cox: `from lsaten import LSATCoxCV, make_survival_transfer, survival_metrics

data = make_survival_transfer(
    n_source=140, n_target=90, n_test=100,
    n_features=36, block_size=6, random_state=43)
model = LSATCoxCV(lambdas=(0.05, 0.15), alphas=(0.5,),
    tau_grid=(0.0, 0.5), cv=3, coordinate_passes=1)
${common}
risk = model.predict(data["X_test"])  # Higher = higher event risk.
print(survival_metrics(data["y_target"], data["y_test"], risk,
    times=(2.0, 3.0, 4.0, 5.0), on_error="nan"))
survival = model.predict_survival_function(
    data["X_test"], times=[2.0, 3.0, 4.0, 5.0])
print(model.feature_table().head())
# Synthetic follow-up uses years; use matching units in actual data.
# Unsupported IPCW evaluations remain missing, not zero.`};
function codeView(kind) {
  const target = document.querySelector('#code-example');
  if (!target) return;
  target.innerHTML = `<pre><code>${escapeHTML(snippets[kind])}</code></pre>`;
  document.querySelectorAll('[data-code]').forEach(b => b.classList.toggle('selected',b.dataset.code===kind));
}
codeView('binary');
const information = {
cohorts: ['Define source and target', 'Use a larger source cohort to estimate an anchor. Reserve target test patients before preprocessing or tuning. Age and race describe the saved study cohorts; neither is a requirement of the estimator. Race should not be interpreted as a direct genetic or biological category.'],
omics: ['Prepare aligned inputs', 'Rows are patients; columns are matched molecular features. Impute and standardize using training data only. The public package accepts all supplied columns; elastic-net shrinkage during fitting is distinct from screening. Historical study panels used different feature-selection workflows.'],
'sat-binary': ['SAT-EN · binary', 'Fit a source logistic anchor, then a target logistic model with an elastic-net penalty on the deviation δ. For fixed-horizon survival, retain only known outcomes at the horizon. Historical SAT-EN is not a separate estimator in the current public LSAT-EN API.', 'min L_T(b_T + X_T(β̂_S + δ)) + λ[α‖δ‖₁ + (1−α)‖δ‖²₂/2]'],
'sat-cox': ['SAT-EN · time to event', 'Use a Cox partial-likelihood target loss with EN-regularized source-to-target deviations. Retain censored observations. The resulting linear predictor ranks relative event risk; baseline survival is needed for absolute probabilities.', 'min −ℓ_T(β̂_S + δ) + λ[α‖δ‖₁ + (1−α)‖δ‖²₂/2]'],
'lsat-binary': ['LSAT-EN · binary', 'Fit a source logistic anchor and tune λ, α and τ_g on target-training validation folds. Each molecular layer has its own source-anchoring strength. Held-out target patients are not used for tuning.', 'min L_T(β) + P_EN(β) + ½Σ_g τ_g‖β_g−β̂_Sg‖²₂'],
'lsat-cox': ['LSAT-Cox-EN · time to event', 'Replace logistic loss with negative Cox partial likelihood. Retain censored patients and evaluate time-dependent risk using IPCW AUC where censoring support is adequate.', 'min −ℓ_T(β) + P_EN(β) + ½Σ_g τ_g‖β_g−β̂_Sg‖²₂'],
rmst: ['Risk groups & RMST', 'Use held-out risk scores and a documented cutoff to define groups. Estimate Kaplan–Meier curves and integrate to each horizon. Positive RMST_low − RMST_high means longer restricted mean event-free time in the low-risk group. The preview is the saved breast cancer Overall RSF analysis; no LSAT claim is made.', 'RMST_k(t) = ∫₀ᵗ Ŝ_k(u) du; Δ(t) = RMST_low(t) − RMST_high(t)'],
pathway: ['Risk scores & molecular programs', 'Correlate each patient’s held-out risk with Hallmark scores and miRNA abundance. Pink means positive correlation; blue means negative. Orange labels mark HMDD breast-associated families, not necessarily oncogenic or protective functions. The preview is the saved Overall RSF analysis.', 'ρ_g(t) = Spearman(risk(t), molecular_score_g)']};
document.querySelectorAll('[data-detail]').forEach(b => b.addEventListener('click', () => {
  const [title,body,formula] = information[b.dataset.detail];
  detail.innerHTML = `<p class="eyebrow">WORKFLOW DETAIL</p><h2>${title}</h2><p>${body}</p>${formula?`<div class="modal-formula">${formula}</div>`:''}<p><a href="#examples" id="read-methods">Worked examples →</a></p>`;
  document.querySelector('#read-methods').onclick = () => dialog.close();
  dialog.showModal();
}));
document.querySelectorAll('[data-image]').forEach(b => b.addEventListener('click', () => {
  const img = b.querySelector('img');
  detail.replaceChildren();
  const full = document.createElement('img'); full.src=b.dataset.image; full.alt=img.alt; full.className='lightbox'; detail.append(full);
  const link=document.createElement('a'); link.href=b.dataset.image; link.download=''; link.textContent='Download full-resolution figure ↓'; detail.append(link);
  dialog.showModal();
}));
document.querySelector('.close').onclick=()=>dialog.close();
dialog.addEventListener('click', e=> { if(e.target===dialog) dialog.close(); });
document.querySelectorAll('[data-code]').forEach(b=>b.onclick=()=>codeView(b.dataset.code));
function route() {
  const requested=location.hash.slice(1)||'overview';
  const id=requested==='methods'?'examples':requested;
  const valid=['overview','installation','datasets','examples'].includes(id)?id:'overview';
  document.querySelectorAll('.page').forEach(p=>p.classList.toggle('active',p.id===valid));
  document.querySelectorAll('nav a').forEach(a=> {if(a.hash==='#'+valid)a.setAttribute('aria-current','page');else a.removeAttribute('aria-current');});
  window.scrollTo(0,0);
}
window.addEventListener('hashchange',route);route();
document.querySelectorAll('[data-scroll]').forEach(a=>a.addEventListener('click',()=>setTimeout(()=>document.getElementById(a.dataset.scroll).scrollIntoView({behavior:'smooth'}),10)));
