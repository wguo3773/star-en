"""Held-out risk grouping, event composition, RMST and molecular correlations."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .outcomes import validate_outcome


def risk_scores(model, X, *, event_class=None, horizon=None):
    """Return higher-is-worse risk for either fitted estimator family.

    Binary: explicitly identify the event class (0 for binary_at_horizon).
    Cox: relative log hazard, or 1-S(horizon) when horizon is supplied.
    """
    if model.endpoint == 'binary':
        if event_class not in (0,1):
            raise ValueError('Binary risk requires event_class=0 or 1 explicitly')
        return model.predict_proba(X)[:,int(event_class)]
    if horizon is not None:
        return 1-model.predict_survival_function(X, [horizon])[:,0]
    return model.predict(X)


def aggregate_heldout_predictions(records):
    """Average only held-out predictions in patient/repeat/risk records.

    Caller must supply held-out rows only and comparable risk scales. This
    function never adds missing patients or predicts in-sample patients.
    """
    required = {'patient','repeat','risk'}
    if not required.issubset(records.columns) or records[list(required)].isna().any().any():
        raise ValueError('Require nonmissing patient, repeat and risk columns')
    if records.duplicated(['patient','repeat']).any() or not np.isfinite(records.risk).all():
        raise ValueError('One finite risk per patient per repeat is required')
    return records.groupby('patient', sort=False).risk.agg(risk='mean', n_heldout='size')


def _risk_vector(risk):
    risk = np.asarray(risk, dtype=float)
    if risk.ndim != 1 or not len(risk) or not np.isfinite(risk).all():
        raise ValueError('risk must be a nonempty finite vector')
    return risk


def stratify_risk(risk, outcome, horizon, *, cutoff, subgroups=None,
                  n_bootstrap=200, random_state=42):
    """Event composition and KM RMST contrasts for binary OR Cox risk scores.

    outcome must be original survival_outcome(time,event), even for a binary
    model. cutoff must be fixed independently (e.g. training median). subgroups
    maps display labels to aligned Boolean masks. CIs are pointwise percentile
    patient bootstraps conditional on fitted risks and fixed risk groups, not
    full model-development uncertainty. With n_bootstrap=0 only points return.
    Times/horizon share units; unsupported subgroup contrasts are marked NE.
    """
    from lifelines import KaplanMeierFitter
    from lifelines.utils import restricted_mean_survival_time
    risk = _risk_vector(risk)
    y = validate_outcome(outcome, 'cox', require_events=False)
    if len(risk)!=len(y) or not np.isscalar(cutoff) or not np.isfinite(cutoff):
        raise ValueError('Align risk/outcome and supply a finite scalar cutoff')
    if not np.isscalar(horizon) or not np.isfinite(horizon) or horizon<=0:
        raise ValueError('horizon must be finite and positive')
    if not isinstance(n_bootstrap, (int,np.integer)) or n_bootstrap<0:
        raise ValueError('n_bootstrap must be a nonnegative integer')
    high = risk>=cutoff
    frame = pd.DataFrame({'risk':risk,'time':y['time'],'event':y['event'],
                          'group':np.where(high,'High','Low')})
    counts=[]
    for name, mask in [('Low',~high),('High',high)]:
        ev = mask & y['event'] & (y['time']<=horizon)
        unknown = mask & ~y['event'] & (y['time']<horizon)
        n=int(mask.sum())
        counts.append({'group':name,'n':n,'events':int(ev.sum()),
            'unknown':int(unknown.sum()),'event_free':n-int(ev.sum()+unknown.sum()),
            'observed_event_pct':100*ev.sum()/n if n else np.nan})

    def rmst(indices):
        if len(indices)<2 or y['time'][indices].max()<horizon:
            raise ValueError('insufficient group size or follow-up support')
        km=KaplanMeierFitter().fit(y['time'][indices],y['event'][indices])
        return float(restricted_mean_survival_time(km,t=horizon))

    masks={'Overall':np.ones(len(y),dtype=bool)}
    for name,mask in (subgroups or {}).items():
        mask=np.asarray(mask)
        if name=='Overall' or mask.shape!=(len(y),) or mask.dtype!=bool:
            raise ValueError('subgroups need distinct labels and aligned Boolean masks')
        masks[name]=mask
    rng=np.random.default_rng(random_state)
    contrasts=[]
    for name,mask in masks.items():
        low_ids=np.flatnonzero(mask & ~high)
        high_ids=np.flatnonzero(mask & high)
        row={'subgroup':name,'n_low':len(low_ids),'n_high':len(high_ids),
             'difference':np.nan,'ci_low':np.nan,'ci_high':np.nan,
             'bootstrap_valid':0,'status':'NE'}
        try:
            row['difference']=rmst(low_ids)-rmst(high_ids)
            row['status']='estimated'
            samples=[]
            for _ in range(n_bootstrap):
                try:
                    samples.append(rmst(rng.choice(low_ids,len(low_ids),replace=True))-
                                   rmst(rng.choice(high_ids,len(high_ids),replace=True)))
                except ValueError:
                    pass
            row['bootstrap_valid']=len(samples)
            if n_bootstrap and len(samples)>=max(20,.8*n_bootstrap):
                row['ci_low'],row['ci_high']=np.quantile(samples,[.025,.975])
        except ValueError as exc:
            row['status']='NE: '+str(exc)
        contrasts.append(row)
    return {'patients':frame,'composition':pd.DataFrame(counts),
            'rmst':pd.DataFrame(contrasts),'horizon':float(horizon),'cutoff':float(cutoff)}


def pathway_scores(expression, gene_sets, *, reference):
    """Mean gene-wise z scores using a training/reference expression matrix.

    Rows are patients; columns are gene identifiers. Not GSVA or ssGSEA.
    No held-out outcomes or held-out expression are used to fit scaling.
    """
    if not expression.columns.is_unique or not reference.columns.is_unique:
        raise ValueError('Gene columns must be unique')
    if not len(reference) or not gene_sets:
        raise ValueError('Supply reference patients and named gene sets')
    columns=expression.columns.intersection(reference.columns)
    x=expression[columns].astype(float)
    ref=reference[columns].astype(float)
    mean,sd=ref.mean(),ref.std(ddof=0)
    z=(x-mean)/sd.where(sd>0)
    scores={}
    for name,genes in gene_sets.items():
        common=columns.intersection(list(genes))
        scores[name]=z[common].mean(axis=1) if len(common) else np.nan
    return pd.DataFrame(scores,index=expression.index)


def correlate_programs(risk, programs):
    """Spearman correlations and BH q-values; rows must align with risk.

    Supply precomputed pathways or miRNA abundances. All tested programs
    are retained; plotting can filter nominal p or adjusted q explicitly.
    """
    risk=_risk_vector(risk)
    if len(risk)!=len(programs) or not programs.columns.is_unique:
        raise ValueError('Align patients and use unique program columns')
    rows=[]
    for name in programs.columns:
        values=pd.to_numeric(programs[name],errors='coerce').to_numpy()
        keep=np.isfinite(values)
        rho,p=np.nan,np.nan
        if keep.sum()>=3 and np.unique(values[keep]).size>1 and np.unique(risk[keep]).size>1:
            rho,p=spearmanr(risk[keep],values[keep])
        rows.append({'program':name,'n':int(keep.sum()),'rho':rho,'p':p})
    table=pd.DataFrame(rows,columns=['program','n','rho','p'])
    table['q']=np.nan
    valid=table.p.notna()
    p=table.loc[valid,'p'].to_numpy()
    if len(p):
        order=np.argsort(p)
        adjusted=np.minimum.accumulate((p[order]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
        q=np.empty(len(p)); q[order]=np.minimum(adjusted,1.)
        table.loc[valid,'q']=q
    return table.sort_values('p').reset_index(drop=True)


def plot_risk_stratification(result):
    """Return matplotlib figure: low/high event-composition donut and RMST forest.

    Accepts stratify_risk output from either binary or survival risk scores.
    Observed event fractions are not censoring-adjusted event probabilities.
    """
    import matplotlib.pyplot as plt
    fig,(a,b)=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1,1.8]})
    c=result['composition'].set_index('group')
    values=[c.loc[g,k] for g in ['Low','High'] for k in ['events','event_free','unknown']]
    a.pie(values,colors=['#2815ab','#bcb3e8','#dddddd','#eb0964','#f5bdd3','#aaaaaa'],
          startangle=90,wedgeprops={'width':.35,'edgecolor':'white'})
    a.text(0,0,f"Horizon {result['horizon']:g}\nLow: {c.loc['Low','n']}\nHigh: {c.loc['High','n']}",
           ha='center',va='center')
    a.set_title('Risk-group event composition')
    a.legend(['Low: event','Low: event-free','Low: unknown','High: event',
              'High: event-free','High: unknown'],loc='lower center',bbox_to_anchor=(.5,-.3),fontsize=8)
    r=result['rmst']
    for i,row in r.iterrows():
        if np.isfinite(row.difference):
            b.plot(row.difference,i,'o',color='#147c82')
            if np.isfinite(row.ci_low) and np.isfinite(row.ci_high):
                b.hlines(i,row.ci_low,row.ci_high,color='#147c82')
        else:
            b.text(.98,i,'NE',transform=b.get_yaxis_transform(),ha='right')
    b.axvline(0,color='gray',ls='--'); b.set_yticks(range(len(r)),r.subgroup)
    b.invert_yaxis(); b.set_xlabel('RMST(Low) minus RMST(High), input time units')
    b.set_title('Restricted mean event-free time')
    fig.tight_layout()
    return fig


def plot_molecular_correlations(pathways, mirnas=None, *, threshold=.05, significance='p'):
    """Return pathway/miRNA correlation figure; retain both correlation signs.

    significance='p' is nominal; 'q' selects BH adjustment. Asterisks always
    denote nominal p (<.05, <.005, <.001). Empty panels are labeled explicitly.
    """
    import matplotlib.pyplot as plt
    if significance not in ('p','q') or not 0<threshold<=1:
        raise ValueError('Select p/q and threshold in (0,1]')
    tables=[('Pathway scores',pathways)]+([] if mirnas is None else [('miRNA abundance',mirnas)])
    fig,axes=plt.subplots(1,len(tables),figsize=(7*len(tables),7),squeeze=False)
    for ax,(title,table) in zip(axes[0],tables):
        shown=table.loc[table[significance]<threshold].sort_values('rho')
        if len(shown):
            ax.barh(shown.program,shown.rho,color=np.where(shown.rho>=0,'#ff0067','#4289ff'))
            for i,row in enumerate(shown.itertuples()):
                stars='***' if row.p<.001 else '**' if row.p<.005 else '*' if row.p<.05 else ''
                ax.text(row.rho+(.015 if row.rho>=0 else -.015),i,stars,
                        ha='left' if row.rho>=0 else 'right',va='center')
        else:
            ax.text(.5,.5,'No programs meet threshold',transform=ax.transAxes,ha='center')
        ax.axvline(0,color='gray',ls='--'); ax.set_xlim(-1,1)
        ax.set_xlabel("Spearman's rho"); ax.set_title(f'{title} ({significance} < {threshold:g})')
    fig.tight_layout()
    return fig
