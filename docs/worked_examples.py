"""Run all four STAR-EN workflows on small reproducible synthetic data.

python examples/star_workflows.py --output star_outputs
No clinical results are refitted. Time units in this demonstration are arbitrary.
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
from staren import (SATBinaryCV, SATCoxCV, LSATBinaryCV, LSATCoxCV,
    make_survival_transfer, binary_at_horizon, risk_scores, binary_metrics,
    stratify_risk, pathway_scores, correlate_programs,
    plot_risk_stratification, plot_molecular_correlations)


def run_case(method, endpoint, output):
    """Prediction followed by both risk figures, for one method/endpoint."""
    data=make_survival_transfer(n_source=70,n_target=50,n_test=40,
        n_features=12,block_size=4,random_state=42)
    horizon=.25
    xs,xt,xe=(data[k] for k in ['X_source','X_target','X_test'])
    ys,yt,ye=(data[k] for k in ['y_source','y_target','y_test'])
    classes={('sat','binary'):SATBinaryCV,('sat','cox'):SATCoxCV,
             ('lsat','binary'):LSATBinaryCV,('lsat','cox'):LSATCoxCV}
    kwargs=dict(lambdas=(.05,.15),alphas=(.5,),cv=2)
    if method=='lsat': kwargs.update(tau_grid=(0.,.5),coordinate_passes=1)
    model=classes[method,endpoint](**kwargs)
    if endpoint=='binary':
        source_known,source_labels=binary_at_horizon(ys,horizon)
        target_known,target_labels=binary_at_horizon(yt,horizon)
        model.fit(xs[source_known],source_labels,xt[target_known],target_labels,groups=data['groups'])
        # Label 0 means event by horizon; prediction also works for censored test patients.
        risk=risk_scores(model,xe,event_class=0)
        cutoff=float(np.median(risk_scores(model,xt[target_known],event_class=0)))
        known,labels=binary_at_horizon(ye,horizon)
        metric=binary_metrics(labels,1-risk[known])['auc']
        print(method,endpoint,'known-status AUC',round(metric,4))
    else:
        model.fit(xs,ys,xt,yt,groups=data['groups'])
        risk=risk_scores(model,xe)
        cutoff=float(np.median(risk_scores(model,xt)))
    if not model.converged_: raise RuntimeError('Model did not converge')
    result=stratify_risk(risk,ye,horizon,cutoff=cutoff,n_bootstrap=30,
        subgroups={'Synthetic subgroup':np.arange(len(ye))%2==0})
    columns=[f'gene{i}' for i in range(xs.shape[1])]
    training=pd.DataFrame(np.vstack([xs,xt]),columns=columns)
    expression=pd.DataFrame(xe,columns=columns)
    # Demonstration gene sets are NOT biological Hallmark pathways; replace with real sets.
    scores=pathway_scores(expression,{'Demonstration set A':columns[:4],
        'Demonstration set B':columns[4:8]},reference=training)
    correlations=correlate_programs(risk,scores)
    mirnas=correlate_programs(risk,expression[columns[8:]])
    out=Path(output)/f'{method}_{endpoint}'; out.mkdir(parents=True,exist_ok=True)
    model.feature_table().to_csv(out/'coefficients.csv',index=False)
    result['patients'].to_csv(out/'heldout_risks.csv',index=False)
    result['composition'].to_csv(out/'composition.csv',index=False)
    result['rmst'].to_csv(out/'rmst.csv',index=False)
    correlations.to_csv(out/'pathway_correlations.csv',index=False)
    mirnas.to_csv(out/'mirna_correlations.csv',index=False)
    import matplotlib.pyplot as plt
    for name,fig in [('risk_rmst',plot_risk_stratification(result)),
                     ('molecular',plot_molecular_correlations(correlations,mirnas,threshold=1.))]:
        fig.savefig(out/f'{name}.png',dpi=160,bbox_inches='tight')
        fig.savefig(out/f'{name}.pdf',bbox_inches='tight'); plt.close(fig)
    print(method,endpoint,'RMST difference',round(result['rmst'].iloc[0].difference,4))
    return model,result,correlations


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='star_outputs')
    args=parser.parse_args()
    for method in ['sat','lsat']:
        for endpoint in ['binary','cox']: run_case(method,endpoint,args.output)
