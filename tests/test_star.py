import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from staren import (SATBinary, SATCox, SATBinaryCV, SATCoxCV, LSATBinaryCV,
    LSATCoxCV, make_binary_transfer, make_survival_transfer, risk_scores,
    stratify_risk, correlate_programs, pathway_scores, survival_outcome,
    aggregate_heldout_predictions, plot_risk_stratification, plot_molecular_correlations)


@pytest.mark.parametrize('cls',[SATBinary,SATBinaryCV,SATCox,SATCoxCV])
def test_sat_estimators(cls):
    data=(make_binary_transfer if 'Binary' in cls.__name__ else make_survival_transfer)(
        n_source=70,n_target=50,n_test=30,n_features=8,block_size=4)
    kwargs={'cv':2,'lambdas':(.1,)} if 'CV' in cls.__name__ else {}
    model=cls(**kwargs)
    assert 'tau' not in model.get_params() and 'tau_grid' not in model.get_params()
    assert clone(model).get_params()==model.get_params()
    model.fit(data['X_source'],data['y_source'],data['X_target'],data['y_target'],groups=data['groups'])
    assert model.converged_ and 'tau' not in model.best_params_
    assert len(model.feature_table())==8
    np.testing.assert_allclose(model.coef_standardized_,
        model.anchor_coef_standardized_+model.deviation_standardized_)
    assert np.isfinite(risk_scores(model,data['X_test'],event_class=0)).all()
    if model.endpoint=='cox':
        assert model.predict_survival_function(data['X_test'],[0,1]).shape==(30,2)


def test_sat_large_penalty_preserves_anchor_not_zero():
    data=make_binary_transfer(n_features=8,block_size=4)
    m=SATBinary(lam=100.,alpha=1.).fit(data['X_source'],data['y_source'],
        data['X_target'],data['y_target'],groups=data['groups'])
    np.testing.assert_allclose(m.deviation_standardized_,0,atol=1e-5)
    assert np.any(np.abs(m.anchor_coef_standardized_)>1e-6)


def test_composition_rmst_and_ne(tmp_path):
    y=survival_outcome([1,2,3,4,1,2,3,4],[1,1,0,0,1,0,0,0])
    risk=np.arange(8)
    result=stratify_risk(risk,y,2,cutoff=4,n_bootstrap=30,
        subgroups={'Too small':np.array([True]+[False]*7)})
    assert result['composition'].n.sum()==8
    assert result['rmst'].iloc[1].status.startswith('NE')
    fig=plot_risk_stratification(result); fig.savefig(tmp_path/'risk.png')
    assert (tmp_path/'risk.png').stat().st_size>1000
    with pytest.raises(ValueError): stratify_risk(risk,y,float('inf'),cutoff=4)


def test_correlations_and_training_reference(tmp_path):
    risk=np.arange(10)
    programs=pd.DataFrame({'Positive':risk,'Negative':-risk,'Constant':np.ones(10)})
    table=correlate_programs(risk,programs)
    assert table.set_index('program').loc['Positive','rho']>.99
    assert table.set_index('program').loc['Negative','rho']<-.99
    assert table.set_index('program').loc['Constant','p']!=table.set_index('program').loc['Constant','p']
    assert (table.q.dropna()>=table.p.dropna()).all()
    scores=pathway_scores(programs,{'Set':['Positive']},reference=programs.iloc[:5])
    assert scores.iloc[0,0]==pytest.approx((0-2)/np.std(np.arange(5)))
    fig=plot_molecular_correlations(table,table,threshold=.05)
    fig.savefig(tmp_path/'correlations.png')
    assert (tmp_path/'correlations.png').stat().st_size>1000


def test_heldout_aggregation():
    records=pd.DataFrame({'patient':['a','a','b'],'repeat':[1,2,1],'risk':[.2,.4,.8]})
    out=aggregate_heldout_predictions(records)
    assert out.loc['a','risk']==pytest.approx(.3)
    assert out.loc['a','n_heldout']==2 and len(out)==2
    with pytest.raises(ValueError): aggregate_heldout_predictions(pd.concat([records,records]))


def test_explicit_binary_risk_direction():
    data=make_binary_transfer(n_features=8,block_size=4)
    m=SATBinary().fit(data['X_source'],data['y_source'],data['X_target'],
        data['y_target'],groups=data['groups'])
    with pytest.raises(ValueError): risk_scores(m,data['X_test'])
    np.testing.assert_allclose(risk_scores(m,data['X_test'],event_class=0),
        1-m.predict_proba(data['X_test'])[:,1])
