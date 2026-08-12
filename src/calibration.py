from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS=1e-6

def _logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p/(1-p))

def fit_probability_calibrator(walk_forward_preds):
    """Fit a conservative calibration map from strictly out-of-sample top picks.

    The historical walk-forward table contains only predictions made with data
    prior to each race, so it is safe to use for probability calibration.
    """
    if walk_forward_preds is None or walk_forward_preds.empty:
        return None, {'samples':0,'positives':0,'brier_raw':None,'brier_cal':None}
    d=walk_forward_preds.copy()
    p=pd.to_numeric(d.get('prob'),errors='coerce')
    y=d.get('hit',pd.Series(index=d.index,dtype=float)).astype(float)
    m=p.notna() & y.notna()
    p=p[m].clip(EPS,1-EPS)
    y=y[m].astype(int)
    if len(p)<300 or y.sum()<20 or y.nunique()<2:
        return None, {'samples':int(len(p)),'positives':int(y.sum()),'brier_raw':None,'brier_cal':None}
    x=_logit(p).reshape(-1,1)
    model=LogisticRegression(C=0.25,solver='lbfgs',max_iter=1000)
    model.fit(x,y)
    cal=model.predict_proba(x)[:,1]
    raw=float(np.mean((p.to_numpy()-y.to_numpy())**2))
    bcal=float(np.mean((cal-y.to_numpy())**2))
    return model, {
        'samples':int(len(p)), 'positives':int(y.sum()),
        'brier_raw':raw, 'brier_cal':bcal,
        'coef':float(model.coef_[0,0]), 'intercept':float(model.intercept_[0])
    }

def calibrate_distribution(tri, calibrator, blend=0.55):
    """Calibrate trifecta probabilities then renormalize to sum to one.

    Blend with raw probability to avoid over-reacting to a limited calibration
    sample. This intentionally favors stability over aggressive correction.
    """
    out=tri.copy()
    raw=pd.to_numeric(out['prob'],errors='coerce').fillna(0).clip(EPS,1-EPS).to_numpy()
    if calibrator is None:
        out['raw_prob']=raw
        out['prob']=raw/raw.sum() if raw.sum()>0 else raw
        return out
    pred=calibrator.predict_proba(_logit(raw).reshape(-1,1))[:,1]
    q=blend*raw+(1-blend)*pred
    q=np.maximum(q,EPS)
    q=q/q.sum()
    out['raw_prob']=raw
    out['prob']=q
    return out
