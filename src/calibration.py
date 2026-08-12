from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

EPS=1e-6
TICKET_CODES=("3t","3f","2t","2f")
TICKET_NAMES={"3t":"3連単","3f":"3連複","2t":"2連単","2f":"2連複"}

def _logit(p):
    p=np.clip(np.asarray(p,float),EPS,1-EPS)
    return np.log(p/(1-p))

def _fit_binary_calibrator(prob, hit, min_samples=180, min_pos=15):
    p=pd.to_numeric(prob,errors='coerce')
    y=pd.Series(hit).astype(float)
    m=p.notna() & y.notna()
    p=p[m].clip(EPS,1-EPS)
    y=y[m].astype(int)
    info={'samples':int(len(p)),'positives':int(y.sum()),'brier_raw':None,'brier_cal':None,
          'mean_pred':None,'hit_rate':None,'overconfidence_ratio':None,'safe_factor':1.0}
    if len(p):
        info['mean_pred']=float(p.mean()); info['hit_rate']=float(y.mean())
        if float(p.mean())>0:
            ratio=float(y.mean()/p.mean())
            info['overconfidence_ratio']=ratio
            # Never increase today's probability because of historical under-confidence.
            # If the model was over-confident, shrink toward the observed/expected ratio.
            info['safe_factor']=float(np.clip((ratio*len(p)+1.0*120)/(len(p)+120),0.55,1.0))
    if len(p)<min_samples or y.sum()<min_pos or y.nunique()<2:
        return None,info
    x=_logit(p).reshape(-1,1)
    model=LogisticRegression(C=0.20,solver='lbfgs',max_iter=1000)
    model.fit(x,y)
    cal=model.predict_proba(x)[:,1]
    info['brier_raw']=float(np.mean((p.to_numpy()-y.to_numpy())**2))
    info['brier_cal']=float(np.mean((cal-y.to_numpy())**2))
    info['coef']=float(model.coef_[0,0]); info['intercept']=float(model.intercept_[0])
    return model,info

def fit_probability_calibrator(walk_forward_preds):
    if walk_forward_preds is None or walk_forward_preds.empty:
        return None, {'samples':0,'positives':0,'brier_raw':None,'brier_cal':None}
    return _fit_binary_calibrator(walk_forward_preds.get('prob'),walk_forward_preds.get('hit'),300,20)

def fit_ticket_calibrators(walk_forward_preds):
    """Fit separate OOS probability calibrators for each ticket type.

    Uses only walk-forward predictions, so every row was predicted by a model that
    had not seen that race yet. This evaluates probability reliability without
    requiring historical real-time odds for the non-trifecta ticket types.
    """
    models={}; stats={}
    p=walk_forward_preds if walk_forward_preds is not None else pd.DataFrame()
    for code in TICKET_CODES:
        prob_col=f'{code}_prob'; hit_col=f'{code}_hit'
        if p.empty or prob_col not in p or hit_col not in p:
            models[code]=None; stats[code]={'samples':0,'positives':0,'safe_factor':0.70}
            continue
        model,info=_fit_binary_calibrator(p[prob_col],p[hit_col],180,15)
        models[code]=model; stats[code]=info
    return models,stats

def calibrate_distribution(tri, calibrator, blend=0.55):
    out=tri.copy()
    raw=pd.to_numeric(out['prob'],errors='coerce').fillna(0).clip(EPS,1-EPS).to_numpy()
    if calibrator is None:
        out['raw_prob']=raw
        out['prob']=raw/raw.sum() if raw.sum()>0 else raw
        return out
    pred=calibrator.predict_proba(_logit(raw).reshape(-1,1))[:,1]
    q=blend*raw+(1-blend)*pred
    q=np.maximum(q,EPS); q=q/q.sum()
    out['raw_prob']=raw; out['prob']=q
    return out

def calibrate_ticket_table(df, code, ticket_models, ticket_stats, blend=0.60):
    """Apply ticket-specific calibration and a conservative safety shrink.

    The safety factor only reduces probabilities when historical OOS predictions
    were over-confident. It never increases EV because of apparent under-confidence.
    """
    out=df.copy()
    raw=pd.to_numeric(out['prob'],errors='coerce').fillna(0).clip(EPS,1-EPS).to_numpy()
    model=(ticket_models or {}).get(code)
    stat=(ticket_stats or {}).get(code,{})
    if model is not None:
        cal=model.predict_proba(_logit(raw).reshape(-1,1))[:,1]
        q=blend*raw+(1-blend)*cal
    else:
        q=raw.copy()
    sf=float(stat.get('safe_factor',0.75 if model is None else 1.0))
    n=int(stat.get('samples',0) or 0)
    uncertainty=float(np.clip(np.sqrt(max(n,1)/400.0),0.60,1.0))
    effective=sf*uncertainty
    q=np.clip(q*effective,EPS,1-EPS)
    out['raw_prob']=raw
    out['prob']=q
    out['safety_factor']=sf
    out['uncertainty_factor']=uncertainty
    out['effective_factor']=effective
    return out

def ticket_reliability_gate(ticket_stats,min_samples=200,max_overconfidence=1.20):
    """Return which ticket types are reliable enough to be considered for purchase.

    Reliability is probability-only because historical real-time odds for all
    ticket types are not yet archived. The global ROI gate remains the final gate.
    """
    result={}
    for code in TICKET_CODES:
        s=(ticket_stats or {}).get(code,{})
        n=int(s.get('samples',0) or 0)
        mp=s.get('mean_pred'); hr=s.get('hit_rate')
        ratio=s.get('overconfidence_ratio')
        b0=s.get('brier_raw'); b1=s.get('brier_cal')
        reasons=[]
        if n<min_samples: reasons.append(f'未見校正{min_samples}件未満')
        if ratio is not None and ratio < 1/max_overconfidence:
            reasons.append('予測確率の過大評価が大きい')
        if b0 is not None and b1 is not None and b1>b0*1.03:
            reasons.append('確率校正でBrier悪化')
        result[code]={'passed':len(reasons)==0,'reasons':reasons,'samples':n,
                      'mean_pred':mp,'hit_rate':hr,'ratio':ratio,
                      'safe_factor':float(s.get('safe_factor',1.0) or 1.0)}
    return result
