from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from .model import fit_models,trifecta_table,selection_signals

def generate_walk_forward_predictions(hist,min_train_days=35,test_days=7,max_test_days=42):
    """
    Base-model walk-forward predictions. Every test race is predicted only
    by models trained on dates strictly before its test window.
    """
    if hist is None or hist.empty:
        return pd.DataFrame()
    h=hist.copy()
    h['race_date_dt']=pd.to_datetime(h['race_date'],errors='coerce')
    h=h[h['race_date_dt'].notna() & h['finish_num'].notna()].copy()
    dates=sorted(h['race_date_dt'].dt.normalize().drop_duplicates())
    if len(dates)<min_train_days+3:
        return pd.DataFrame()

    test_dates=dates[-min(max_test_days,len(dates)-min_train_days):]
    rows=[]
    for start_i in range(0,len(test_dates),test_days):
        window=test_dates[start_i:start_i+test_days]
        if not window:
            continue
        cutoff=window[0]
        train=h[h['race_date_dt']<cutoff]
        test=h[h['race_date_dt'].dt.normalize().isin(window)]
        if train['race_id'].nunique()<300:
            continue
        models=fit_models(train)
        if models is None:
            continue

        for rid,g in test.groupby('race_id'):
            actual=g['trifecta_result'].dropna()
            payout=pd.to_numeric(g['trifecta_payout'],errors='coerce').dropna()
            if not len(actual):
                continue
            try:
                tri=trifecta_table(models,g)
                top=tri.iloc[0]
                sig=selection_signals(tri)
                pred=str(top['combo'])
                act=str(actual.iloc[0])
                hit=pred==act
                pay=float(payout.iloc[0]) if len(payout) else 0.0
                rows.append({
                    'race_id':rid,
                    'race_date':str(g['race_date'].iloc[0])[:10],
                    'venue':g['venue'].iloc[0],
                    'stadium_no':int(g['stadium_no'].iloc[0]),
                    'race_no':int(g['race_no'].iloc[0]),
                    'pred':pred,'prob':float(top['prob']),
                    'prob_margin':sig['prob_margin'],
                    'confidence':sig['confidence'],
                    'first_lane':sig['first_lane'],
                    'actual':act,'hit':bool(hit),
                    'payout':pay,'stake':100.0,
                    'return_yen':pay if hit else 0.0,
                    'profit_yen':(pay if hit else 0.0)-100.0,
                })
            except Exception:
                pass
    return pd.DataFrame(rows)

def summarize(df):
    if df is None or df.empty:
        return {'races':0,'hit_rate':0.0,'roi':0.0,'profit':0.0}
    stake=100.0*len(df)
    ret=pd.to_numeric(df['return_yen'],errors='coerce').fillna(0).sum()
    return {
        'races':int(len(df)),
        'hit_rate':float(df['hit'].mean()),
        'roi':float(ret/stake) if stake else 0.0,
        'profit':float(ret-stake),
    }

def _apply_rule(df,rule):
    m=pd.Series(True,index=df.index)
    if rule.get('min_prob') is not None:
        m &= df['prob']>=rule['min_prob']
    if rule.get('min_margin') is not None:
        m &= df['prob_margin']>=rule['min_margin']
    if rule.get('min_conf') is not None:
        m &= df['confidence']>=rule['min_conf']
    lane=rule.get('first_lane')
    if lane is not None:
        m &= df['first_lane']==lane
    race_max=rule.get('race_no_max')
    if race_max is not None:
        m &= df['race_no']<=race_max
    return df[m].copy()

def _candidate_rules(train):
    """
    Search a deliberately small, interpretable rule family.
    This reduces over-fitting compared with venue-by-venue arbitrary rules.
    """
    if train.empty:
        return []
    probs=sorted(set([0.0]+[
        float(train['prob'].quantile(q)) for q in (.35,.50,.65,.75,.85,.90)
    ]))
    margins=sorted(set([0.0]+[
        float(train['prob_margin'].quantile(q)) for q in (.40,.60,.75,.85)
    ]))
    confs=sorted(set([0.0]+[
        float(train['confidence'].quantile(q)) for q in (.40,.60,.75,.85)
    ]))
    rules=[]
    # Keep search constrained to robust dimensions.
    for p,m,c in itertools.product(probs,margins,confs):
        rules.append({'min_prob':p,'min_margin':m,'min_conf':c,'first_lane':None,'race_no_max':None})
    # Add lane-1 variants only; not all lane-specific rules.
    for p,c in itertools.product(probs,confs):
        rules.append({'min_prob':p,'min_margin':0.0,'min_conf':c,'first_lane':1,'race_no_max':None})
    return rules

def optimize_rule(train,min_bets=80):
    """
    Conservative objective:
      - require enough bets,
      - shrink observed ROI toward 100%,
      - penalize very low hit-rate / tiny samples.
    """
    if train is None or train.empty:
        return None,{}
    best=None
    best_stats=None
    best_score=-1e9
    for rule in _candidate_rules(train):
        x=_apply_rule(train,rule)
        n=len(x)
        if n<min_bets:
            continue
        s=summarize(x)
        # Empirical-Bayes style shrinkage to neutral ROI=1 with 150 pseudo-bets.
        shrunk_roi=(s['roi']*n + 1.0*150)/(n+150)
        # Encourage sufficient frequency and some hit consistency.
        score=(shrunk_roi-1.0)*100 + min(n/400,1.0)*2 + min(s['hit_rate']/0.10,1.0)
        if score>best_score:
            best_score=score; best=rule; best_stats=s|{'shrunk_roi':shrunk_roi,'score':score}
    return best,best_stats or {}

def selector_walk_forward(preds,lookback_days=21,step_days=7,min_bets=80):
    """
    Second-stage nested walk-forward:
    optimize the buy/no-buy rule on PRIOR base predictions, then apply it to
    the following unseen period. This is the key anti-overfit evaluation.
    """
    if preds is None or preds.empty:
        return pd.DataFrame(),{},pd.DataFrame()
    p=preds.copy()
    p['date_dt']=pd.to_datetime(p['race_date'],errors='coerce')
    p=p[p['date_dt'].notna()].sort_values('date_dt')
    dates=sorted(p['date_dt'].dt.normalize().drop_duplicates())
    if len(dates)<lookback_days+step_days:
        return pd.DataFrame(),{},pd.DataFrame()

    eval_rows=[]
    rule_log=[]
    for i in range(lookback_days,len(dates),step_days):
        test_dates=dates[i:i+step_days]
        if not test_dates:
            break
        test_start=test_dates[0]
        train_dates=dates[max(0,i-lookback_days):i]
        train=p[p['date_dt'].dt.normalize().isin(train_dates)]
        test=p[p['date_dt'].dt.normalize().isin(test_dates)]
        rule,stats=optimize_rule(train,min_bets=min_bets)
        if rule is None:
            continue
        selected=_apply_rule(test,rule)
        if len(selected):
            selected=selected.copy()
            selected['selector_train_roi']=stats.get('roi')
            selected['selector_train_bets']=stats.get('races')
            selected['rule_min_prob']=rule.get('min_prob')
            selected['rule_min_margin']=rule.get('min_margin')
            selected['rule_min_conf']=rule.get('min_conf')
            selected['rule_first_lane']=rule.get('first_lane')
            eval_rows.append(selected)
        rule_log.append({
            'test_start':str(pd.Timestamp(test_start).date()),
            **rule,
            'train_bets':stats.get('races'),
            'train_hit_rate':stats.get('hit_rate'),
            'train_roi':stats.get('roi'),
            'train_shrunk_roi':stats.get('shrunk_roi'),
        })
    out=pd.concat(eval_rows,ignore_index=True) if eval_rows else pd.DataFrame()
    return out,summarize(out),pd.DataFrame(rule_log)

def fit_current_selector(preds,lookback_days=28,min_bets=100):
    """Fit today's selector only on the most recent historical walk-forward predictions."""
    if preds is None or preds.empty:
        return None,{}
    p=preds.copy()
    p['date_dt']=pd.to_datetime(p['race_date'],errors='coerce')
    p=p[p['date_dt'].notna()].sort_values('date_dt')
    dates=sorted(p['date_dt'].dt.normalize().drop_duplicates())
    if not dates:
        return None,{}
    keep=dates[-min(lookback_days,len(dates)):]
    train=p[p['date_dt'].dt.normalize().isin(keep)]
    return optimize_rule(train,min_bets=min_bets)

def current_buy_score(race_row,rule):
    """
    0-100 selector score from distance above optimized historical thresholds.
    Current real odds/EV are handled separately by the app.
    """
    if rule is None:
        return 0.0,False
    p=float(race_row.get('確率1位%',0))/100.0
    m=float(race_row.get('確率差',0))
    c=float(race_row.get('確信度',0))
    checks=[
        p>=float(rule.get('min_prob') or 0),
        m>=float(rule.get('min_margin') or 0),
        c>=float(rule.get('min_conf') or 0),
    ]
    if rule.get('first_lane') is not None:
        try:
            first=int(str(race_row.get('確率1位','')).split('-')[0])
            checks.append(first==int(rule['first_lane']))
        except Exception:
            checks.append(False)
    passed=all(checks)
    # Smooth score for ranking, not a calibrated probability.
    def ratio(x,t):
        if not t or t<=0:return 1.0
        return min(x/t,2.0)/2.0
    score=100*(0.40*ratio(p,rule.get('min_prob'))+
               0.30*ratio(m,rule.get('min_margin'))+
               0.30*ratio(c,rule.get('min_conf')))
    if not passed: score*=0.5
    return float(score),bool(passed)
