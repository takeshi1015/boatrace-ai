from __future__ import annotations
import itertools
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

RANK_MAP={'A1':4.0,'A2':3.0,'B1':2.0,'B2':1.0}

BASE_FEATURES=[
    'stadium_no','race_no','lane','course','rank_score','age','weight',
    'flying_count','late_count','avg_start',
    'racer_win_rate','racer_2ren_rate','racer_3ren_rate',
    'local_win_rate','local_2ren_rate',
    'motor_2ren_rate','boat_2ren_rate',
    'preview_start_timing','exhibition_time',
    'wind_speed','wind_direction','wave_height_cm',
    'air_temperature','water_temperature',
]
RELATIVE_COLS=[
    'racer_win_rate','racer_2ren_rate','local_win_rate',
    'motor_2ren_rate','boat_2ren_rate',
    'avg_start','preview_start_timing','exhibition_time',
]
FEATURES=BASE_FEATURES+[f'{c}_rel' for c in RELATIVE_COLS]+[
    'inside_lane','course_shift','start_advantage','exhibition_advantage'
]

def _numeric(s):
    return pd.to_numeric(s,errors='coerce')

def feature_frame(df):
    a=df.copy()
    a['rank_score']=a.get('rank_number',pd.Series(index=a.index,dtype=object)).map(RANK_MAP)
    a['course']=_numeric(a.get('course',a.get('lane')))
    a['course']=a['course'].fillna(_numeric(a.get('lane')))
    a['inside_lane']=7-_numeric(a.get('lane'))
    a['course_shift']=_numeric(a['course'])-_numeric(a.get('lane'))

    for c in BASE_FEATURES:
        if c not in a:
            a[c]=np.nan
        a[c]=_numeric(a[c])

    # relative-to-field features are important in a 6-boat race
    if 'race_id' in a:
        grouped=a.groupby('race_id')
        for c in RELATIVE_COLS:
            mean=grouped[c].transform('mean')
            a[f'{c}_rel']=a[c]-mean
    else:
        for c in RELATIVE_COLS:
            a[f'{c}_rel']=a[c]-a[c].mean()

    # Lower ST/exhibition time is better, hence invert relative sign
    a['start_advantage']=-a['preview_start_timing_rel']
    a['exhibition_advantage']=-a['exhibition_time_rel']

    for c in FEATURES:
        if c not in a:
            a[c]=np.nan
        a[c]=_numeric(a[c])
    return a[FEATURES]

def _make_classifier():
    return Pipeline([
        ('imp',SimpleImputer(strategy='median')),
        ('clf',HistGradientBoostingClassifier(
            max_iter=160,max_depth=5,learning_rate=.05,
            l2_regularization=1.5,random_state=42
        ))
    ])

def fit_models(hist):
    if hist is None or hist.empty or 'finish_num' not in hist:
        return None
    h=hist[hist['finish_num'].notna()].copy()
    if h['race_id'].nunique()<100:
        return None
    X=feature_frame(h)
    models={}
    for place,target in [(1,'is_first'),(2,'is_second'),(3,'is_third')]:
        y=(pd.to_numeric(h['finish_num'],errors='coerce')==place).astype(int)
        if y.sum()<30:
            return None
        m=_make_classifier()
        m.fit(X,y)
        models[place]=m
    return models

def _fallback_scores(g, place):
    lane=_numeric(g['lane']).fillna(3.5)
    wr=_numeric(g.get('racer_win_rate')).fillna(5.0)
    mot=_numeric(g.get('motor_2ren_rate')).fillna(.30)
    ex=_numeric(g.get('exhibition_time')).fillna(_numeric(g.get('exhibition_time')).median())
    ex=ex.fillna(6.8)
    base=.48*wr+.85*mot+.22*(7-lane)-.35*(ex-ex.mean())
    if place==2:
        base=.36*wr+.65*mot+.08*(7-lane)-.20*(ex-ex.mean())
    if place==3:
        base=.28*wr+.55*mot+.03*(7-lane)-.15*(ex-ex.mean())
    return np.exp(base-np.nanmax(base))

def place_scores(models,g):
    out={}
    xf=feature_frame(g)
    for place in (1,2,3):
        if models is None or place not in models:
            raw=_fallback_scores(g,place)
        else:
            raw=models[place].predict_proba(xf)[:,1]
        raw=np.maximum(np.asarray(raw,float),1e-8)
        out[place]=raw
    return out

def trifecta_table(models,g):
    scores=place_scores(models,g)
    lanes=g['lane'].astype(int).tolist()
    idx={lane:i for i,lane in enumerate(lanes)}
    rows=[]
    for a,b,c in itertools.permutations(lanes,3):
        # sequential conditional normalization:
        # first model for 1st; second-place model among remaining;
        # third-place model among remaining after first+second.
        s1=scores[1]
        p1=s1[idx[a]]/s1.sum()

        rem2=[x for x in lanes if x!=a]
        denom2=sum(scores[2][idx[x]] for x in rem2)
        p2=scores[2][idx[b]]/max(denom2,1e-12)

        rem3=[x for x in lanes if x not in (a,b)]
        denom3=sum(scores[3][idx[x]] for x in rem3)
        p3=scores[3][idx[c]]/max(denom3,1e-12)

        rows.append({'combo':f'{a}-{b}-{c}','prob':float(p1*p2*p3)})
    df=pd.DataFrame(rows)
    total=df['prob'].sum()
    if total>0:
        df['prob']/=total
    return df.sort_values('prob',ascending=False).reset_index(drop=True)

def race_confidence(tri):
    if tri is None or tri.empty:
        return 0.0
    p=tri['prob'].to_numpy(float)
    entropy=-(p*np.log(np.maximum(p,1e-12))).sum()
    return float(p[0]/max(entropy,1e-9))


def selection_signals(tri):
    """Pre-race signals used by the second-stage bet selector."""
    if tri is None or tri.empty:
        return {
            'top_prob':0.0,'prob_margin':0.0,'confidence':0.0,
            'first_lane':None,'second_lane':None,'third_lane':None,
        }
    x=tri.sort_values('prob',ascending=False).reset_index(drop=True)
    top=float(x.loc[0,'prob'])
    second=float(x.loc[1,'prob']) if len(x)>1 else 0.0
    combo=str(x.loc[0,'combo'])
    try:
        a,b,c=[int(v) for v in combo.split('-')]
    except Exception:
        a=b=c=None
    return {
        'top_prob':top,
        'prob_margin':top-second,
        'confidence':race_confidence(x),
        'first_lane':a,'second_lane':b,'third_lane':c,
    }
