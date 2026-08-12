from __future__ import annotations
import itertools
import math
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
    'racer_win_rate','racer_2ren_rate','racer_3ren_rate','local_win_rate','local_2ren_rate',
    'motor_2ren_rate','boat_2ren_rate','avg_start','preview_start_timing','exhibition_time',
]
RANK_COLS=[
    'racer_win_rate','racer_2ren_rate','local_win_rate','motor_2ren_rate','boat_2ren_rate',
]
FEATURES=BASE_FEATURES+[f'{c}_rel' for c in RELATIVE_COLS]+[f'{c}_rank' for c in RANK_COLS]+[
    'inside_lane','lane_sq','course_shift','course_shift_abs','course_inside',
    'start_advantage','exhibition_advantage','start_rank','exhibition_rank',
    'local_vs_national','motor_boat_strength','racer_form_strength',
    'preview_complete','weather_complete','race_progress',
]


def _numeric(s):
    return pd.to_numeric(s,errors='coerce')


def _race_pct_rank(a,col,ascending=False):
    if col not in a:
        return pd.Series(np.nan,index=a.index)
    x=_numeric(a[col])
    if 'race_id' not in a:
        return x.rank(pct=True,ascending=ascending)
    # 1.0 = best in field for intuitive monotonicity
    r=x.groupby(a['race_id']).rank(pct=True,ascending=ascending,method='average')
    return 1.0-r+1.0/6.0 if not ascending else 1.0-r+1.0/6.0


def feature_frame(df):
    a=df.copy()
    a['rank_score']=a.get('rank_number',pd.Series(index=a.index,dtype=object)).map(RANK_MAP)
    a['course']=_numeric(a.get('course',a.get('lane')))
    a['course']=a['course'].fillna(_numeric(a.get('lane')))
    a['inside_lane']=7-_numeric(a.get('lane'))
    a['lane_sq']=_numeric(a.get('lane'))**2
    a['course_shift']=_numeric(a['course'])-_numeric(a.get('lane'))
    a['course_shift_abs']=a['course_shift'].abs()
    a['course_inside']=7-_numeric(a['course'])
    a['race_progress']=(_numeric(a.get('race_no'))-1)/11.0

    for c in BASE_FEATURES:
        if c not in a:
            a[c]=np.nan
        a[c]=_numeric(a[c])

    if 'race_id' in a:
        grouped=a.groupby('race_id')
        for c in RELATIVE_COLS:
            mean=grouped[c].transform('mean')
            a[f'{c}_rel']=a[c]-mean
    else:
        for c in RELATIVE_COLS:
            a[f'{c}_rel']=a[c]-a[c].mean()

    # Lower ST/exhibition time is better, hence invert relative sign.
    a['start_advantage']=-a['preview_start_timing_rel']
    a['exhibition_advantage']=-a['exhibition_time_rel']

    # Field ranks are robust to venue/season scale changes. 1.0 is best.
    if 'race_id' in a:
        for c in RANK_COLS:
            # larger is better
            a[f'{c}_rank']=a.groupby('race_id')[c].rank(pct=True,ascending=True)
        # smaller is better
        a['start_rank']=1.0-a.groupby('race_id')['preview_start_timing'].rank(pct=True,ascending=True)+1/6
        a['exhibition_rank']=1.0-a.groupby('race_id')['exhibition_time'].rank(pct=True,ascending=True)+1/6
    else:
        for c in RANK_COLS:
            a[f'{c}_rank']=a[c].rank(pct=True,ascending=True)
        a['start_rank']=1.0-a['preview_start_timing'].rank(pct=True,ascending=True)+1/6
        a['exhibition_rank']=1.0-a['exhibition_time'].rank(pct=True,ascending=True)+1/6

    a['local_vs_national']=a['local_win_rate']-a['racer_win_rate']
    a['motor_boat_strength']=0.65*a['motor_2ren_rate']+0.35*a['boat_2ren_rate']
    a['racer_form_strength']=0.55*a['racer_win_rate']+0.25*a['local_win_rate']+0.20*a['racer_2ren_rate']
    a['preview_complete']=a[['course','preview_start_timing','exhibition_time']].notna().mean(axis=1)
    a['weather_complete']=a[['wind_speed','wave_height_cm','air_temperature','water_temperature']].notna().mean(axis=1)

    for c in FEATURES:
        if c not in a:
            a[c]=np.nan
        a[c]=_numeric(a[c])
    return a[FEATURES]


def _make_classifier():
    return Pipeline([
        ('imp',SimpleImputer(strategy='median')),
        ('clf',HistGradientBoostingClassifier(
            max_iter=190,max_depth=5,learning_rate=.045,
            min_samples_leaf=28,l2_regularization=2.2,
            random_state=42
        ))
    ])


def _recency_weights(h, half_life_days=50):
    if 'race_date' not in h:
        return np.ones(len(h),dtype=float)
    d=pd.to_datetime(h['race_date'],errors='coerce')
    if not d.notna().any():
        return np.ones(len(h),dtype=float)
    age=(d.max()-d).dt.total_seconds().fillna(0)/86400.0
    w=np.exp(-math.log(2)*np.maximum(age,0)/half_life_days)
    # Avoid effectively deleting older history; it remains useful for rare contexts.
    return np.clip(np.asarray(w,float),0.25,1.0)


def _weighted_group_prior(h, keys, target, weights, prior_strength=60):
    tmp=h.copy()
    tmp['_target']=target.astype(float).to_numpy()
    tmp['_w']=weights
    global_rate=float(np.average(tmp['_target'],weights=tmp['_w'])) if len(tmp) else 1/6
    tmp['_wy']=tmp['_w']*tmp['_target']
    gr=tmp.groupby(keys,dropna=False).agg(_sw=('_w','sum'),_sy=('_wy','sum'),_n=('_target','size')).reset_index()
    gr['rate']=(gr['_sy']+prior_strength*global_rate)/(gr['_sw']+prior_strength)
    mapping={}
    for _,r in gr.iterrows():
        k=tuple(r[x] for x in keys) if len(keys)>1 else r[keys[0]]
        mapping[k]=(float(r['rate']),int(r['_n']))
    return {'map':mapping,'global':global_rate,'keys':keys,'prior_strength':prior_strength}


def _build_priors(h, weights):
    priors={}
    finish=pd.to_numeric(h['finish_num'],errors='coerce')
    for place in (1,2,3):
        y=(finish==place).astype(int)
        priors[place]={
            'venue_lane':_weighted_group_prior(h,['stadium_no','lane'],y,weights,45),
            'venue_course':_weighted_group_prior(h,['stadium_no','course'],y,weights,45),
            'racer':_weighted_group_prior(h,['racer_id'],y,weights,80),
            'motor':_weighted_group_prior(h,['stadium_no','motor_no'],y,weights,70),
            'rank':_weighted_group_prior(h,['rank_number'],y,weights,60),
        }
    return priors


def fit_models(hist):
    if hist is None or hist.empty or 'finish_num' not in hist:
        return None
    h=hist[hist['finish_num'].notna()].copy()
    if h['race_id'].nunique()<100:
        return None
    X=feature_frame(h)
    weights=_recency_weights(h,half_life_days=50)
    models={}
    for place in (1,2,3):
        y=(pd.to_numeric(h['finish_num'],errors='coerce')==place).astype(int)
        if y.sum()<30:
            return None
        m=_make_classifier()
        # Recent races matter more, while older races still retain a floor weight.
        m.fit(X,y,clf__sample_weight=weights)
        models[place]=m
    models['_priors']=_build_priors(h,weights)
    models['_meta']={'half_life_days':50,'train_races':int(h['race_id'].nunique()),'train_rows':int(len(h))}
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


def _prior_value(spec,row):
    if not spec:return 1/6,0
    keys=spec.get('keys',[])
    try:
        if len(keys)>1:key=tuple(row.get(k) for k in keys)
        else:key=row.get(keys[0])
        return spec.get('map',{}).get(key,(spec.get('global',1/6),0))
    except Exception:
        return spec.get('global',1/6),0


def _blend_with_priors(raw,g,place,priors):
    if not priors or place not in priors:return raw
    cfg=priors[place]
    exponents={
        'venue_lane':0.20 if place==1 else 0.13,
        'venue_course':0.12 if place==1 else 0.10,
        'racer':0.14,
        'motor':0.08,
        'rank':0.06,
    }
    out=np.maximum(np.asarray(raw,float),1e-8).copy()
    for i,(_,row) in enumerate(g.reset_index(drop=True).iterrows()):
        multiplier=1.0
        for name,expo in exponents.items():
            spec=cfg.get(name,{})
            rate,n=_prior_value(spec,row)
            glob=max(float(spec.get('global',1/6)),1e-6)
            # sample reliability further damps small empirical groups
            reliability=min(1.0,math.sqrt(max(n,0)/80.0))
            ratio=float(np.clip(rate/glob,0.60,1.65))
            multiplier*=ratio**(expo*reliability)
        out[i]*=float(np.clip(multiplier,0.62,1.55))
    return out


def place_scores(models,g):
    out={}
    xf=feature_frame(g)
    priors=(models or {}).get('_priors',{}) if isinstance(models,dict) else {}
    for place in (1,2,3):
        if models is None or place not in models:
            raw=_fallback_scores(g,place)
        else:
            raw=models[place].predict_proba(xf)[:,1]
        raw=_blend_with_priors(raw,g,place,priors)
        raw=np.maximum(np.asarray(raw,float),1e-8)
        out[place]=raw
    return out


def race_data_quality(g):
    """Return readiness of the genuinely pre-race information.

    A race can still be scored with imputers, but it must not become a real buy
    candidate until course, exhibition ST and exhibition time are mostly present.
    """
    if g is None or len(g)==0:
        return {'ready':False,'factor':0.70,'preview_ratio':0.0,'weather_ratio':0.0,'reason':'展示データなし'}
    key=['course','preview_start_timing','exhibition_time']
    avail=[]
    for c in key:
        if c in g:avail.append(float(pd.to_numeric(g[c],errors='coerce').notna().mean()))
        else:avail.append(0.0)
    preview_ratio=float(np.mean(avail))
    weather=[]
    for c in ['wind_speed','wave_height_cm','air_temperature','water_temperature']:
        weather.append(float(pd.to_numeric(g.get(c,pd.Series(index=g.index,dtype=float)),errors='coerce').notna().mean()))
    weather_ratio=float(np.mean(weather))
    factor=float(np.clip(0.72+0.23*preview_ratio+0.05*weather_ratio,0.72,1.0))
    ready=bool(min(avail)>=5/6 and preview_ratio>=0.90)
    reason='OK' if ready else f'展示/ST/進入不足 ({preview_ratio*100:.0f}%)'
    return {'ready':ready,'factor':factor,'preview_ratio':preview_ratio,'weather_ratio':weather_ratio,'reason':reason}


def trifecta_table(models,g):
    scores=place_scores(models,g)
    lanes=g['lane'].astype(int).tolist()
    idx={lane:i for i,lane in enumerate(lanes)}
    rows=[]
    for a,b,c in itertools.permutations(lanes,3):
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
