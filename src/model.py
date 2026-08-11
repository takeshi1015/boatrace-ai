import itertools, numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
FEATURES=['lane','rank_number','age','weight','flying_count','late_count','avg_start','racer_win_rate','racer_2ren_rate','racer_3ren_rate','local_win_rate','local_2ren_rate','motor_2ren_rate','boat_2ren_rate','course','preview_start_timing','exhibition_time','wind_speed','wave_height_cm','air_temperature','water_temperature']
def X(df):
    a=df.copy()
    for c in FEATURES:
        if c not in a:a[c]=np.nan
        a[c]=pd.to_numeric(a[c],errors='coerce')
    return a[FEATURES]
def fit_model(hist):
    if hist.empty or hist['win'].sum()<20:return None
    m=Pipeline([('imp',SimpleImputer(strategy='median')),('clf',HistGradientBoostingClassifier(max_iter=120,max_depth=5,learning_rate=.06,l2_regularization=1.0,random_state=42))])
    m.fit(X(hist),hist['win'].astype(int)); return m
def race_probs(model,g):
    if model is None:
        wr=pd.to_numeric(g.get('racer_win_rate'),errors='coerce').fillna(5); lane=pd.to_numeric(g['lane'],errors='coerce'); z=np.exp(0.45*wr+0.35*(7-lane))
    else:z=model.predict_proba(X(g))[:,1]
    z=np.maximum(np.asarray(z,float),1e-8); return z/z.sum()
def trifecta_table(g,p):
    q=dict(zip(g['lane'].astype(int),p)); out=[]
    for a,b,c in itertools.permutations(sorted(q),3):
        pa=q[a]; pb=q[b]/max(1e-9,1-q[a]); pc=q[c]/max(1e-9,1-q[a]-q[b]); out.append({'combo':f'{a}-{b}-{c}','prob':max(0.0,float(pa*pb*pc))})
    s=sum(x['prob'] for x in out)
    if s>0:
        for x in out:x['prob']/=s
    return pd.DataFrame(out).sort_values('prob',ascending=False).reset_index(drop=True)
