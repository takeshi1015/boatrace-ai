from __future__ import annotations
import math
import numpy as np
import pandas as pd

TICKET_CODES=("3t","3f","2t","2f")
TICKET_NAMES={"3t":"3連単","3f":"3連複","2t":"2連単","2f":"2連複"}


def _safe_num(x, default=np.nan):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default


def lane1_band(p):
    p=_safe_num(p)
    if not math.isfinite(p): return "不明"
    if p>=0.55: return "1号艇かなり強い"
    if p>=0.40: return "1号艇強め"
    if p>=0.25: return "1号艇五分"
    return "1号艇弱め"


def race_band(r):
    try:r=int(r)
    except Exception:return "不明"
    if r<=4:return "1-4R"
    if r<=8:return "5-8R"
    return "9-12R"


def wind_band(v):
    v=_safe_num(v)
    if not math.isfinite(v): return "不明"
    if v<2:return "0-1m"
    if v<4:return "2-3m"
    if v<6:return "4-5m"
    return "6m以上"


def wave_band(v):
    v=_safe_num(v)
    if not math.isfinite(v): return "不明"
    if v<=2:return "0-2cm"
    if v<=5:return "3-5cm"
    if v<=10:return "6-10cm"
    return "11cm以上"


def market_odds_band(v):
    v=_safe_num(v)
    if not math.isfinite(v): return "不明"
    if v<3:return "3倍未満"
    if v<10:return "3-9.9倍"
    if v<30:return "10-29.9倍"
    if v<100:return "30-99.9倍"
    if v<300:return "100-299倍"
    return "300倍以上"


def market_popularity_band(rank, total):
    try:
        rank=int(rank); total=max(1,int(total))
    except Exception:return "不明"
    q=rank/total
    if q<=0.10:return "上位10%"
    if q<=0.25:return "上位25%"
    if q<=0.50:return "中位"
    if q<=0.75:return "下位25-50%"
    return "下位25%"


def add_context_columns(preds:pd.DataFrame)->pd.DataFrame:
    p=preds.copy()
    if p.empty:return p
    p['venue_ctx']=p.get('venue',pd.Series(index=p.index,dtype=object)).fillna('不明').astype(str)
    p['race_ctx']=p.get('race_no',pd.Series(index=p.index,dtype=float)).map(race_band)
    p['lane1_ctx']=p.get('lane1_first_prob',pd.Series(index=p.index,dtype=float)).map(lane1_band)
    p['wind_ctx']=p.get('wind_speed',pd.Series(index=p.index,dtype=float)).map(wind_band)
    p['wave_ctx']=p.get('wave_height_cm',pd.Series(index=p.index,dtype=float)).map(wave_band)
    return p


def current_race_context(g, tri):
    lane1_prob=0.0
    try:
        lane1_prob=float(tri.loc[tri['combo'].astype(str).str.startswith('1-'),'prob'].sum())
    except Exception:pass
    def first_col(name):
        try:return pd.to_numeric(g[name],errors='coerce').dropna().iloc[0]
        except Exception:return np.nan
    return {
        'venue_ctx':str(g['venue'].iloc[0]) if 'venue' in g and len(g) else '不明',
        'race_ctx':race_band(g['race_no'].iloc[0] if 'race_no' in g and len(g) else np.nan),
        'lane1_ctx':lane1_band(lane1_prob),
        'wind_ctx':wind_band(first_col('wind_speed')),
        'wave_ctx':wave_band(first_col('wave_height_cm')),
        'lane1_first_prob':lane1_prob,
    }


def _group_stats(df, prob_col, hit_col, group_col, prior=60, min_samples=25):
    rows=[]
    if group_col not in df or prob_col not in df or hit_col not in df:return rows
    for key,g in df.groupby(group_col,dropna=False):
        pr=pd.to_numeric(g[prob_col],errors='coerce')
        hh=pd.Series(g[hit_col],index=g.index).astype(float)
        m=pr.notna() & hh.notna()
        pr=pr[m]; hh=hh[m]
        n=len(pr)
        if n==0:continue
        mp=float(pr.mean()); hr=float(hh.mean())
        ratio=hr/mp if mp>0 else 1.0
        shrunk=(ratio*n + 1.0*prior)/(n+prior)
        # never boost above 1.0; poor contexts get a meaningful penalty
        factor=float(np.clip(shrunk,0.55,1.0))
        quality='十分' if n>=min_samples else '少数'
        rows.append({'key':str(key),'samples':int(n),'mean_pred':mp,'hit_rate':hr,'ratio':ratio,'factor':factor,'quality':quality})
    return rows


def fit_context_reliability(walk_forward_preds, min_samples=25):
    p=add_context_columns(walk_forward_preds if walk_forward_preds is not None else pd.DataFrame())
    out={}
    dims=('venue_ctx','race_ctx','lane1_ctx','wind_ctx','wave_ctx')
    for code in TICKET_CODES:
        prob=f'{code}_prob'; hit=f'{code}_hit'
        out[code]={}
        if p.empty or prob not in p or hit not in p:continue
        for dim in dims:
            stats=_group_stats(p,prob,hit,dim,prior=60,min_samples=min_samples)
            out[code][dim]={r['key']:r for r in stats}
    return out


def context_factor(code, ctx, context_stats, min_samples=25):
    dims=('venue_ctx','race_ctx','lane1_ctx','wind_ctx','wave_ctx')
    factors=[]; details=[]
    for dim in dims:
        key=str(ctx.get(dim,'不明'))
        stat=((context_stats or {}).get(code,{}) or {}).get(dim,{}).get(key)
        if not stat:continue
        f=float(stat.get('factor',1.0))
        n=int(stat.get('samples',0))
        # small groups are only allowed to apply a mild penalty
        if n<min_samples:
            f=max(f,0.90)
        factors.append(f)
        details.append({'dimension':dim,'condition':key,**stat})
    # use the two worst independent conditions, but damp combination with sqrt
    if not factors:return 1.0,details
    worst=sorted(factors)[:2]
    combined=float(np.clip(math.sqrt(np.prod(worst)),0.55,1.0))
    return combined,details


def apply_context_factor(df, factor):
    out=df.copy()
    if 'context_base_prob' not in out:
        out['context_base_prob']=pd.to_numeric(out['prob'],errors='coerce')
    out['context_factor']=float(factor)
    out['prob']=np.clip(pd.to_numeric(out['prob'],errors='coerce').fillna(0)*float(factor),1e-6,1-1e-6)
    if 'odds' in out:
        out['expected_value']=out['prob']*pd.to_numeric(out['odds'],errors='coerce')
        out['prob_pct']=out['prob']*100
    return out


def market_risk_factor(odds, rank, total):
    """Conservative live-market overlay.

    Historical pre-race odds are not yet persisted for enough races, so this is
    intentionally a risk cap, not a learned historical factor. It can only lower
    today's probability and will be replaced by learned market factors once the
    app has accumulated enough archived pre-race odds.
    """
    ob=market_odds_band(odds); pb=market_popularity_band(rank,total)
    f=1.0
    if ob=='100-299倍':f*=0.90
    elif ob=='300倍以上':f*=0.80
    if pb=='下位25-50%':f*=0.95
    elif pb=='下位25%':f*=0.88
    return float(np.clip(f,0.70,1.0)),ob,pb


def apply_market_overlay(df):
    out=df.copy()
    if out.empty:return out
    out['odds']=pd.to_numeric(out.get('odds'),errors='coerce')
    ranked=out['odds'].rank(method='min',ascending=True,na_option='bottom')
    total=int(out['odds'].notna().sum())
    factors=[]; obs=[]; pbs=[]
    for i,row in out.iterrows():
        rank=int(ranked.loc[i]) if pd.notna(ranked.loc[i]) else total
        f,ob,pb=market_risk_factor(row.get('odds'),rank,max(total,1))
        factors.append(f);obs.append(ob);pbs.append(pb)
    out['market_factor']=factors;out['odds_band']=obs;out['popularity_band']=pbs
    out['prob']=np.clip(pd.to_numeric(out['prob'],errors='coerce').fillna(0)*out['market_factor'],1e-6,1-1e-6)
    out['prob_pct']=out['prob']*100
    out['expected_value']=out['prob']*out['odds']
    return out
