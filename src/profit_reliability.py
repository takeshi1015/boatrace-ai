from __future__ import annotations
import math
import numpy as np
import pandas as pd

def _safe_num(x, default=np.nan):
    try:
        v=float(x)
        return v if math.isfinite(v) else default
    except Exception:
        return default

def _race_band(r):
    try:r=int(r)
    except Exception:return "不明"
    if r<=4:return "1-4R"
    if r<=8:return "5-8R"
    return "9-12R"

def _lane1_band(p):
    p=_safe_num(p)
    if not math.isfinite(p): return "不明"
    if p>=0.55:return "1号艇かなり強い"
    if p>=0.40:return "1号艇強め"
    if p>=0.25:return "1号艇五分"
    return "1号艇弱め"

def _prob_band(p):
    p=_safe_num(p)
    if not math.isfinite(p):return "不明"
    if p>=0.10:return "10%以上"
    if p>=0.06:return "6-10%"
    if p>=0.035:return "3.5-6%"
    if p>=0.02:return "2-3.5%"
    return "2%未満"

def _conf_band(v):
    v=_safe_num(v)
    if not math.isfinite(v):return "不明"
    if v>=0.75:return "高"
    if v>=0.50:return "中高"
    if v>=0.30:return "中"
    return "低"

def _wind_band(v):
    v=_safe_num(v)
    if not math.isfinite(v):return "不明"
    if v<2:return "0-1m"
    if v<4:return "2-3m"
    if v<6:return "4-5m"
    return "6m以上"

def _wave_band(v):
    v=_safe_num(v)
    if not math.isfinite(v):return "不明"
    if v<=2:return "0-2cm"
    if v<=5:return "3-5cm"
    if v<=10:return "6-10cm"
    return "11cm以上"

def add_profit_context(df):
    x=df.copy()
    if x.empty:return x
    x["profit_venue"]=x.get("venue",pd.Series(index=x.index,dtype=object)).fillna("不明").astype(str)
    x["profit_race_band"]=x.get("race_no",pd.Series(index=x.index,dtype=float)).map(_race_band)
    x["profit_lane1_band"]=x.get("lane1_first_prob",pd.Series(index=x.index,dtype=float)).map(_lane1_band)
    x["profit_first_lane"]=x.get("first_lane",pd.Series(index=x.index,dtype=float)).map(
        lambda v: f"{int(v)}号艇" if pd.notna(v) else "不明"
    )
    x["profit_prob_band"]=x.get("prob",pd.Series(index=x.index,dtype=float)).map(_prob_band)
    x["profit_conf_band"]=x.get("confidence",pd.Series(index=x.index,dtype=float)).map(_conf_band)
    x["profit_wind_band"]=x.get("wind_speed",pd.Series(index=x.index,dtype=float)).map(_wind_band)
    x["profit_wave_band"]=x.get("wave_height_cm",pd.Series(index=x.index,dtype=float)).map(_wave_band)
    return x

DIMS=(
    "profit_venue","profit_race_band","profit_lane1_band","profit_first_lane",
    "profit_prob_band","profit_conf_band","profit_wind_band","profit_wave_band",
)

def _roi_stat(g, prior=100):
    n=len(g)
    if n<=0:return None
    ret=pd.to_numeric(g.get("return_yen"),errors="coerce").fillna(0).sum()
    roi=float(ret/(100.0*n))
    hit=float(pd.Series(g.get("hit"),index=g.index).astype(float).mean()) if "hit" in g else 0.0
    shrunk=float((roi*n + 1.0*prior)/(n+prior))
    # This factor can only reduce confidence, never increase it.
    if shrunk < 0.65: factor=0.70
    elif shrunk < 0.78: factor=0.80
    elif shrunk < 0.90: factor=0.90
    elif shrunk < 0.98: factor=0.96
    else: factor=1.00
    return {
        "samples":int(n),"roi":roi,"hit_rate":hit,"shrunk_roi":shrunk,
        "factor":float(factor),
        "hard_block":bool(n>=70 and shrunk<0.72),
    }

def fit_race_profit_filter(train, min_samples=30, prior=100):
    """Fit race-level profitability conditions using historical OOS predictions only."""
    x=add_profit_context(train if train is not None else pd.DataFrame())
    model={"dims":{},"min_samples":int(min_samples),"prior":int(prior)}
    if x.empty:return model
    for dim in DIMS:
        model["dims"][dim]={}
        if dim not in x:continue
        for key,g in x.groupby(dim,dropna=False):
            st=_roi_stat(g,prior=prior)
            if st is None:continue
            st["quality"]="十分" if st["samples"]>=min_samples else "少数"
            model["dims"][dim][str(key)]=st
    return model

def _row_context(row):
    return {
        "profit_venue":str(row.get("venue","不明")),
        "profit_race_band":_race_band(row.get("race_no")),
        "profit_lane1_band":_lane1_band(row.get("lane1_first_prob")),
        "profit_first_lane":f"{int(row.get('first_lane'))}号艇" if pd.notna(row.get("first_lane")) else "不明",
        "profit_prob_band":_prob_band(row.get("prob")),
        "profit_conf_band":_conf_band(row.get("confidence")),
        "profit_wind_band":_wind_band(row.get("wind_speed")),
        "profit_wave_band":_wave_band(row.get("wave_height_cm")),
    }

def row_profit_factor(row, model, min_samples=30):
    """Combine at most two worst supported conditions to avoid over-penalization."""
    ctx=_row_context(row)
    supported=[];details=[];blocked=[]
    dims=(model or {}).get("dims",{})
    for dim,key in ctx.items():
        stat=(dims.get(dim,{}) or {}).get(str(key))
        if not stat:continue
        n=int(stat.get("samples",0))
        f=float(stat.get("factor",1.0))
        if n<min_samples:
            # small groups can only have a mild effect
            f=max(f,0.96)
        supported.append(f)
        d={"dimension":dim,"condition":key,**stat}
        details.append(d)
        if bool(stat.get("hard_block")) and n>=min_samples:
            blocked.append(d)
    if not supported:
        return 1.0,False,details
    worst=sorted(supported)[:2]
    combined=float(np.clip(math.sqrt(np.prod(worst)),0.70,1.0))
    return combined,bool(blocked),details

def apply_race_profit_filter(df, model, min_factor=0.88, min_samples=30):
    if df is None or df.empty:return df.copy()
    x=df.copy()
    factors=[];blocks=[]
    for _,r in x.iterrows():
        f,b,_=row_profit_factor(r,model,min_samples=min_samples)
        factors.append(f);blocks.append(b)
    x["profit_context_factor"]=factors
    x["profit_context_block"]=blocks
    return x[(x["profit_context_factor"]>=float(min_factor)) & (~x["profit_context_block"])].copy()

def current_profit_context(venue,race_no,lane1_first_prob,first_lane,prob,confidence,wind_speed,wave_height_cm):
    return {
        "venue":venue,"race_no":race_no,"lane1_first_prob":lane1_first_prob,
        "first_lane":first_lane,"prob":prob,"confidence":confidence,
        "wind_speed":wind_speed,"wave_height_cm":wave_height_cm,
    }

def summarize_profit_model(model, topn=8):
    rows=[]
    for dim,mp in ((model or {}).get("dims",{}) or {}).items():
        for key,st in (mp or {}).items():
            if int(st.get("samples",0))<30:continue
            rows.append({
                "条件種別":dim,"条件":key,"件数":int(st.get("samples",0)),
                "実回収率":float(st.get("roi",0)),
                "縮小回収率":float(st.get("shrunk_roi",0)),
                "補正係数":float(st.get("factor",1)),
                "除外":bool(st.get("hard_block",False)),
            })
    if not rows:return pd.DataFrame()
    out=pd.DataFrame(rows).sort_values(["縮小回収率","件数"],ascending=[True,False])
    return out.head(int(topn)).reset_index(drop=True)
