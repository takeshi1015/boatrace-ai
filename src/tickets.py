from __future__ import annotations
import itertools
import pandas as pd

TICKET_NAMES={'3t':'3連単','3f':'3連複','2t':'2連単','2f':'2連複'}
MIN_PROB={'3t':0.008,'3f':0.030,'2t':0.030,'2f':0.050}

def _parts(combo):
    return tuple(int(x) for x in str(combo).replace('=','-').split('-'))

def probability_tables_from_trifecta(tri):
    """Derive all requested ticket probabilities from the 120 trifecta distribution."""
    t=tri[['combo','prob']].copy()
    t['prob']=pd.to_numeric(t['prob'],errors='coerce').fillna(0.0)
    out={'3t':t.copy()}

    # Trio / 3連複: sum all six orders for the same three boats.
    rows=[]
    for _,r in t.iterrows():
        p=tuple(sorted(_parts(r['combo'])))
        rows.append({'combo':'-'.join(map(str,p)),'prob':float(r['prob'])})
    out['3f']=pd.DataFrame(rows).groupby('combo',as_index=False)['prob'].sum()

    # Exacta / 2連単: sum over all possible third-place boats.
    rows=[]
    for _,r in t.iterrows():
        a,b,_=_parts(r['combo'])
        rows.append({'combo':f'{a}-{b}','prob':float(r['prob'])})
    out['2t']=pd.DataFrame(rows).groupby('combo',as_index=False)['prob'].sum()

    # Quinella / 2連複: ignore order of first two boats.
    rows=[]
    for _,r in out['2t'].iterrows():
        a,b=_parts(r['combo'])
        x=tuple(sorted((a,b)))
        rows.append({'combo':f'{x[0]}-{x[1]}','prob':float(r['prob'])})
    out['2f']=pd.DataFrame(rows).groupby('combo',as_index=False)['prob'].sum()

    for k,v in out.items():
        out[k]=v.sort_values('prob',ascending=False).reset_index(drop=True)
    return out

def merge_odds(prob_df, odds_df):
    p=prob_df.copy()
    p['combo']=p['combo'].astype(str).str.replace('=','-',regex=False).str.strip()
    o=odds_df.copy() if odds_df is not None else pd.DataFrame(columns=['combo','odds'])
    if 'combo' not in o: o['combo']=pd.Series(dtype=object)
    if 'odds' not in o: o['odds']=pd.Series(dtype=float)
    o['combo']=o['combo'].astype(str).str.replace('=','-',regex=False).str.strip()
    o['odds']=pd.to_numeric(o['odds'],errors='coerce')
    z=p.merge(o[['combo','odds']].drop_duplicates('combo',keep='last'),on='combo',how='left')
    z['expected_value']=z['prob']*z['odds']
    z['prob_pct']=z['prob']*100
    return z

def choose_ticket_candidates(ticket_tables):
    """Return a probability-oriented and EV-oriented candidate across ticket types."""
    safe_rows=[]; value_rows=[]
    for code,df in ticket_tables.items():
        if df is None or df.empty:
            continue
        x=df[df['odds'].notna() & (df['prob']>=MIN_PROB[code])].copy()
        if x.empty:
            continue
        safe_pool=x[x['expected_value']>=0.90]
        if safe_pool.empty: safe_pool=x
        s=safe_pool.sort_values(['prob','expected_value'],ascending=False).iloc[0].to_dict()
        s.update({'ticket_code':code,'ticket':TICKET_NAMES[code]})
        safe_rows.append(s)
        v=x.sort_values(['expected_value','prob'],ascending=False).iloc[0].to_dict()
        v.update({'ticket_code':code,'ticket':TICKET_NAMES[code]})
        value_rows.append(v)
    safe=None if not safe_rows else max(safe_rows,key=lambda r:(r['prob'],r.get('expected_value') or 0))
    value=None if not value_rows else max(value_rows,key=lambda r:(r.get('expected_value') or 0,r['prob']))
    return safe,value

def priority_candidate(safe,value):
    if safe is None and value is None: return None,'候補なし'
    if safe is None: return value,'高期待値案のみ取得'
    if value is None: return safe,'高確率案のみ取得'
    sp=float(safe['prob']); se=float(safe.get('expected_value') or 0)
    vp=float(value['prob']); ve=float(value.get('expected_value') or 0)
    # Favor an easier ticket when its probability is materially better and EV is not poor.
    if sp>=0.15 and se>=1.00 and ve<se*1.45:
        return safe,'高確率かつ期待値1.0以上を優先'
    if ve>=max(1.40,se*1.45):
        return value,'期待値が高確率案を大きく上回る'
    if sp>=vp*2.0 and se>=0.95:
        return safe,'的中確率が高期待値案の2倍以上'
    return value,'期待値優位を優先'
