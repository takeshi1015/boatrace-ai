from __future__ import annotations
import pandas as pd
from .model import fit_models,trifecta_table

def walk_forward_backtest(hist,min_train_days=35,test_days=7,max_test_days=28):
    """
    Train only on dates before each test window.
    One 100-yen bet per race on model's top trifecta.
    ROI uses official trifecta payout for a winning 100-yen ticket.
    """
    if hist is None or hist.empty:
        return pd.DataFrame(),{}
    h=hist.copy()
    h['race_date_dt']=pd.to_datetime(h['race_date'],errors='coerce')
    h=h[h['race_date_dt'].notna() & h['finish_num'].notna()]
    dates=sorted(h['race_date_dt'].dt.normalize().drop_duplicates())
    if len(dates)<min_train_days+3:
        return pd.DataFrame(),{}

    test_dates=dates[-min(max_test_days,len(dates)-min_train_days):]
    # Split the final evaluation period into small windows.
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
                pred=str(top['combo'])
                act=str(actual.iloc[0])
                hit=pred==act
                pay=float(payout.iloc[0]) if len(payout) else 0.0
                rows.append({
                    'race_id':rid,'race_date':g['race_date'].iloc[0],
                    'venue':g['venue'].iloc[0],'race_no':int(g['race_no'].iloc[0]),
                    'pred':pred,'prob':float(top['prob']),'actual':act,'hit':hit,
                    'payout':pay,'stake':100.0,'return_yen':pay if hit else 0.0
                })
            except Exception:
                pass

    res=pd.DataFrame(rows)
    if res.empty:
        return res,{}
    stake=res['stake'].sum()
    ret=res['return_yen'].sum()
    metrics={
        'races':len(res),
        'hit_rate':float(res['hit'].mean()),
        'roi':float(ret/stake) if stake>0 else 0.0,
        'profit':float(ret-stake),
        'avg_prob':float(res['prob'].mean()),
    }
    return res,metrics
