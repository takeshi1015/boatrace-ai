from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS = [
    "recorded_at","race_id","race_date","venue","race_no","closed_at",
    "combo","pred_prob","odds","expected_value","confidence",
    "actual_combo","actual_payout","hit","stake_yen","return_yen","profit_yen",
    "status","miss_type"
]

def empty_ledger():
    return pd.DataFrame(columns=COLUMNS)

def load_ledger(path="data/prediction_log.csv"):
    p=Path(path)
    if not p.exists():
        return empty_ledger()
    try:
        df=pd.read_csv(p)
        for c in COLUMNS:
            if c not in df:
                df[c]=pd.NA
        return df[COLUMNS]
    except Exception:
        return empty_ledger()

def save_ledger(df,path="data/prediction_log.csv"):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    df.to_csv(p,index=False,encoding="utf-8-sig")

def upsert_predictions(ledger,preds,recorded_at,stake_yen=100):
    if preds is None or preds.empty:
        return ledger
    rows=[]
    existing=set(zip(
        ledger.get("race_id",pd.Series(dtype=str)).astype(str),
        ledger.get("combo",pd.Series(dtype=str)).astype(str)
    ))
    for _,r in preds.iterrows():
        key=(str(r["race_id"]),str(r["推奨3連単"]))
        if key in existing:
            continue
        rows.append({
            "recorded_at":recorded_at,
            "race_id":r["race_id"],
            "race_date":str(r.get("race_date","")),
            "venue":r["場"],
            "race_no":int(r["R"]),
            "closed_at":r["締切"],
            "combo":r["推奨3連単"],
            "pred_prob":float(r["予測確率%"])/100.0,
            "odds":r.get("実オッズ"),
            "expected_value":r.get("期待値"),
            "confidence":r.get("確信度"),
            "actual_combo":pd.NA,
            "actual_payout":pd.NA,
            "hit":pd.NA,
            "stake_yen":stake_yen,
            "return_yen":pd.NA,
            "profit_yen":pd.NA,
            "status":"pending",
            "miss_type":pd.NA,
        })
    if rows:
        ledger=pd.concat([ledger,pd.DataFrame(rows)],ignore_index=True)
    return ledger

def classify_miss(pred,actual):
    if not pred or not actual:
        return "結果未確定"
    try:
        p=pred.split("-"); a=actual.split("-")
        if p[0] != a[0]:
            return "1着予測外れ"
        if p[1:] == a[1:]:
            return "的中"
        if set(p[1:]) == set(a[1:]):
            return "2・3着順違い"
        if p[1] == a[1]:
            return "3着外れ"
        if p[2] == a[2]:
            return "2着外れ"
        return "相手艇外れ"
    except Exception:
        return "組合せ外れ"

def apply_results(ledger,result_df):
    if ledger.empty or result_df is None or result_df.empty:
        return ledger
    result_map={}
    for rid,g in result_df.groupby("race_id"):
        actual=g["trifecta_result"].dropna()
        payout=pd.to_numeric(g["trifecta_payout"],errors="coerce").dropna()
        if len(actual):
            result_map[str(rid)]=(
                str(actual.iloc[0]),
                float(payout.iloc[0]) if len(payout) else None
            )
    for i,row in ledger.iterrows():
        if str(row["status"])=="settled":
            continue
        rid=str(row["race_id"])
        if rid not in result_map:
            continue
        actual,payout=result_map[rid]
        pred=str(row["combo"])
        hit=(pred==actual)
        stake=float(row["stake_yen"]) if pd.notna(row["stake_yen"]) else 100.0
        ret=float(payout) if hit and payout is not None else 0.0
        ledger.at[i,"actual_combo"]=actual
        ledger.at[i,"actual_payout"]=payout
        ledger.at[i,"hit"]=bool(hit)
        ledger.at[i,"return_yen"]=ret
        ledger.at[i,"profit_yen"]=ret-stake
        ledger.at[i,"status"]="settled"
        ledger.at[i,"miss_type"]=classify_miss(pred,actual)
    return ledger
