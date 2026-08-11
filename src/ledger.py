from __future__ import annotations
from pathlib import Path
import pandas as pd

COLUMNS=[
    "recorded_at","race_id","race_date","venue","race_no","closed_at",
    "combo","pred_prob","odds","expected_value","confidence",
    "actual_combo","actual_payout","hit","stake_yen","return_yen","profit_yen",
    "status","miss_type","settled_at"
]

STRING_COLUMNS=[
    "recorded_at","race_id","race_date","venue","closed_at",
    "combo","actual_combo","status","miss_type","settled_at"
]
NUMERIC_COLUMNS=[
    "race_no","pred_prob","odds","expected_value","confidence",
    "actual_payout","stake_yen","return_yen","profit_yen"
]

def empty_ledger():
    df=pd.DataFrame(columns=COLUMNS)
    return normalize_ledger_types(df)

def normalize_ledger_types(df):
    """
    CSVから読み込むと、全件空欄だった actual_combo 等が float64 と推定される。
    その状態で '1-2-3' を代入すると pandas 2.x/3.x 系で TypeError になるため、
    書込み前に列型を明示的に正規化する。
    """
    if df is None:
        df=pd.DataFrame(columns=COLUMNS)
    df=df.copy()

    for c in COLUMNS:
        if c not in df.columns:
            df[c]=pd.NA

    # 組合せ・状態等は必ず object/string-compatible にする
    for c in STRING_COLUMNS:
        df[c]=df[c].astype("object")

    # 数値列は安全に数値へ
    for c in NUMERIC_COLUMNS:
        df[c]=pd.to_numeric(df[c],errors="coerce")

    # hit は True/False/NA を保持する object にして代入衝突を防ぐ
    if "hit" not in df:
        df["hit"]=pd.NA
    df["hit"]=df["hit"].astype("object")

    return df[COLUMNS]

def load_ledger(path="data/prediction_log.csv"):
    p=Path(path)
    if not p.exists():
        return empty_ledger()
    try:
        df=pd.read_csv(p)
        return normalize_ledger_types(df)
    except Exception:
        return empty_ledger()

def save_ledger(df,path="data/prediction_log.csv"):
    p=Path(path)
    p.parent.mkdir(parents=True,exist_ok=True)
    df=normalize_ledger_types(df)
    df.to_csv(p,index=False,encoding="utf-8-sig")

def upsert_predictions(ledger,preds,recorded_at,stake_yen=100):
    ledger=normalize_ledger_types(ledger)

    if preds is None or preds.empty:
        return ledger

    existing=set(zip(
        ledger.get("race_id",pd.Series(dtype=object)).astype(str),
        ledger.get("combo",pd.Series(dtype=object)).astype(str)
    ))

    rows=[]
    for _,r in preds.iterrows():
        key=(str(r["race_id"]),str(r["推奨3連単"]))
        if key in existing:
            continue

        rows.append({
            "recorded_at":str(recorded_at),
            "race_id":str(r["race_id"]),
            "race_date":str(r.get("race_date","")),
            "venue":str(r["場"]),
            "race_no":int(r["R"]),
            "closed_at":str(r["締切"]),
            "combo":str(r["推奨3連単"]),
            "pred_prob":float(r["予測確率%"])/100.0,
            "odds":r.get("実オッズ"),
            "expected_value":r.get("期待値"),
            "confidence":r.get("確信度"),
            "actual_combo":pd.NA,
            "actual_payout":pd.NA,
            "hit":pd.NA,
            "stake_yen":float(stake_yen),
            "return_yen":pd.NA,
            "profit_yen":pd.NA,
            "status":"pending",
            "miss_type":pd.NA,
            "settled_at":pd.NA
        })

    if rows:
        ledger=pd.concat([ledger,pd.DataFrame(rows)],ignore_index=True)

    return normalize_ledger_types(ledger)

def classify_miss(pred,actual):
    if not pred or not actual:
        return "結果未確定"
    try:
        p=str(pred).split("-")
        a=str(actual).split("-")
        if p==a:
            return "的中"
        if p[0]!=a[0]:
            return "1着予測外れ"
        if set(p[1:])==set(a[1:]):
            return "2・3着順違い"
        if p[1]==a[1]:
            return "3着外れ"
        if p[2]==a[2]:
            return "2着外れ"
        return "相手艇外れ"
    except Exception:
        return "組合せ外れ"

def apply_results(ledger,result_df,settled_at=None):
    ledger=normalize_ledger_types(ledger)

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
        is_hit=(pred==actual)

        stake=(
            float(row["stake_yen"])
            if pd.notna(row["stake_yen"])
            else 100.0
        )

        ret=(
            float(payout)
            if is_hit and payout is not None
            else 0.0
        )

        # 文字列列はobjectへ正規化済みなので安全に代入可能
        ledger.at[i,"actual_combo"]=str(actual)
        ledger.at[i,"actual_payout"]=payout
        ledger.at[i,"hit"]=bool(is_hit)
        ledger.at[i,"return_yen"]=float(ret)
        ledger.at[i,"profit_yen"]=float(ret-stake)
        ledger.at[i,"status"]="settled"
        ledger.at[i,"miss_type"]=classify_miss(pred,actual)
        ledger.at[i,"settled_at"]=str(settled_at) if settled_at is not None else pd.NA

    return normalize_ledger_types(ledger)
