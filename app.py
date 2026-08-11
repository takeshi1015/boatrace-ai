from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st

from src.data import fetch_today, flatten, historical_dataset, fetch_results_for_dates
from src.model import fit_models, trifecta_table, race_confidence
from src.odds import fetch_trifecta_odds
from src.ledger import load_ledger,save_ledger,upsert_predictions,apply_results
from src.backtest import walk_forward_backtest

JST=ZoneInfo("Asia/Tokyo")
st.set_page_config(page_title="BOAT RACE AI",layout="wide")
st.title("BOAT RACE AI 予想ダッシュボード")
st.caption("無料版 v2.6：3着別AI・結果再取得・自動精算・オッズ再試行・時系列バックテスト")

now=pd.Timestamp.now(tz=JST)
st.write(f"現在時刻：{now:%Y/%m/%d %H:%M:%S}")

@st.cache_data(ttl=21600,show_spinner="過去実戦データを読み込んでいます…")
def load_hist():
    return historical_dataset(90)

@st.cache_resource(ttl=21600,show_spinner="1着・2着・3着AIを学習しています…")
def load_models():
    return fit_models(load_hist())

@st.cache_data(ttl=120,show_spinner=False)
def odds_cached(d,v,r):
    return fetch_trifecta_odds(d,v,r,timeout=7)

@st.cache_data(ttl=21600,show_spinner="時系列バックテスト中…")
def cached_backtest():
    return walk_forward_backtest(load_hist(),min_train_days=35,test_days=7,max_test_days=28)

try:
    today=flatten(fetch_today(),False)
except Exception:
    st.error("本日の実レースデータを取得できません。")
    st.stop()
if today.empty:
    st.warning("本日のデータがありません。")
    st.stop()

hist=load_hist()
models=load_models()
st.success(f'過去実戦データ {hist["race_id"].nunique() if not hist.empty else 0:,}レースを読み込みました。')
if models is None:
    st.warning("学習データ不足のため本番予想を停止しています。")
    st.stop()

dt=pd.to_datetime(today["closed_at"],errors="coerce")
if dt.dt.tz is None:
    dt=dt.dt.tz_localize(JST,nonexistent="shift_forward",ambiguous="NaT")
else:
    dt=dt.dt.tz_convert(JST)
today["closed_at_jst"]=dt
today["minutes_left"]=(dt-now).dt.total_seconds()/60
valid=today[(today["minutes_left"]>=10)&(today["minutes_left"]<=360)].copy()

base_rows=[]
base_detail={}
for rid,g in valid.groupby("race_id"):
    try:
        tri=trifecta_table(models,g)
        base_detail[rid]=tri
        top=tri.iloc[0]
        base_rows.append({
            "race_id":rid,"race_date":g["race_date"].iloc[0],
            "場":g["venue"].iloc[0],"R":int(g["race_no"].iloc[0]),
            "締切":pd.Timestamp(g["closed_at_jst"].iloc[0]).strftime("%H:%M"),
            "残り分":float(g["minutes_left"].iloc[0]),
            "確率1位":top["combo"],"確率1位%":float(top["prob"]*100),
            "確信度":race_confidence(tri)
        })
    except Exception:
        pass

base=pd.DataFrame(base_rows)
st.subheader("AI一次評価（締切まで10分以上）")
if base.empty:
    st.info("現在、条件を満たすレースはありません。")
    st.stop()
st.dataframe(base.sort_values(["残り分","確信度"],ascending=[True,False]),use_container_width=True,hide_index=True)

# Max 24: near-term purchase candidates, mixed confidence.
pool=base[base["残り分"]<=240].copy()
if len(pool)>24:
    safe_ids=pool.sort_values("確信度",ascending=False).head(12)["race_id"].tolist()
    broad_ids=pool.sort_values("残り分").head(12)["race_id"].tolist()
    selected=list(dict.fromkeys(safe_ids+broad_ids))[:24]
    pool=pool[pool["race_id"].isin(selected)].copy()

def fetch_one(row, retry=False):
    d=pd.Timestamp(row["race_date"]).strftime("%Y%m%d")
    if retry:
        # uncached longer retry
        odds,url,diag=fetch_trifecta_odds(d,row["場"],int(row["R"]),timeout=12)
    else:
        odds,url,diag=odds_cached(d,row["場"],int(row["R"]))
    return row["race_id"],odds,url,diag

odds_map={}
with st.status("実オッズを取得しています…",expanded=False) as status:
    rows=[r for _,r in pool.iterrows()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures=[ex.submit(fetch_one,r,False) for r in rows]
        for fut in as_completed(futures):
            try:
                rid,odds,url,diag=fut.result()
                odds_map[rid]=(odds,url,diag)
            except Exception:
                pass

    # Retry only missing/incomplete odds once, uncached and longer timeout.
    missing=[]
    for _,r in pool.iterrows():
        rid=r["race_id"]
        odds=odds_map.get(rid,(pd.DataFrame(),None,[]))[0]
        if odds is None or len(odds)<100:
            missing.append(r)
    if missing:
        status.update(label=f"未取得 {len(missing)}レースを再試行しています…",state="running")
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures=[ex.submit(fetch_one,r,True) for r in missing]
            for fut in as_completed(futures):
                try:
                    rid,odds,url,diag=fut.result()
                    old=odds_map.get(rid,(pd.DataFrame(),None,[]))
                    if odds is not None and len(odds)>len(old[0]):
                        odds_map[rid]=(odds,url,(old[2] or [])+(diag or []))
                    else:
                        odds_map[rid]=(old[0],old[1],(old[2] or [])+(diag or []))
                except Exception:
                    pass
    status.update(label="実オッズ取得処理が完了しました",state="complete")

race_rows=[]
detail={}
for _,row in pool.iterrows():
    rid=row["race_id"]
    tri=base_detail[rid].copy()
    odds,url,diag=odds_map.get(rid,(pd.DataFrame(columns=["combo","odds"]),None,[]))
    if odds is None or odds.empty:
        odds=pd.DataFrame(columns=["combo","odds"])
    odds=odds[[c for c in ["combo","odds"] if c in odds.columns]].copy()
    if "combo" not in odds: odds["combo"]=pd.Series(dtype=object)
    if "odds" not in odds: odds["odds"]=pd.Series(dtype=float)
    tri=tri.merge(odds[["combo","odds"]],on="combo",how="left")
    tri["expected_value"]=tri["prob"]*tri["odds"]
    detail[rid]=tri

    top=tri.iloc[0]
    ev=tri.dropna(subset=["expected_value"]).sort_values("expected_value",ascending=False)
    best=ev.iloc[0] if len(ev) else top
    race_rows.append({
        "race_id":rid,"race_date":row["race_date"],"場":row["場"],"R":row["R"],
        "締切":row["締切"],"残り分":row["残り分"],"確信度":row["確信度"],
        "確率1位":top["combo"],"確率1位%":float(top["prob"]*100),
        "推奨3連単":best["combo"],"予測確率%":float(best["prob"]*100),
        "実オッズ":best.get("odds",np.nan),"期待値":best.get("expected_value",np.nan),
        "オッズ取得":bool(len(odds)>=100),"取得組合せ数":int(len(odds)),
        "取得経路":diag[-1].get("route") if diag else ""
    })
races=pd.DataFrame(race_rows)

st.subheader("最終評価対象")
st.dataframe(races,use_container_width=True,hide_index=True)
if not races.empty:
    ok=races[races["オッズ取得"]]
    st.success(f"実オッズ100通り以上取得：{len(ok)}/{len(races)}レース")

safe=races.copy()
safe=safe[safe["予測確率%"]>=2.0]
safe=safe[(safe["期待値"].isna())|(safe["期待値"]>=0.90)]
safe=safe.sort_values(["確信度","予測確率%"],ascending=False).head(10)

holes=races.dropna(subset=["期待値"]).copy()
holes=holes[(holes["予測確率%"]>=1.0)&(holes["期待値"]>=1.05)]
holes=holes.sort_values(["期待値","予測確率%"],ascending=False).head(10)

# Prediction ledger
ledger=load_ledger()
recordable=races[
    races["オッズ取得"] & races["期待値"].notna() & races["実オッズ"].notna()
].copy()
ledger=upsert_predictions(
    ledger,recordable,now.strftime("%Y-%m-%d %H:%M:%S"),stake_yen=100
)

# Reliable result retrieval: retry every pending date, not only today's in-memory data.
pending_dates=ledger.loc[ledger["status"]=="pending","race_date"].dropna().tolist()
if pending_dates:
    result_df=fetch_results_for_dates(pending_dates)
    ledger=apply_results(
        ledger,result_df,settled_at=now.strftime("%Y-%m-%d %H:%M:%S")
    )
save_ledger(ledger)

c1,c2=st.columns(2)
with c1:
    st.subheader("堅め候補 TOP10")
    st.dataframe(safe[["場","R","締切","残り分","推奨3連単","予測確率%","実オッズ","期待値","確信度"]],use_container_width=True,hide_index=True)
with c2:
    st.subheader("穴候補 TOP10（期待値順）")
    if holes.empty:
        st.info("条件を満たす穴候補はありません。")
    else:
        st.dataframe(holes[["場","R","締切","残り分","推奨3連単","予測確率%","実オッズ","期待値","確信度"]],use_container_width=True,hide_index=True)

st.subheader("実戦成績・自動精算")
settled=ledger[ledger["status"]=="settled"].copy()
pending=ledger[ledger["status"]=="pending"].copy()
m1,m2,m3,m4=st.columns(4)
m1.metric("記録予想数",f"{len(ledger):,}")
m2.metric("結果確定",f"{len(settled):,}")
if len(settled):
    hit=pd.to_numeric(settled["hit"],errors="coerce").fillna(False).astype(bool)
    stake=pd.to_numeric(settled["stake_yen"],errors="coerce").fillna(0).sum()
    ret=pd.to_numeric(settled["return_yen"],errors="coerce").fillna(0).sum()
    m3.metric("的中率",f"{hit.mean()*100:.1f}%")
    m4.metric("回収率",f"{(ret/stake*100 if stake else 0):.1f}%")
    profit=pd.to_numeric(settled["profit_yen"],errors="coerce").fillna(0).sum()
    st.write(f"100円/予想換算の累計収支：**{profit:+,.0f}円**")
    miss=settled[~hit]
    if len(miss):
        st.subheader("外れ原因")
        st.dataframe(miss["miss_type"].fillna("不明").value_counts().rename_axis("外れ方").reset_index(name="件数"),use_container_width=True,hide_index=True)
else:
    m3.metric("的中率","—"); m4.metric("回収率","—")
st.caption(f"結果待ち：{len(pending)}件。未確定レースは次回表示時にも自動再取得します。")

st.download_button(
    "成績CSVを保存",
    data=ledger.to_csv(index=False).encode("utf-8-sig"),
    file_name="boatrace_ai_prediction_log_v26.csv",
    mime="text/csv"
)

st.subheader("3連単120通りのAI確率・期待値")
if detail:
    rid=st.selectbox("レースを選択",list(detail.keys()))
    t=detail[rid].copy()
    t["prob_pct"]=t["prob"]*100
    st.dataframe(t[["combo","prob_pct","odds","expected_value"]].sort_values("expected_value",ascending=False,na_position="last"),use_container_width=True,hide_index=True)

st.subheader("時系列バックテスト（未来データを学習に使わない）")
bt,metrics=cached_backtest()
if metrics:
    b1,b2,b3,b4=st.columns(4)
    b1.metric("検証レース",f'{metrics["races"]:,}')
    b2.metric("3連単1点的中率",f'{metrics["hit_rate"]*100:.1f}%')
    b3.metric("100円1点回収率",f'{metrics["roi"]*100:.1f}%')
    b4.metric("検証収支",f'{metrics["profit"]:+,.0f}円')
    st.caption("各検証期間より前のデータだけで学習しています。払戻は公式3連単払戻額を100円購入として計算します。")
    st.dataframe(bt.tail(50),use_container_width=True,hide_index=True)
else:
    st.info("バックテストに必要な期間のデータが不足しています。")

st.subheader("AI再学習")
if st.button("最新結果で再学習する"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("キャッシュをクリアしました。ページを再読み込みすると最新結果で再学習します。")

st.caption("予測は統計モデルによる推定で、的中・利益を保証しません。実オッズ未取得時は期待値を推測しません。")
