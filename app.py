from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st

from src.data import fetch_today, flatten, historical_dataset, fetch_results_for_dates
from src.model import fit_models, trifecta_table, selection_signals
from src.odds import fetch_trifecta_odds
from src.ledger import load_ledger, save_ledger, upsert_predictions, apply_results
from src.backtest import (
    generate_walk_forward_predictions,
    nested_selector_backtest,
    deployment_gate,
    fit_current_selector,
    current_buy_score,
    summarize,
)

JST = ZoneInfo("Asia/Tokyo")
st.set_page_config(page_title="BOAT RACE AI", layout="wide")
st.title("BOAT RACE AI 予想ダッシュボード")
st.caption("無料版 v2.8：未見データ100%超ゲート・利益選別AI")

now = pd.Timestamp.now(tz=JST)
st.write(f"現在時刻：{now:%Y/%m/%d %H:%M:%S}")

@st.cache_data(ttl=21600, show_spinner="過去実戦データを読み込んでいます…")
def load_hist():
    return historical_dataset(90)

@st.cache_resource(ttl=21600, show_spinner="1着・2着・3着AIを学習しています…")
def load_models():
    return fit_models(load_hist())

@st.cache_data(ttl=120, show_spinner=False)
def odds_cached(d, v, r):
    return fetch_trifecta_odds(d, v, r, timeout=7)

@st.cache_data(ttl=21600, show_spinner="未見データで利益選別AIを検証しています…")
def validation_assets():
    preds = generate_walk_forward_predictions(
        load_hist(), min_train_days=35, test_days=7, max_test_days=49
    )
    selected, metrics, folds = nested_selector_backtest(
        preds, lookback_days=28, step_days=7, min_bets=100
    )
    gate = deployment_gate(
        metrics,
        folds,
        min_oos_bets=200,
        min_oos_roi=1.00,
        min_positive_fold_ratio=0.60,
        min_recent_fold_ratio=0.50,
    )
    current_rule, current_stats = fit_current_selector(
        preds, lookback_days=35, min_bets=120
    )
    return preds, selected, metrics, folds, gate, current_rule, current_stats

try:
    today = flatten(fetch_today(), False)
except Exception:
    st.error("本日の実レースデータを取得できません。")
    st.stop()

if today.empty:
    st.warning("本日のデータがありません。")
    st.stop()

hist = load_hist()
models = load_models()
if models is None:
    st.error("学習モデルを作成できませんでした。")
    st.stop()

pred_hist, selector_bt, selector_metrics, folds, gate, current_rule, current_stats = validation_assets()
base_stats = summarize(pred_hist)

st.success(
    f'過去実戦データ {hist["race_id"].nunique():,}レース。'
    f' 時系列予測 {len(pred_hist):,}レース。'
)

st.subheader("実戦投入ゲート")
g1, g2, g3, g4 = st.columns(4)
g1.metric("未見検証購入数", f'{gate["oos_bets"]:,}')
g2.metric("未見総回収率", f'{gate["oos_roi"]*100:.1f}%')
g3.metric("未見3連単的中率", f'{gate["oos_hit_rate"]*100:.1f}%')
g4.metric("未見黒字期間率", f'{gate["positive_fold_ratio"]*100:.1f}%')

if gate["passed"]:
    st.success("実戦投入ゲート：合格。未見データで100%超と期間安定性を確認しました。")
else:
    st.error("実戦投入ゲート：不合格。実戦購入候補は0件にします。")
    st.write("不合格理由：", " / ".join(gate["reasons"]))

if folds is not None and not folds.empty:
    st.subheader("未見期間別検証")
    st.dataframe(
        folds[[
            "test_start","test_end","test_bets","test_hit_rate","test_roi","test_profit",
            "train_bets","train_roi","train_shrunk_roi"
        ]],
        use_container_width=True,
        hide_index=True,
    )

dt = pd.to_datetime(today["closed_at"], errors="coerce")
if dt.dt.tz is None:
    dt = dt.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
else:
    dt = dt.dt.tz_convert(JST)

today["closed_at_jst"] = dt
today["minutes_left"] = (dt-now).dt.total_seconds()/60
valid = today[(today["minutes_left"]>=10) & (today["minutes_left"]<=360)].copy()

base_rows = []
base_detail = {}

for rid, g in valid.groupby("race_id"):
    try:
        tri = trifecta_table(models, g)
        sig = selection_signals(tri)
        top = tri.iloc[0]

        score, rule_pass = current_buy_score({
            "確率1位%": float(top["prob"]*100),
            "確率差": float(sig["prob_margin"]),
            "確信度": float(sig["confidence"]),
            "確率1位": str(top["combo"]),
            "R": int(g["race_no"].iloc[0]),
        }, current_rule)

        base_detail[rid] = tri
        base_rows.append({
            "race_id": rid,
            "race_date": g["race_date"].iloc[0],
            "場": g["venue"].iloc[0],
            "R": int(g["race_no"].iloc[0]),
            "締切": pd.Timestamp(g["closed_at_jst"].iloc[0]).strftime("%H:%M"),
            "残り分": float(g["minutes_left"].iloc[0]),
            "確率1位": top["combo"],
            "確率1位%": float(top["prob"]*100),
            "確率差": float(sig["prob_margin"]),
            "確信度": float(sig["confidence"]),
            "利益選別スコア": score,
            "過去条件通過": rule_pass,
        })
    except Exception:
        pass

base = pd.DataFrame(base_rows)
st.subheader("当日一次評価")

if base.empty:
    st.info("現在、条件を満たすレースはありません。")
    st.stop()

pool = base[base["残り分"]<=240].copy()
preferred = pool[pool["過去条件通過"]].sort_values(
    ["利益選別スコア","確率1位%"], ascending=False
)
fallback = pool[~pool["過去条件通過"]].sort_values("残り分")
ids = list(dict.fromkeys(preferred["race_id"].tolist() + fallback["race_id"].tolist()))[:24]
pool = pool[pool["race_id"].isin(ids)].copy()

st.dataframe(
    base.sort_values(
        ["過去条件通過","利益選別スコア","残り分"],
        ascending=[False,False,True]
    ),
    use_container_width=True,
    hide_index=True,
)

def fetch_one(row, retry=False):
    d = pd.Timestamp(row["race_date"]).strftime("%Y%m%d")
    if retry:
        odds, url, diag = fetch_trifecta_odds(d, row["場"], int(row["R"]), timeout=12)
    else:
        odds, url, diag = odds_cached(d, row["場"], int(row["R"]))
    return row["race_id"], odds, url, diag

odds_map = {}

with st.status("実オッズ取得中…", expanded=False) as status:
    rows = [r for _, r in pool.iterrows()]

    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_one, r, False) for r in rows]
        for fut in as_completed(futures):
            try:
                rid, odds, url, diag = fut.result()
                odds_map[rid] = (odds, url, diag)
            except Exception:
                pass

    missing = []
    for _, r in pool.iterrows():
        odds = odds_map.get(r["race_id"], (pd.DataFrame(), None, []))[0]
        if odds is None or len(odds) < 100:
            missing.append(r)

    if missing:
        status.update(label=f"未取得 {len(missing)}レースを再試行中…", state="running")
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(fetch_one, r, True) for r in missing]
            for fut in as_completed(futures):
                try:
                    rid, odds, url, diag = fut.result()
                    old = odds_map.get(rid, (pd.DataFrame(), None, []))
                    if odds is not None and len(odds) > len(old[0]):
                        odds_map[rid] = (odds, url, (old[2] or []) + (diag or []))
                    else:
                        odds_map[rid] = (old[0], old[1], (old[2] or []) + (diag or []))
                except Exception:
                    pass

    status.update(label="実オッズ取得完了", state="complete")

race_rows = []
detail = {}

for _, row in pool.iterrows():
    rid = row["race_id"]
    tri = base_detail[rid].copy()
    odds, url, diag = odds_map.get(
        rid, (pd.DataFrame(columns=["combo","odds"]), None, [])
    )

    if odds is None or odds.empty:
        odds = pd.DataFrame(columns=["combo","odds"])
    if "combo" not in odds:
        odds["combo"] = pd.Series(dtype=object)
    if "odds" not in odds:
        odds["odds"] = pd.Series(dtype=float)

    tri = tri.merge(odds[["combo","odds"]], on="combo", how="left")
    tri["expected_value"] = tri["prob"] * tri["odds"]
    detail[rid] = tri

    ev = tri.dropna(subset=["expected_value"]).copy()
    ev = ev[ev["prob"] >= 0.008]
    best = ev.sort_values(
        ["expected_value","prob"], ascending=False
    ).iloc[0] if len(ev) else tri.iloc[0]

    has_odds = len(odds) >= 100
    ev_val = best.get("expected_value", np.nan)
    historical_pass = bool(row["過去条件通過"])

    reference_signal = bool(
        historical_pass and
        has_odds and
        pd.notna(ev_val) and
        float(ev_val) >= 1.05 and
        float(best["prob"]) >= 0.008
    )

    real_candidate = bool(gate["passed"] and reference_signal)

    race_rows.append({
        "race_id": rid,
        "race_date": row["race_date"],
        "場": row["場"],
        "R": row["R"],
        "締切": row["締切"],
        "残り分": row["残り分"],
        "利益選別スコア": row["利益選別スコア"],
        "過去条件通過": historical_pass,
        "推奨3連単": best["combo"],
        "予測確率%": float(best["prob"]*100),
        "実オッズ": best.get("odds", np.nan),
        "期待値": ev_val,
        "確信度": row["確信度"],
        "確率差": row["確率差"],
        "取得組合せ数": int(len(odds)),
        "参考シグナル": reference_signal,
        "実戦候補": real_candidate,
    })

races = pd.DataFrame(race_rows)

st.subheader("最終評価")
st.dataframe(
    races.sort_values(
        ["実戦候補","参考シグナル","利益選別スコア","期待値"],
        ascending=[False,False,False,False]
    ),
    use_container_width=True,
    hide_index=True,
)

real = races[races["実戦候補"]].sort_values(
    ["利益選別スコア","期待値"], ascending=False
).head(10)

reference = races[races["参考シグナル"]].sort_values(
    ["利益選別スコア","期待値"], ascending=False
).head(10)

st.subheader("実戦購入候補")
if real.empty:
    st.warning(
        "実戦購入候補 0件です。未見データ100%超ゲートを満たさない限り、"
        "当日期待値が高くても実戦候補にはしません。"
    )
else:
    st.dataframe(
        real[["場","R","締切","残り分","推奨3連単","予測確率%","実オッズ","期待値","利益選別スコア"]],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("参考予想（ゲート外なら購入対象外）")
if reference.empty:
    st.info("参考シグナルもありません。")
else:
    st.dataframe(
        reference[["場","R","締切","推奨3連単","予測確率%","実オッズ","期待値","利益選別スコア"]],
        use_container_width=True,
        hide_index=True,
    )

ledger = load_ledger()
ledger = upsert_predictions(
    ledger,
    real,
    now.strftime("%Y-%m-%d %H:%M:%S"),
    stake_yen=100,
)

pending_dates = ledger.loc[ledger["status"]=="pending","race_date"].dropna().tolist()
if pending_dates:
    result_df = fetch_results_for_dates(pending_dates)
    ledger = apply_results(
        ledger,
        result_df,
        settled_at=now.strftime("%Y-%m-%d %H:%M:%S"),
    )
save_ledger(ledger)

st.subheader("元AI vs 未見選別AI")
a1, a2, a3, a4 = st.columns(4)
a1.metric("元AI検証数", f'{base_stats["races"]:,}')
a2.metric("元AI回収率", f'{base_stats["roi"]*100:.1f}%')
a3.metric("未見選別数", f'{selector_metrics.get("races",0):,}')
a4.metric("未見選別回収率", f'{selector_metrics.get("roi",0)*100:.1f}%')

st.subheader("現在の選別条件")
if current_rule:
    st.write({
        "最低AI確率%": round(float(current_rule.get("min_prob") or 0)*100, 3),
        "最低確率差%": round(float(current_rule.get("min_margin") or 0)*100, 3),
        "最低確信度": round(float(current_rule.get("min_conf") or 0), 4),
        "1着艇限定": current_rule.get("first_lane") or "なし",
        "R下限": current_rule.get("race_no_min") or "なし",
        "R上限": current_rule.get("race_no_max") or "なし",
        "内側学習回収率%": round(float(current_stats.get("roi",0))*100, 1),
        "縮小補正回収率%": round(float(current_stats.get("shrunk_roi",0))*100, 1),
    })

st.subheader("実戦成績・自動精算（ゲート合格候補のみ）")
settled = ledger[ledger["status"]=="settled"].copy()
pending = ledger[ledger["status"]=="pending"].copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("記録実戦候補", f"{len(ledger):,}")
m2.metric("結果確定", f"{len(settled):,}")

if len(settled):
    hit = pd.to_numeric(settled["hit"], errors="coerce").fillna(False).astype(bool)
    stake = pd.to_numeric(settled["stake_yen"], errors="coerce").fillna(0).sum()
    ret = pd.to_numeric(settled["return_yen"], errors="coerce").fillna(0).sum()
    m3.metric("実戦的中率", f"{hit.mean()*100:.1f}%")
    m4.metric("実戦回収率", f"{(ret/stake*100 if stake else 0):.1f}%")
else:
    m3.metric("実戦的中率", "—")
    m4.metric("実戦回収率", "—")

st.caption(f"結果待ち：{len(pending)}件")

st.download_button(
    "成績CSVを保存",
    data=ledger.to_csv(index=False).encode("utf-8-sig"),
    file_name="boatrace_ai_v28_live_candidates.csv",
    mime="text/csv",
)

st.subheader("3連単120通りのAI確率・期待値")
if detail:
    rid = st.selectbox("レースを選択", list(detail.keys()))
    t = detail[rid].copy()
    t["prob_pct"] = t["prob"]*100
    st.dataframe(
        t[["combo","prob_pct","odds","expected_value"]]
        .sort_values(["expected_value","prob_pct"], ascending=False, na_position="last"),
        use_container_width=True,
        hide_index=True,
    )

st.subheader("AI再学習")
if st.button("最新結果で再学習する"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("キャッシュをクリアしました。ページ再読込で最新結果を反映します。")

st.caption(
    "v2.8は未見データ回収率100%超を保証するのではなく、確認できた場合だけ実戦候補を許可します。"
    "未見検証が100%以下なら実戦候補は0件です。将来の利益・回収率は保証されません。"
)
