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
st.set_page_config(page_title="BOAT RACE AI v2.9", layout="wide")
st.title("BOAT RACE AI 購入判断ダッシュボード")
st.caption("無料版 v2.9：実戦購入判断画面・未見データ100%超ゲート・実オッズ期待値")

now = pd.Timestamp.now(tz=JST)

top1, top2 = st.columns([3, 1])
with top1:
    st.write(f"現在時刻：**{now:%Y/%m/%d %H:%M:%S}**")
with top2:
    if st.button("最新データに更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

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

# 時刻処理
dt = pd.to_datetime(today["closed_at"], errors="coerce")
if dt.dt.tz is None:
    dt = dt.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
else:
    dt = dt.dt.tz_convert(JST)

today["closed_at_jst"] = dt
today["minutes_left"] = (dt - now).dt.total_seconds() / 60

# 締切10分以上を原則。購入判断画面は4時間以内を優先。
valid = today[(today["minutes_left"] >= 10) & (today["minutes_left"] <= 360)].copy()

base_rows = []
base_detail = {}
for rid, g in valid.groupby("race_id"):
    try:
        tri = trifecta_table(models, g)
        sig = selection_signals(tri)
        top = tri.iloc[0]
        score, rule_pass = current_buy_score({
            "確率1位%": float(top["prob"] * 100),
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
            "確率1位%": float(top["prob"] * 100),
            "確率差": float(sig["prob_margin"]),
            "確信度": float(sig["confidence"]),
            "利益選別スコア": float(score),
            "過去条件通過": bool(rule_pass),
        })
    except Exception:
        pass

base = pd.DataFrame(base_rows)
if base.empty:
    st.info("現在、締切まで10分以上ある評価可能レースはありません。")
    st.stop()

# 直近4時間・最大24レースのオッズを取得
pool = base[base["残り分"] <= 240].copy()
preferred = pool[pool["過去条件通過"]].sort_values(
    ["利益選別スコア", "確率1位%"], ascending=False
)
fallback = pool[~pool["過去条件通過"]].sort_values("残り分")
ids = list(dict.fromkeys(preferred["race_id"].tolist() + fallback["race_id"].tolist()))[:24]
pool = pool[pool["race_id"].isin(ids)].copy()

def fetch_one(row, retry=False):
    d = pd.Timestamp(row["race_date"]).strftime("%Y%m%d")
    if retry:
        odds, url, diag = fetch_trifecta_odds(d, row["場"], int(row["R"]), timeout=12)
    else:
        odds, url, diag = odds_cached(d, row["場"], int(row["R"]))
    return row["race_id"], odds, url, diag

odds_map = {}
with st.status("本日の実オッズを取得しています…", expanded=False) as status:
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
        status.update(label=f"オッズ未取得 {len(missing)}レースを再試行しています…", state="running")
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

    status.update(label="本日の実オッズ取得が完了しました", state="complete")

race_rows = []
detail = {}

for _, row in pool.iterrows():
    rid = row["race_id"]
    tri = base_detail[rid].copy()

    odds, url, diag = odds_map.get(
        rid, (pd.DataFrame(columns=["combo", "odds"]), None, [])
    )
    if odds is None or odds.empty:
        odds = pd.DataFrame(columns=["combo", "odds"])
    if "combo" not in odds:
        odds["combo"] = pd.Series(dtype=object)
    if "odds" not in odds:
        odds["odds"] = pd.Series(dtype=float)

    # 120通り詳細にも必ず同じ実オッズを結合
    tri = tri.merge(
        odds[["combo", "odds"]].drop_duplicates("combo"),
        on="combo",
        how="left",
    )
    tri["expected_value"] = tri["prob"] * tri["odds"]
    detail[rid] = tri

    # 低すぎる確率の超高配当は除き、EV最大を最終買い目候補にする
    ev = tri.dropna(subset=["expected_value"]).copy()
    ev = ev[ev["prob"] >= 0.008]
    best = (
        ev.sort_values(["expected_value", "prob"], ascending=False).iloc[0]
        if len(ev)
        else tri.iloc[0]
    )

    has_odds = len(odds) >= 100
    ev_val = best.get("expected_value", np.nan)
    historical_pass = bool(row["過去条件通過"])

    reference_signal = bool(
        historical_pass
        and has_odds
        and pd.notna(ev_val)
        and float(ev_val) >= 1.05
        and float(best["prob"]) >= 0.008
    )

    real_candidate = bool(gate["passed"] and reference_signal)

    # 購入判断理由を明文化
    reasons = []
    if not gate["passed"]:
        reasons.append("未見200件ゲート未合格")
    if not historical_pass:
        reasons.append("過去選別条件外")
    if not has_odds:
        reasons.append("実オッズ不足")
    if pd.isna(ev_val):
        reasons.append("期待値未計算")
    elif float(ev_val) < 1.05:
        reasons.append("期待値1.05未満")
    if float(best["prob"]) < 0.008:
        reasons.append("予測確率0.8%未満")

    judgement = "買い" if real_candidate else "見送り"
    reason_text = "全条件クリア" if real_candidate else " / ".join(reasons)

    race_rows.append({
        "race_id": rid,
        "race_date": row["race_date"],
        "場": row["場"],
        "R": row["R"],
        "締切": row["締切"],
        "残り分": row["残り分"],
        "判断": judgement,
        "推奨3連単": best["combo"],
        "予測確率%": float(best["prob"] * 100),
        "実オッズ": best.get("odds", np.nan),
        "期待値": ev_val,
        "利益選別スコア": row["利益選別スコア"],
        "過去条件通過": historical_pass,
        "確信度": row["確信度"],
        "確率差": row["確率差"],
        "取得組合せ数": int(len(odds)),
        "参考シグナル": reference_signal,
        "実戦候補": real_candidate,
        "判断理由": reason_text,
    })

races = pd.DataFrame(race_rows)

# ==========================================================
# ここから購入判断専用画面
# ==========================================================
st.divider()
st.header("本日の購入判断")

gate1, gate2, gate3, gate4 = st.columns(4)
gate1.metric("未見検証数", f'{gate["oos_bets"]:,} / 200件')
gate2.metric("未見回収率", f'{gate["oos_roi"]*100:.1f}%')
gate3.metric("未見的中率", f'{gate["oos_hit_rate"]*100:.1f}%')
gate4.metric("黒字期間率", f'{gate["positive_fold_ratio"]*100:.1f}%')

if gate["passed"]:
    st.success("実戦投入ゲート：合格。条件を満たすレースのみ「買い」と表示します。")
else:
    st.error(
        "実戦投入ゲート：未合格。現在は全レースを「見送り」とします。"
        "参考予想は確認できます。"
    )
    st.caption("理由：" + " / ".join(gate["reasons"]))

real = races[races["実戦候補"]].sort_values(
    ["残り分", "利益選別スコア", "期待値"], ascending=[True, False, False]
).copy()

reference = races[races["参考シグナル"]].sort_values(
    ["残り分", "利益選別スコア", "期待値"], ascending=[True, False, False]
).copy()

if len(real):
    st.success(f"現在の購入候補：{len(real)}レース")
    for _, r in real.head(10).iterrows():
        with st.container(border=True):
            c1, c2, c3, c4, c5 = st.columns([1.3, 1.1, 1.4, 1.4, 2.0])
            c1.markdown(f"### {r['場']} {int(r['R'])}R")
            c2.metric("締切", r["締切"])
            c3.metric("残り", f"{r['残り分']:.0f}分")
            c4.metric("期待値", f"{r['期待値']:.2f}")
            c5.markdown(f"### 買い目 **{r['推奨3連単']}**")
            st.write(
                f"予測確率 **{r['予測確率%']:.2f}%** ／ "
                f"実オッズ **{r['実オッズ']:.1f}倍** ／ "
                f"利益選別スコア **{r['利益選別スコア']:.1f}**"
            )
            st.success("判定：買い　―　全条件クリア")
else:
    st.warning("現在の実戦購入候補は **0件** です。無理に舟券を購入する必要はありません。")

st.subheader("参考予想")
st.caption("未見ゲート未合格時は、ここに表示されても購入対象外です。")
if reference.empty:
    st.info("現在、参考シグナルもありません。")
else:
    ref_show = reference.head(10).copy()
    st.dataframe(
        ref_show[[
            "場","R","締切","残り分","推奨3連単","予測確率%",
            "実オッズ","期待値","利益選別スコア","判断理由"
        ]],
        use_container_width=True,
        hide_index=True,
    )

st.subheader("全評価レース")
display = races.copy()
display["残り分"] = display["残り分"].round(1)
display["予測確率%"] = display["予測確率%"].round(3)
display["期待値"] = pd.to_numeric(display["期待値"], errors="coerce").round(3)
display["実オッズ"] = pd.to_numeric(display["実オッズ"], errors="coerce").round(1)

st.dataframe(
    display[[
        "判断","場","R","締切","残り分","推奨3連単","予測確率%",
        "実オッズ","期待値","利益選別スコア","取得組合せ数","判断理由"
    ]].sort_values(["判断","残り分"], ascending=[True, True]),
    use_container_width=True,
    hide_index=True,
)

# ==========================================================
# 実戦ログ・結果自動精算
# ==========================================================
ledger = load_ledger()
ledger = upsert_predictions(
    ledger,
    real,
    now.strftime("%Y-%m-%d %H:%M:%S"),
    stake_yen=100,
)

pending_dates = ledger.loc[
    ledger["status"] == "pending", "race_date"
].dropna().tolist()

if pending_dates:
    try:
        result_df = fetch_results_for_dates(pending_dates)
        ledger = apply_results(
            ledger,
            result_df,
            settled_at=now.strftime("%Y-%m-%d %H:%M:%S"),
        )
    except Exception:
        st.warning("結果自動精算で一部取得できませんでした。次回更新時に再試行します。")

save_ledger(ledger)

st.divider()
st.header("実戦成績")

settled = ledger[ledger["status"] == "settled"].copy()
pending = ledger[ledger["status"] == "pending"].copy()

m1, m2, m3, m4 = st.columns(4)
m1.metric("記録候補", f"{len(ledger):,}")
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

if len(settled):
    st.dataframe(
        settled.tail(30)[[
            "race_date","venue","race_no","combo","pred_prob","odds",
            "expected_value","actual_combo","actual_payout","hit",
            "profit_yen","miss_type"
        ]],
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "成績CSVを保存",
    data=ledger.to_csv(index=False).encode("utf-8-sig"),
    file_name="boatrace_ai_v29_performance.csv",
    mime="text/csv",
)

# ==========================================================
# 検証情報は下部へ
# ==========================================================
with st.expander("AI検証・選別条件を確認する"):
    st.subheader("元AI vs 未見選別AI")
    a1, a2, a3, a4 = st.columns(4)
    a1.metric("元AI検証数", f'{base_stats["races"]:,}')
    a2.metric("元AI回収率", f'{base_stats["roi"]*100:.1f}%')
    a3.metric("未見選別数", f'{selector_metrics.get("races",0):,}')
    a4.metric("未見選別回収率", f'{selector_metrics.get("roi",0)*100:.1f}%')

    if folds is not None and not folds.empty:
        st.subheader("未見期間別検証")
        st.dataframe(
            folds[[
                "test_start","test_end","test_bets","test_hit_rate",
                "test_roi","test_profit","train_bets","train_roi",
                "train_shrunk_roi"
            ]],
            use_container_width=True,
            hide_index=True,
        )

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

st.subheader("3連単120通り詳細")
if detail:
    # オッズ取得済みを先頭に並べる
    detail_ids = list(detail.keys())
    default_id = detail_ids[0]
    rid = st.selectbox(
        "レースを選択",
        detail_ids,
        index=detail_ids.index(default_id),
    )
    t = detail[rid].copy()
    t["prob_pct"] = t["prob"] * 100
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
    st.success("キャッシュをクリアしました。ページを再読み込みすると最新結果で再学習します。")

st.caption(
    "この画面は購入判断支援用です。的中・利益を保証しません。"
    "未見検証ゲートが不合格なら、実オッズや期待値が高くても「見送り」とします。"
)
