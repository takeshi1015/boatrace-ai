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
APP_VERSION = "v2.10"
V291_CUTOFF = pd.Timestamp("2026-08-11 11:09:00")  # v2.9.1画面確認時刻を基準
MIN_COMBO_PROB = 0.008
MIN_EV = 1.05
DIRECT_REFRESH_MINUTES = 30

st.set_page_config(page_title="BOAT RACE AI v2.10", layout="wide")
st.title("BOAT RACE AI 購入判断ダッシュボード")
st.caption("v2.10：200件自動ゲート・2案提示・30分以内直前オッズ再判定・世代別成績")

now = pd.Timestamp.now(tz=JST)

h1, h2 = st.columns([4, 1])
with h1:
    st.write(f"現在時刻：**{now:%Y/%m/%d %H:%M:%S}**")
with h2:
    if st.button("最新データに更新", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

@st.cache_data(ttl=21600, show_spinner="過去実戦データを読み込んでいます…")
def load_hist():
    return historical_dataset(90)

@st.cache_resource(ttl=21600, show_spinner="AIモデルを学習しています…")
def load_models():
    return fit_models(load_hist())

@st.cache_data(ttl=120, show_spinner=False)
def odds_cached(d, v, r):
    return fetch_trifecta_odds(d, v, r, timeout=7)

@st.cache_data(ttl=21600, show_spinner="未見データで購入条件を検証しています…")
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
    st.error("本日のレースデータを取得できません。")
    st.stop()

if today.empty:
    st.warning("本日のレースデータがありません。")
    st.stop()

hist = load_hist()
models = load_models()
if models is None:
    st.error("AIモデルを作成できませんでした。")
    st.stop()

pred_hist, selector_bt, selector_metrics, folds, gate, current_rule, current_stats = validation_assets()
base_stats = summarize(pred_hist)

# ① 200件到達後は毎回自動でゲート判定
# deployment_gate自体が oos_bets>=200 を条件に含むため、到達後は次回表示時に自動判定される。
# 200件未満は残件数を明示する。
remaining_to_gate = max(0, 200 - int(gate["oos_bets"]))

dt = pd.to_datetime(today["closed_at"], errors="coerce")
if dt.dt.tz is None:
    dt = dt.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
else:
    dt = dt.dt.tz_convert(JST)

today["closed_at_jst"] = dt
today["minutes_left"] = (dt - now).dt.total_seconds() / 60
valid = today[(today["minutes_left"] >= 10) & (today["minutes_left"] <= 360)].copy()

# ---------------------------
# AI probability tables
# ---------------------------
base_rows = []
prob_map = {}

for rid, g in valid.groupby("race_id"):
    try:
        tri = trifecta_table(models, g).copy()
        sig = selection_signals(tri)
        top = tri.iloc[0]

        score, rule_pass = current_buy_score({
            "確率1位%": float(top["prob"] * 100),
            "確率差": float(sig["prob_margin"]),
            "確信度": float(sig["confidence"]),
            "確率1位": str(top["combo"]),
            "R": int(g["race_no"].iloc[0]),
        }, current_rule)

        prob_map[str(rid)] = tri[["combo", "prob"]].copy()

        base_rows.append({
            "race_id": str(rid),
            "race_date": g["race_date"].iloc[0],
            "場": g["venue"].iloc[0],
            "R": int(g["race_no"].iloc[0]),
            "締切": pd.Timestamp(g["closed_at_jst"].iloc[0]).strftime("%H:%M"),
            "残り分": float(g["minutes_left"].iloc[0]),
            "AI1位": str(top["combo"]),
            "AI1位確率%": float(top["prob"] * 100),
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

pool = base[base["残り分"] <= 240].copy()
preferred = pool[pool["過去条件通過"]].sort_values(
    ["利益選別スコア", "AI1位確率%"], ascending=False
)
fallback = pool[~pool["過去条件通過"]].sort_values("残り分")
ids = list(dict.fromkeys(
    preferred["race_id"].tolist() + fallback["race_id"].tolist()
))[:24]
pool = pool[pool["race_id"].isin(ids)].copy()

# ---------------------------
# ④ odds fetch with direct refresh inside 30 minutes
# ---------------------------
def fetch_one(row):
    d = pd.Timestamp(row["race_date"]).strftime("%Y%m%d")
    left = float(row["残り分"])

    # 締切30分以内はキャッシュを使わず直取り
    if left <= DIRECT_REFRESH_MINUTES:
        odds, url, diag = fetch_trifecta_odds(
            d, row["場"], int(row["R"]), timeout=12
        )
        refresh_mode = "直前再取得"
    else:
        odds, url, diag = odds_cached(
            d, row["場"], int(row["R"])
        )
        refresh_mode = "通常取得"

    return str(row["race_id"]), odds, url, diag, refresh_mode

def retry_one(row):
    d = pd.Timestamp(row["race_date"]).strftime("%Y%m%d")
    odds, url, diag = fetch_trifecta_odds(
        d, row["場"], int(row["R"]), timeout=15
    )
    return str(row["race_id"]), odds, url, diag, "再試行"

odds_map = {}

with st.status("実オッズを取得しています…", expanded=False) as status:
    rows = [r for _, r in pool.iterrows()]
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(fetch_one, r) for r in rows]
        for fut in as_completed(futures):
            try:
                rid, odds, url, diag, mode = fut.result()
                odds_map[rid] = (odds, url, diag, mode)
            except Exception:
                pass

    missing = []
    for _, r in pool.iterrows():
        rid = str(r["race_id"])
        odds = odds_map.get(rid, (pd.DataFrame(), None, [], ""))[0]
        if odds is None or len(odds) < 100:
            missing.append(r)

    if missing:
        status.update(label=f"未取得 {len(missing)}レースを再試行しています…", state="running")
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(retry_one, r) for r in missing]
            for fut in as_completed(futures):
                try:
                    rid, odds, url, diag, mode = fut.result()
                    old = odds_map.get(rid, (pd.DataFrame(), None, [], ""))
                    if odds is not None and len(odds) > len(old[0]):
                        odds_map[rid] = (odds, url, (old[2] or []) + (diag or []), mode)
                    else:
                        odds_map[rid] = (old[0], old[1], (old[2] or []) + (diag or []), old[3])
                except Exception:
                    pass

    status.update(label="実オッズ取得完了", state="complete")

# ---------------------------
# ③ two picks: high-probability and high-EV
# ---------------------------
detail_map = {}
race_rows = []

for _, row in pool.iterrows():
    rid = str(row["race_id"])
    probs = prob_map.get(rid, pd.DataFrame(columns=["combo", "prob"])).copy()
    odds, url, diag, refresh_mode = odds_map.get(
        rid, (pd.DataFrame(columns=["combo", "odds"]), None, [], "")
    )

    if odds is None:
        odds = pd.DataFrame(columns=["combo", "odds"])
    odds = odds.copy()

    if "combo" not in odds.columns:
        odds["combo"] = pd.Series(dtype=object)
    if "odds" not in odds.columns:
        odds["odds"] = pd.Series(dtype=float)

    probs["combo"] = probs["combo"].astype(str).str.strip()
    odds["combo"] = odds["combo"].astype(str).str.strip()
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")

    odds_clean = (
        odds[["combo", "odds"]]
        .dropna(subset=["combo"])
        .drop_duplicates("combo", keep="last")
    )

    tri = probs.merge(
        odds_clean, on="combo", how="left", validate="one_to_one"
    )
    tri["expected_value"] = tri["prob"] * tri["odds"]
    tri["prob_pct"] = tri["prob"] * 100

    matched_odds = int(tri["odds"].notna().sum())
    detail_map[rid] = tri.copy()

    # 高確率寄り:
    # EVが極端に悪くない(>=0.90)中で予測確率最大
    safe_pool = tri[
        tri["odds"].notna()
        & (tri["prob"] >= MIN_COMBO_PROB)
        & (tri["expected_value"] >= 0.90)
    ].copy()
    if safe_pool.empty:
        safe_pool = tri[
            tri["odds"].notna()
            & (tri["prob"] >= MIN_COMBO_PROB)
        ].copy()
    if len(safe_pool):
        safe = safe_pool.sort_values(
            ["prob", "expected_value"], ascending=False
        ).iloc[0]
    else:
        safe = tri.sort_values("prob", ascending=False).iloc[0]

    # 高期待値寄り:
    # AI確率0.8%以上の中で期待値最大
    ev_pool = tri[
        tri["odds"].notna()
        & (tri["prob"] >= MIN_COMBO_PROB)
    ].copy()
    if len(ev_pool):
        value = ev_pool.sort_values(
            ["expected_value", "prob"], ascending=False
        ).iloc[0]
    else:
        value = safe

    has_full_odds = matched_odds >= 100
    historical_pass = bool(row["過去条件通過"])

    value_ev = pd.to_numeric(pd.Series([value.get("expected_value")]), errors="coerce").iloc[0]
    reference_signal = bool(
        historical_pass
        and has_full_odds
        and pd.notna(value_ev)
        and float(value_ev) >= MIN_EV
        and float(value["prob"]) >= MIN_COMBO_PROB
    )
    real_candidate = bool(gate["passed"] and reference_signal)

    reasons = []
    if not gate["passed"]:
        if remaining_to_gate > 0:
            reasons.append(f"未見200件まであと{remaining_to_gate}件")
        else:
            reasons.append("未見ゲート不合格")
    if not historical_pass:
        reasons.append("過去選別条件外")
    if not has_full_odds:
        reasons.append(f"実オッズ不足({matched_odds}/120)")
    if pd.isna(value_ev):
        reasons.append("期待値未計算")
    elif float(value_ev) < MIN_EV:
        reasons.append("期待値1.05未満")

    race_rows.append({
        "race_id": rid,
        "race_date": row["race_date"],
        "場": row["場"],
        "R": int(row["R"]),
        "締切": row["締切"],
        "残り分": float(row["残り分"]),
        "判断": "買い" if real_candidate else "見送り",
        "高確率買い目": str(safe["combo"]),
        "高確率AI確率%": float(safe["prob"]) * 100,
        "高確率オッズ": safe.get("odds", np.nan),
        "高確率期待値": safe.get("expected_value", np.nan),
        "高期待値買い目": str(value["combo"]),
        "高期待値AI確率%": float(value["prob"]) * 100,
        "高期待値オッズ": value.get("odds", np.nan),
        "高期待値期待値": value_ev,
        "利益選別スコア": float(row["利益選別スコア"]),
        "取得組合せ数": matched_odds,
        "オッズ更新": refresh_mode,
        "参考シグナル": reference_signal,
        "実戦候補": real_candidate,
        "判断理由": "全条件クリア" if real_candidate else " / ".join(reasons),
    })

races = pd.DataFrame(race_rows)

# ---------------------------
# Main purchase screen
# ---------------------------
st.divider()
st.header("本日の購入判断")

g1, g2, g3, g4, g5 = st.columns(5)
g1.metric("未見検証数", f'{gate["oos_bets"]:,}/200')
g2.metric("200件まで", f"{remaining_to_gate}件")
g3.metric("未見回収率", f'{gate["oos_roi"]*100:.1f}%')
g4.metric("未見的中率", f'{gate["oos_hit_rate"]*100:.1f}%')
g5.metric("黒字期間率", f'{gate["positive_fold_ratio"]*100:.1f}%')

if gate["passed"]:
    st.success("実戦投入ゲート：合格。200件到達後の未見成績条件を満たしています。")
else:
    st.error("実戦投入ゲート：未合格 → 現在は全レース『見送り』")
    st.caption("理由：" + " / ".join(gate["reasons"]))

real = races[races["実戦候補"]].sort_values(
    ["残り分", "高期待値期待値"], ascending=[True, False]
)
reference = races[races["参考シグナル"]].sort_values(
    ["残り分", "高期待値期待値"], ascending=[True, False]
)
headline = real if len(real) else reference

if headline.empty:
    st.warning("現在、条件に近い参考候補もありません。")
else:
    st.subheader("購入候補" if len(real) else "参考候補（ゲート外のため購入対象外）")

    for _, r in headline.head(3).iterrows():
        with st.container(border=True):
            a,b,c,d,e = st.columns([1.1,1.1,1.1,1.2,1.5])
            if r["判断"] == "買い":
                a.success("## 買い")
            else:
                a.warning("## 見送り")
            b.markdown(f"### {r['場']} {r['R']}R")
            c.metric("締切まで", f"{r['残り分']:.0f}分")
            d.metric("オッズ更新", r["オッズ更新"])
            e.metric("取得", f"{r['取得組合せ数']}/120")

            st.markdown("#### 高確率寄り")
            x1,x2,x3,x4 = st.columns(4)
            x1.metric("買い目", r["高確率買い目"])
            x2.metric("AI確率", f"{r['高確率AI確率%']:.2f}%")
            x3.metric("実オッズ", f"{r['高確率オッズ']:.1f}倍" if pd.notna(r["高確率オッズ"]) else "—")
            x4.metric("期待値", f"{r['高確率期待値']:.2f}" if pd.notna(r["高確率期待値"]) else "—")

            st.markdown("#### 高期待値寄り")
            y1,y2,y3,y4 = st.columns(4)
            y1.metric("買い目", r["高期待値買い目"])
            y2.metric("AI確率", f"{r['高期待値AI確率%']:.2f}%")
            y3.metric("実オッズ", f"{r['高期待値オッズ']:.1f}倍" if pd.notna(r["高期待値オッズ"]) else "—")
            y4.metric("期待値", f"{r['高期待値期待値']:.2f}" if pd.notna(r["高期待値期待値"]) else "—")

            if r["残り分"] <= DIRECT_REFRESH_MINUTES:
                st.info("締切30分以内：直前オッズを再取得して最終判定済み")
            st.caption("判断理由：" + r["判断理由"])

# All races table
st.subheader("全評価レース")
st.dataframe(
    races[[
        "判断","場","R","締切","残り分",
        "高確率買い目","高確率AI確率%","高確率オッズ","高確率期待値",
        "高期待値買い目","高期待値AI確率%","高期待値オッズ","高期待値期待値",
        "オッズ更新","取得組合せ数","判断理由"
    ]].sort_values(["判断","残り分"], ascending=[True,True]),
    use_container_width=True,
    hide_index=True
)

# ---------------------------
# Logging: v2.10 separately tagged
# ---------------------------
ledger = load_ledger()

# Log only real candidates, using high-EV pick as executed strategy candidate
real_for_log = real.copy()
real_for_log["推奨3連単"] = real_for_log["高期待値買い目"]
real_for_log["予測確率%"] = real_for_log["高期待値AI確率%"]
real_for_log["実オッズ"] = real_for_log["高期待値オッズ"]
real_for_log["期待値"] = real_for_log["高期待値期待値"]
# ledger expects confidence; keep selector score as available proxy if missing
real_for_log["確信度"] = real_for_log["利益選別スコア"] / 100.0

ledger = upsert_predictions(
    ledger,
    real_for_log,
    now.strftime("%Y-%m-%d %H:%M:%S"),
    stake_yen=100,
    strategy_version=APP_VERSION,
)

pending_dates = ledger.loc[
    ledger["status"]=="pending", "race_date"
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
        st.warning("結果取得に失敗したレースがあります。次回更新時に再試行します。")

save_ledger(ledger)

# ---------------------------
# ② separate v2.9.1+ performance
# ---------------------------
st.divider()
st.header("実戦成績")

settled_all = ledger[ledger["status"]=="settled"].copy()

# Since old files had no version tag, use recorded_at cutoff to isolate v2.9.1+
recorded_dt = pd.to_datetime(ledger["recorded_at"], errors="coerce")
v291_plus = ledger[recorded_dt >= V291_CUTOFF].copy()
settled_v291 = v291_plus[v291_plus["status"]=="settled"].copy()
pending_v291 = v291_plus[v291_plus["status"]=="pending"].copy()

def performance_metrics(df):
    if df.empty:
        return 0,0,0.0,0.0,0.0
    settled = df[df["status"]=="settled"].copy()
    if settled.empty:
        return len(df),0,0.0,0.0,0.0
    hits = pd.to_numeric(settled["hit"], errors="coerce").fillna(False).astype(bool)
    stake = pd.to_numeric(settled["stake_yen"], errors="coerce").fillna(0).sum()
    ret = pd.to_numeric(settled["return_yen"], errors="coerce").fillna(0).sum()
    profit = pd.to_numeric(settled["profit_yen"], errors="coerce").fillna(0).sum()
    return len(df),len(settled),float(hits.mean()),float(ret/stake if stake else 0),float(profit)

all_n,all_set,all_hit,all_roi,all_profit = performance_metrics(ledger)
new_n,new_set,new_hit,new_roi,new_profit = performance_metrics(v291_plus)

st.subheader("v2.9.1以降")
m1,m2,m3,m4,m5 = st.columns(5)
m1.metric("記録候補", f"{new_n}")
m2.metric("結果確定", f"{new_set}")
m3.metric("的中率", f"{new_hit*100:.1f}%")
m4.metric("回収率", f"{new_roi*100:.1f}%")
m5.metric("収支", f"{new_profit:+,.0f}円")

with st.expander("旧バージョンを含む累計成績"):
    a1,a2,a3,a4,a5 = st.columns(5)
    a1.metric("記録候補", f"{all_n}")
    a2.metric("結果確定", f"{all_set}")
    a3.metric("的中率", f"{all_hit*100:.1f}%")
    a4.metric("回収率", f"{all_roi*100:.1f}%")
    a5.metric("収支", f"{all_profit:+,.0f}円")

if not settled_v291.empty:
    st.dataframe(
        settled_v291.tail(30)[[
            "race_date","venue","race_no","combo","pred_prob","odds",
            "expected_value","actual_combo","actual_payout","hit","profit_yen",
            "miss_type","strategy_version"
        ]],
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "旧ログにはstrategy_version列が無かったため、v2.9.1以降集計は "
    "2026/08/11 11:09以降のrecorded_atを基準に分離しています。"
)

st.download_button(
    "成績CSVを保存",
    data=ledger.to_csv(index=False).encode("utf-8-sig"),
    file_name="boatrace_ai_v210_performance.csv",
    mime="text/csv",
)

# ---------------------------
# 120 details
# ---------------------------
st.divider()
st.header("3連単120通り 詳細")

if detail_map:
    detail_ids = list(detail_map.keys())
    counts = {
        rid:int(detail_map[rid]["odds"].notna().sum())
        for rid in detail_ids
    }
    default_id = max(counts, key=counts.get)
    rid = st.selectbox(
        "レースを選択",
        detail_ids,
        index=detail_ids.index(default_id),
        format_func=lambda x:f"{x}（実オッズ {counts.get(x,0)}/120）"
    )

    t = detail_map[rid].copy()
    t["区分"] = ""
    if len(t):
        # mark high probability and high EV
        valid_t = t[t["odds"].notna() & (t["prob"]>=MIN_COMBO_PROB)].copy()
        if len(valid_t):
            safe_idx = valid_t.sort_values(
                ["prob","expected_value"], ascending=False
            ).index[0]
            value_idx = valid_t.sort_values(
                ["expected_value","prob"], ascending=False
            ).index[0]
            t.loc[safe_idx,"区分"] = "高確率"
            t.loc[value_idx,"区分"] = (
                "高確率・高期待値"
                if value_idx == safe_idx else "高期待値"
            )

    st.dataframe(
        t.sort_values(
            ["expected_value","prob"], ascending=False, na_position="last"
        )[["区分","combo","prob_pct","odds","expected_value"]],
        use_container_width=True,
        hide_index=True
    )

with st.expander("AI検証・現在の選別条件"):
    if folds is not None and not folds.empty:
        st.dataframe(folds, use_container_width=True, hide_index=True)
    st.write({
        "未見検証件数": gate["oos_bets"],
        "200件まで残り": remaining_to_gate,
        "未見回収率%": round(gate["oos_roi"]*100,1),
        "黒字期間率%": round(gate["positive_fold_ratio"]*100,1),
        "ゲート合格": gate["passed"],
    })

st.subheader("AI再学習")
if st.button("最新結果で再学習する"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("キャッシュをクリアしました。再読込で最新結果を反映します。")

st.caption(
    "購入判断支援用です。的中・利益を保証しません。"
    "締切30分以内は直前オッズを再取得します。"
)
