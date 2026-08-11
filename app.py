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
st.set_page_config(page_title="BOAT RACE AI v2.9.1", layout="wide")
st.title("BOAT RACE AI 購入判断ダッシュボード")
st.caption("無料版 v2.9.1：120通り実オッズ期待値最適化・購入判断画面")

now = pd.Timestamp.now(tz=JST)

# -------------------------
# 操作部
# -------------------------
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

# -------------------------
# 今日データ
# -------------------------
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

dt = pd.to_datetime(today["closed_at"], errors="coerce")
if dt.dt.tz is None:
    dt = dt.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
else:
    dt = dt.dt.tz_convert(JST)

today["closed_at_jst"] = dt
today["minutes_left"] = (dt - now).dt.total_seconds() / 60
valid = today[(today["minutes_left"] >= 10) & (today["minutes_left"] <= 360)].copy()

# -------------------------
# AI 120通り予測
# -------------------------
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

        # race_idをキーにAI120通りを完全保持
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

# 直近4時間から24レース
pool = base[base["残り分"] <= 240].copy()
preferred = pool[pool["過去条件通過"]].sort_values(
    ["利益選別スコア", "AI1位確率%"], ascending=False
)
fallback = pool[~pool["過去条件通過"]].sort_values("残り分")
ids = list(dict.fromkeys(
    preferred["race_id"].tolist() + fallback["race_id"].tolist()
))[:24]
pool = pool[pool["race_id"].isin(ids)].copy()

# -------------------------
# 実オッズ
# -------------------------
def fetch_one(row, retry=False):
    d = pd.Timestamp(row["race_date"]).strftime("%Y%m%d")
    if retry:
        odds, url, diag = fetch_trifecta_odds(
            d, row["場"], int(row["R"]), timeout=12
        )
    else:
        odds, url, diag = odds_cached(
            d, row["場"], int(row["R"])
        )
    return str(row["race_id"]), odds, url, diag

odds_map = {}

with st.status("実オッズを取得しています…", expanded=False) as status:
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
        rid = str(r["race_id"])
        odds = odds_map.get(rid, (pd.DataFrame(), None, []))[0]
        if odds is None or len(odds) < 100:
            missing.append(r)

    if missing:
        status.update(
            label=f"未取得 {len(missing)}レースを再試行しています…",
            state="running"
        )
        with ThreadPoolExecutor(max_workers=4) as ex:
            futures = [ex.submit(fetch_one, r, True) for r in missing]
            for fut in as_completed(futures):
                try:
                    rid, odds, url, diag = fut.result()
                    old = odds_map.get(rid, (pd.DataFrame(), None, []))
                    if odds is not None and len(odds) > len(old[0]):
                        odds_map[rid] = (
                            odds, url, (old[2] or []) + (diag or [])
                        )
                    else:
                        odds_map[rid] = (
                            old[0], old[1], (old[2] or []) + (diag or [])
                        )
                except Exception:
                    pass

    status.update(label="実オッズ取得完了", state="complete")

# -------------------------
# 120通り AI確率 × 実オッズ
# -------------------------
detail_map = {}
race_rows = []

MIN_COMBO_PROB = 0.008   # 0.8%
MIN_EV = 1.05

for _, row in pool.iterrows():
    rid = str(row["race_id"])

    probs = prob_map.get(rid, pd.DataFrame(columns=["combo", "prob"])).copy()
    odds, url, diag = odds_map.get(
        rid, (pd.DataFrame(columns=["combo", "odds"]), None, [])
    )

    if odds is None:
        odds = pd.DataFrame(columns=["combo", "odds"])
    odds = odds.copy()

    if "combo" not in odds.columns:
        odds["combo"] = pd.Series(dtype=object)
    if "odds" not in odds.columns:
        odds["odds"] = pd.Series(dtype=float)

    # comboを明示的に文字列化して結合不一致を防ぐ
    probs["combo"] = probs["combo"].astype(str).str.strip()
    odds["combo"] = odds["combo"].astype(str).str.strip()
    odds["odds"] = pd.to_numeric(odds["odds"], errors="coerce")

    # 重複排除
    odds_clean = (
        odds[["combo", "odds"]]
        .dropna(subset=["combo"])
        .drop_duplicates("combo", keep="last")
    )

    # ここを唯一の120通り詳細データ源とする
    tri = probs.merge(
        odds_clean,
        on="combo",
        how="left",
        validate="one_to_one",
    )
    tri["expected_value"] = tri["prob"] * tri["odds"]
    tri["prob_pct"] = tri["prob"] * 100

    # 全120通りのうちオッズが何件一致したかを保持
    matched_odds = int(tri["odds"].notna().sum())
    detail_map[rid] = tri.copy()

    # 購入買い目選定：
    # 予測確率0.8%以上の中から EV 最大を選ぶ。
    # EV同値なら予測確率が高い方。
    eligible = tri[
        tri["odds"].notna()
        & (tri["prob"] >= MIN_COMBO_PROB)
    ].copy()

    if len(eligible):
        best = eligible.sort_values(
            ["expected_value", "prob"],
            ascending=[False, False]
        ).iloc[0]
    else:
        best = tri.sort_values(
            "prob", ascending=False
        ).iloc[0]

    best_prob = float(best["prob"])
    best_odds = pd.to_numeric(
        pd.Series([best.get("odds")]), errors="coerce"
    ).iloc[0]
    best_ev = pd.to_numeric(
        pd.Series([best.get("expected_value")]), errors="coerce"
    ).iloc[0]

    has_full_odds = matched_odds >= 100
    historical_pass = bool(row["過去条件通過"])

    reference_signal = bool(
        historical_pass
        and has_full_odds
        and pd.notna(best_ev)
        and float(best_ev) >= MIN_EV
        and best_prob >= MIN_COMBO_PROB
    )

    real_candidate = bool(
        gate["passed"] and reference_signal
    )

    reasons = []
    if not gate["passed"]:
        reasons.append("未見200件ゲート未合格")
    if not historical_pass:
        reasons.append("過去選別条件外")
    if not has_full_odds:
        reasons.append(f"実オッズ不足({matched_odds}/120)")
    if pd.isna(best_ev):
        reasons.append("期待値未計算")
    elif float(best_ev) < MIN_EV:
        reasons.append("期待値1.05未満")
    if best_prob < MIN_COMBO_PROB:
        reasons.append("AI確率0.8%未満")

    race_rows.append({
        "race_id": rid,
        "race_date": row["race_date"],
        "場": row["場"],
        "R": int(row["R"]),
        "締切": row["締切"],
        "残り分": float(row["残り分"]),
        "判断": "買い" if real_candidate else "見送り",
        "買い目": str(best["combo"]),
        "AI確率%": best_prob * 100,
        "実オッズ": best_odds,
        "期待値": best_ev,
        "利益選別スコア": float(row["利益選別スコア"]),
        "確信度": float(row["確信度"]),
        "確率差": float(row["確率差"]),
        "取得組合せ数": matched_odds,
        "参考シグナル": reference_signal,
        "実戦候補": real_candidate,
        "判断理由": "全条件クリア" if real_candidate else " / ".join(reasons),
    })

races = pd.DataFrame(race_rows)

# -------------------------
# 購入判断を最上段に大きく表示
# -------------------------
st.divider()
st.header("本日の購入判断")

g1, g2, g3, g4 = st.columns(4)
g1.metric("未見検証数", f'{gate["oos_bets"]:,} / 200')
g2.metric("未見回収率", f'{gate["oos_roi"]*100:.1f}%')
g3.metric("未見的中率", f'{gate["oos_hit_rate"]*100:.1f}%')
g4.metric("黒字期間率", f'{gate["positive_fold_ratio"]*100:.1f}%')

if gate["passed"]:
    st.success("実戦投入ゲート：合格")
else:
    st.error("実戦投入ゲート：未合格 → 現在は全レース『見送り』")
    st.caption("理由：" + " / ".join(gate["reasons"]))

real = races[races["実戦候補"]].sort_values(
    ["残り分", "期待値", "利益選別スコア"],
    ascending=[True, False, False]
)
reference = races[races["参考シグナル"]].sort_values(
    ["残り分", "期待値", "利益選別スコア"],
    ascending=[True, False, False]
)

# 一番上に最大3件の大カード
headline = real if len(real) else reference

if headline.empty:
    st.warning("現在、条件に近い買い目もありません。")
else:
    st.subheader(
        "購入候補" if len(real)
        else "参考候補（ゲート未合格のため購入対象外）"
    )

    for _, r in headline.head(3).iterrows():
        with st.container(border=True):
            left, c2, c3, c4, c5, c6 = st.columns(
                [1.2, 1.1, 1.1, 1.2, 1.2, 2.0]
            )

            with left:
                if r["判断"] == "買い":
                    st.success("## 買い")
                else:
                    st.warning("## 見送り")

            c2.markdown(f"### {r['場']} {int(r['R'])}R")
            c3.metric("締切まで", f"{r['残り分']:.0f}分")
            c4.metric("実オッズ", f"{r['実オッズ']:.1f}倍" if pd.notna(r["実オッズ"]) else "未取得")
            c5.metric("AI確率", f"{r['AI確率%']:.2f}%")
            c6.markdown(f"## 買い目 **{r['買い目']}**")

            m1, m2, m3 = st.columns(3)
            m1.metric("期待値", f"{r['期待値']:.2f}" if pd.notna(r["期待値"]) else "—")
            m2.metric("利益選別スコア", f"{r['利益選別スコア']:.1f}")
            m3.metric("オッズ取得", f"{int(r['取得組合せ数'])}/120")

            if r["判断"] == "買い":
                st.success("判断：買い　／　120通りの中で条件を満たす期待値上位買い目")
            else:
                st.info("判断：見送り　／　" + r["判断理由"])

# 参考候補一覧
st.subheader("参考候補一覧")
if reference.empty:
    st.info("現在、参考候補はありません。")
else:
    show = reference.head(10).copy()
    st.dataframe(
        show[[
            "判断","場","R","締切","残り分","買い目",
            "AI確率%","実オッズ","期待値",
            "取得組合せ数","利益選別スコア","判断理由"
        ]],
        use_container_width=True,
        hide_index=True,
    )

# 全レース
st.subheader("全評価レース")
display = races.copy()
display["残り分"] = display["残り分"].round(1)
display["AI確率%"] = display["AI確率%"].round(3)
display["実オッズ"] = pd.to_numeric(
    display["実オッズ"], errors="coerce"
).round(1)
display["期待値"] = pd.to_numeric(
    display["期待値"], errors="coerce"
).round(3)

st.dataframe(
    display[[
        "判断","場","R","締切","残り分","買い目",
        "AI確率%","実オッズ","期待値",
        "利益選別スコア","取得組合せ数","判断理由"
    ]].sort_values(
        ["判断","残り分"],
        ascending=[True, True]
    ),
    use_container_width=True,
    hide_index=True,
)

# -------------------------
# 実戦ログ
# -------------------------
ledger = load_ledger()

# ledger.pyの既存仕様に合わせる列名へ変換
real_for_log = real.rename(columns={
    "買い目": "推奨3連単",
    "AI確率%": "予測確率%",
}).copy()

ledger = upsert_predictions(
    ledger,
    real_for_log,
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
        st.warning("結果取得に失敗したレースがあります。次回更新時に再試行します。")

save_ledger(ledger)

st.divider()
st.header("実戦成績")

settled = ledger[ledger["status"] == "settled"].copy()
pending = ledger[ledger["status"] == "pending"].copy()

s1, s2, s3, s4 = st.columns(4)
s1.metric("記録候補", f"{len(ledger):,}")
s2.metric("結果確定", f"{len(settled):,}")

if len(settled):
    hit = pd.to_numeric(
        settled["hit"], errors="coerce"
    ).fillna(False).astype(bool)

    stake = pd.to_numeric(
        settled["stake_yen"], errors="coerce"
    ).fillna(0).sum()

    ret = pd.to_numeric(
        settled["return_yen"], errors="coerce"
    ).fillna(0).sum()

    s3.metric("実戦的中率", f"{hit.mean()*100:.1f}%")
    s4.metric(
        "実戦回収率",
        f"{(ret/stake*100 if stake else 0):.1f}%"
    )
else:
    s3.metric("実戦的中率", "—")
    s4.metric("実戦回収率", "—")

st.caption(f"結果待ち：{len(pending)}件")

if len(settled):
    st.dataframe(
        settled.tail(30)[[
            "race_date","venue","race_no",
            "combo","pred_prob","odds","expected_value",
            "actual_combo","actual_payout","hit",
            "profit_yen","miss_type"
        ]],
        use_container_width=True,
        hide_index=True,
    )

st.download_button(
    "成績CSVを保存",
    data=ledger.to_csv(index=False).encode("utf-8-sig"),
    file_name="boatrace_ai_v291_performance.csv",
    mime="text/csv",
)

# -------------------------
# 120通り詳細
# -------------------------
st.divider()
st.header("3連単120通り 詳細")

if detail_map:
    detail_ids = list(detail_map.keys())

    # オッズ一致数の多いレースを初期選択
    counts = {
        rid: int(detail_map[rid]["odds"].notna().sum())
        for rid in detail_ids
    }
    default_id = max(counts, key=counts.get)

    rid = st.selectbox(
        "レースを選択",
        detail_ids,
        index=detail_ids.index(default_id),
        format_func=lambda x: (
            f"{x}  （実オッズ {counts.get(x,0)}/120）"
        ),
    )

    t = detail_map[rid].copy()

    # EV順
    t = t.sort_values(
        ["expected_value", "prob"],
        ascending=[False, False],
        na_position="last",
    )

    # 条件を満たす行に判定ラベル
    t["購入条件"] = np.where(
        t["odds"].notna()
        & (t["prob"] >= MIN_COMBO_PROB)
        & (t["expected_value"] >= MIN_EV),
        "候補",
        "",
    )

    st.caption(
        "各買い目について AI確率 × 実オッズ = 期待値。"
        "AI確率0.8%以上を基本条件とし、その中で期待値の高い買い目を優先します。"
    )

    st.dataframe(
        t[[
            "購入条件","combo","prob_pct",
            "odds","expected_value"
        ]],
        use_container_width=True,
        hide_index=True,
    )

# -------------------------
# 検証情報
# -------------------------
with st.expander("AI検証・現在の選別条件"):
    v1, v2, v3, v4 = st.columns(4)
    v1.metric("元AI検証数", f'{base_stats["races"]:,}')
    v2.metric("元AI回収率", f'{base_stats["roi"]*100:.1f}%')
    v3.metric("未見選別数", f'{selector_metrics.get("races",0):,}')
    v4.metric("未見選別回収率", f'{selector_metrics.get("roi",0)*100:.1f}%')

    if folds is not None and not folds.empty:
        st.dataframe(
            folds[[
                "test_start","test_end","test_bets",
                "test_hit_rate","test_roi","test_profit",
                "train_bets","train_roi","train_shrunk_roi"
            ]],
            use_container_width=True,
            hide_index=True,
        )

    if current_rule:
        st.write({
            "最低AI確率%":
                round(float(current_rule.get("min_prob") or 0)*100, 3),
            "最低確率差%":
                round(float(current_rule.get("min_margin") or 0)*100, 3),
            "最低確信度":
                round(float(current_rule.get("min_conf") or 0), 4),
            "1着艇限定":
                current_rule.get("first_lane") or "なし",
            "R下限":
                current_rule.get("race_no_min") or "なし",
            "R上限":
                current_rule.get("race_no_max") or "なし",
        })

st.subheader("AI再学習")
if st.button("最新結果で再学習する"):
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("キャッシュをクリアしました。ページ再読込で最新結果を反映します。")

st.caption(
    "購入判断支援用です。的中・利益を保証しません。"
    "未見検証ゲート不合格時は、期待値が高くても『見送り』です。"
)
