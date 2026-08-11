from concurrent.futures import ThreadPoolExecutor, as_completed
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st

from src.data import fetch_today, flatten, historical_dataset
from src.model import fit_model, race_probs, trifecta_table
from src.odds import fetch_trifecta_odds

JST = ZoneInfo("Asia/Tokyo")

st.set_page_config(page_title="BOAT RACE AI", layout="wide")
st.title("BOAT RACE AI 予想ダッシュボード")
st.caption("無料版 v2.2：高速化・実データ学習・3連単確率・期待値・成績検証")

now = pd.Timestamp.now(tz=JST)
st.write(f"現在時刻：{now:%Y/%m/%d %H:%M:%S}")

@st.cache_data(ttl=21600, show_spinner="過去実戦データを読み込んでいます…")
def load_hist():
    return historical_dataset(60)

@st.cache_resource(ttl=21600, show_spinner="AIを再学習しています…")
def load_model():
    return fit_model(load_hist())

@st.cache_data(ttl=120, show_spinner=False)
def odds_cached(d, v, r):
    return fetch_trifecta_odds(d, v, r, timeout=6)

try:
    today = flatten(fetch_today(), False)
except Exception:
    st.error("本日の実レースデータを取得できません。")
    st.stop()

if today.empty:
    st.warning("本日のデータがありません。")
    st.stop()

hist = load_hist()
model = load_model()
st.success(
    f'過去実戦データ {hist["race_id"].nunique() if not hist.empty else 0:,}レースを学習対象として読み込みました。'
)

if model is None:
    st.warning("学習データ不足のため簡易スコアを使用中です。購入判断には使用しないでください。")

dt = pd.to_datetime(today["closed_at"], errors="coerce")
if dt.dt.tz is None:
    dt = dt.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
else:
    dt = dt.dt.tz_convert(JST)

today["closed_at_jst"] = dt
today["minutes_left"] = (dt - now).dt.total_seconds() / 60

# 原則10分以上、かつ本日中の現実的な購入対象
valid = today[(today["minutes_left"] >= 10) & (today["minutes_left"] <= 360)].copy()

# まずオッズ無しで全候補を高速評価
base_rows = []
base_detail = {}
for rid, g in valid.groupby("race_id"):
    try:
        p = race_probs(model, g)
        tri = trifecta_table(g, p)
        base_detail[rid] = tri
        top = tri.iloc[0]
        fav = float(max(p))
        entropy = float(-(p * np.log(np.maximum(p, 1e-12))).sum())
        confidence = fav / max(entropy, 1e-9)

        base_rows.append({
            "race_id": rid,
            "場": g["venue"].iloc[0],
            "R": int(g["race_no"].iloc[0]),
            "締切": pd.Timestamp(g["closed_at_jst"].iloc[0]).strftime("%H:%M"),
            "残り分": float(g["minutes_left"].iloc[0]),
            "確率1位": top["combo"],
            "確率1位%": float(top["prob"] * 100),
            "確信度": confidence,
            "race_date": g["race_date"].iloc[0],
        })
    except Exception:
        pass

base = pd.DataFrame(base_rows)

st.subheader("AI一次評価（締切まで10分以上）")
if base.empty:
    st.info("現在、条件を満たすレースはありません。")
    st.stop()

st.dataframe(
    base.sort_values(["残り分", "確信度"], ascending=[True, False]),
    use_container_width=True,
    hide_index=True
)

# 実オッズは「直近4時間以内」かつAI評価上位を優先して最大24レース
pool = base[base["残り分"] <= 240].copy()
if len(pool) > 24:
    # 堅め候補12 + 波乱候補12
    safe_ids = pool.sort_values("確信度", ascending=False).head(12)["race_id"].tolist()
    upset_ids = pool.sort_values("確信度", ascending=True).head(12)["race_id"].tolist()
    selected_ids = list(dict.fromkeys(safe_ids + upset_ids))[:24]
    pool = pool[pool["race_id"].isin(selected_ids)].copy()

st.info(f"実オッズは購入時間を優先し、最大 {len(pool)} レースを並列取得します。")

def fetch_one(row):
    d = pd.Timestamp(row["race_date"]).strftime("%Y%m%d")
    odds, url = odds_cached(d, row["場"], int(row["R"]))
    return row["race_id"], odds, url

odds_map = {}
progress = st.progress(0, text="実オッズを取得しています…")

rows_for_fetch = [r for _, r in pool.iterrows()]
done = 0

with ThreadPoolExecutor(max_workers=8) as ex:
    futures = [ex.submit(fetch_one, r) for r in rows_for_fetch]
    for fut in as_completed(futures):
        try:
            rid, odds, url = fut.result()
            odds_map[rid] = (odds, url)
        except Exception:
            pass
        done += 1
        progress.progress(done / max(1, len(futures)), text=f"実オッズ取得中… {done}/{len(futures)}")

progress.empty()

race_rows = []
detail = {}

for _, row in pool.iterrows():
    rid = row["race_id"]
    tri = base_detail[rid].copy()
    odds, url = odds_map.get(rid, (pd.DataFrame(columns=["combo","odds"]), None))

    if odds is None or odds.empty:
        odds = pd.DataFrame(columns=["combo","odds"])
    else:
        odds = odds.copy()
        if "combo" not in odds.columns:
            odds["combo"] = pd.Series(dtype="object")
        if "odds" not in odds.columns:
            odds["odds"] = pd.Series(dtype="float64")
        odds = odds[["combo","odds"]]

    tri = tri.merge(odds, on="combo", how="left")
    tri["expected_value"] = tri["prob"] * tri["odds"]
    detail[rid] = tri

    top = tri.iloc[0]
    evbest = tri.dropna(subset=["expected_value"]).sort_values("expected_value", ascending=False).head(1)
    best = evbest.iloc[0] if len(evbest) else top

    race_rows.append({
        "race_id": rid,
        "場": row["場"],
        "R": row["R"],
        "締切": row["締切"],
        "残り分": row["残り分"],
        "確信度": row["確信度"],
        "確率1位": top["combo"],
        "確率1位%": float(top["prob"] * 100),
        "推奨3連単": best["combo"],
        "予測確率%": float(best["prob"] * 100),
        "実オッズ": best.get("odds", np.nan),
        "期待値": best.get("expected_value", np.nan),
        "オッズ取得": bool(len(odds) > 0),
    })

races = pd.DataFrame(race_rows)

st.subheader("最終評価対象")
st.dataframe(races, use_container_width=True, hide_index=True)

if not races["オッズ取得"].any():
    st.warning(
        "現在、公式3連単オッズを自動取得できていません。"
        "AI予測確率は表示できますが、期待値は計算できないため正式な購入推奨は出しません。"
    )

safe = races.copy()
safe["安全スコア"] = safe["確率1位%"] * 0.7 + safe["確信度"] * 30
safe = safe[(safe["期待値"].isna()) | (safe["期待値"] >= 0.85)]
safe = safe.sort_values(["安全スコア","期待値"], ascending=False).head(10)

holes = races.dropna(subset=["期待値"]).copy()
holes = holes[holes["予測確率%"] >= 1.5]
holes = holes.sort_values("期待値", ascending=False).head(10)

c1, c2 = st.columns(2)
with c1:
    st.subheader("堅め候補 TOP10")
    st.dataframe(
        safe[["場","R","締切","残り分","推奨3連単","予測確率%","実オッズ","期待値","確信度"]],
        use_container_width=True,
        hide_index=True
    )

with c2:
    st.subheader("穴候補 TOP10（期待値順）")
    if holes.empty:
        st.info("実オッズ取得済みの穴候補がありません。")
    else:
        st.dataframe(
            holes[["場","R","締切","残り分","推奨3連単","予測確率%","実オッズ","期待値","確信度"]],
            use_container_width=True,
            hide_index=True
        )

st.subheader("3連単120通りのAI確率・期待値")
if detail:
    rid = st.selectbox("レースを選択", list(detail.keys()))
    t = detail[rid].copy()
    t["prob_pct"] = t["prob"] * 100
    st.dataframe(
        t[["combo","prob_pct","odds","expected_value"]]
        .sort_values("expected_value", ascending=False, na_position="last"),
        use_container_width=True,
        hide_index=True
    )

st.subheader("AIの学習・外れ分析")
if hist.empty or model is None:
    st.info("過去結果が十分に取得できると、ここに検証結果を表示します。")
else:
    recent_ids = hist["race_id"].drop_duplicates().tail(100).tolist()
    checks = []
    for rid, g in hist[hist["race_id"].isin(recent_ids)].groupby("race_id"):
        try:
            p = race_probs(model, g)
            tri = trifecta_table(g, p)
            pred = tri.iloc[0]["combo"]
            actual = g["trifecta_result"].dropna()
            if len(actual):
                checks.append({
                    "race_id": rid,
                    "予想": pred,
                    "結果": actual.iloc[0],
                    "的中": pred == actual.iloc[0],
                    "最大1着確率%": float(max(p) * 100)
                })
        except Exception:
            pass

    chk = pd.DataFrame(checks)
    if not chk.empty:
        st.metric(
            "直近100レース 3連単1点一致率（参考）",
            f'{float(chk["的中"].mean()*100):.1f}%'
        )
        miss = chk[~chk["的中"]]
        st.write(
            f"外れ分析対象：{len(miss)}レース。"
            "結果は次回再学習時の教師データとして自動的に取り込まれます。"
        )
        st.dataframe(miss.tail(20), use_container_width=True, hide_index=True)

st.caption(
    "予測は統計モデルによる推定で、的中・利益を保証しません。"
    "オッズ未取得時は期待値を推測しません。"
)
