from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import streamlit as st
from src.openapi import fetch_today, flatten

JST=ZoneInfo("Asia/Tokyo")
st.set_page_config(page_title="BOAT RACE AI 無料版", layout="wide")
st.title("BOAT RACE AI 予想ダッシュボード")
st.caption("無料運用版 / 本日の実レース表示を最優先")
st.write("現在時刻：" + datetime.now(JST).strftime("%Y/%m/%d %H:%M:%S"))

try:
    df=flatten(fetch_today())
except Exception:
    st.error("本日の実レースデータを取得できませんでした。")
    st.warning("この状態では購入判断に使用しないでください。")
    st.stop()

if df.empty:
    st.warning("本日のレースデータがありません。")
    st.stop()

dt=pd.to_datetime(df["closed_at"], errors="coerce")
if dt.dt.tz is None:
    dt=dt.dt.tz_localize(JST, nonexistent="shift_forward", ambiguous="NaT")
else:
    dt=dt.dt.tz_convert(JST)
df["closed_at_jst"]=dt
now=pd.Timestamp.now(tz=JST)
df["minutes_left"]=(df["closed_at_jst"]-now).dt.total_seconds()/60

summary=(df.groupby(["venue","race_no","closed_at"],as_index=False)
           .agg(minutes_left=("minutes_left","first")))
summary["status"]=summary["minutes_left"].apply(
    lambda x:"購入候補" if pd.notna(x) and x>=10
    else ("締切間近" if pd.notna(x) and x>0 else "締切済")
)

st.success("本日の実レースデータを取得しました。")
st.subheader("本日の実レース")
st.dataframe(summary.sort_values("minutes_left"), use_container_width=True, hide_index=True)

st.subheader("締切まで10分以上あるレース")
valid=summary[summary["minutes_left"]>=10]
st.dataframe(valid.sort_values("minutes_left"), use_container_width=True, hide_index=True)

st.subheader("出走表・直前情報")
venue=st.selectbox("場", sorted(df["venue"].dropna().unique()))
race=st.selectbox("R", sorted(df[df["venue"]==venue]["race_no"].unique()))
g=df[(df["venue"]==venue)&(df["race_no"]==race)]
show=["lane","racer_name","rank_number","national_win_rate","local_win_rate",
      "motor_number","motor_top_2_percent","boat_number","boat_top_2_percent",
      "exhibition_time","preview_course","preview_start_timing",
      "wind_speed","wave_height_cm","air_temperature","water_temperature"]
st.dataframe(g[show], use_container_width=True, hide_index=True)

st.subheader("AI予想・実オッズ")
st.warning("実オッズとAI予測確率が未接続の間は、購入判断には使用しないでください。")
