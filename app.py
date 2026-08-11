from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st

from src.data import fetch_today,flatten,historical_dataset
from src.model import fit_model,race_probs,trifecta_table
from src.odds import fetch_trifecta_odds

JST=ZoneInfo('Asia/Tokyo')
st.set_page_config(page_title='BOAT RACE AI',layout='wide')
st.title('BOAT RACE AI 予想ダッシュボード')
st.caption('無料版 v2.1：実データ学習・3連単確率・期待値・成績検証')
now=pd.Timestamp.now(tz=JST)
st.write(f'現在時刻：{now:%Y/%m/%d %H:%M:%S}')

@st.cache_data(ttl=21600,show_spinner='過去実戦データを読み込んでいます…')
def load_hist():
    return historical_dataset(60)

@st.cache_resource(ttl=21600,show_spinner='AIを再学習しています…')
def load_model():
    return fit_model(load_hist())

@st.cache_data(ttl=120,show_spinner=False)
def odds_cached(d,v,r):
    return fetch_trifecta_odds(d,v,r)

try:
    today=flatten(fetch_today(),False)
except Exception:
    st.error('本日の実レースデータを取得できません。')
    st.stop()

if today.empty:
    st.warning('本日のデータがありません。')
    st.stop()

hist=load_hist()
model=load_model()
st.success(f'過去実戦データ {hist["race_id"].nunique() if not hist.empty else 0:,}レースを学習対象として読み込みました。')
if model is None:
    st.warning('学習データ不足のため簡易スコアを使用中です。購入判断には使用しないでください。')

dt=pd.to_datetime(today['closed_at'],errors='coerce')
if dt.dt.tz is None:
    dt=dt.dt.tz_localize(JST,nonexistent='shift_forward',ambiguous='NaT')
else:
    dt=dt.dt.tz_convert(JST)

today['closed_at_jst']=dt
today['minutes_left']=(dt-now).dt.total_seconds()/60
valid=today[today['minutes_left']>=10].copy()

race_rows=[]
detail={}
race_errors=[]

for rid,g in valid.groupby('race_id'):
    try:
        p=race_probs(model,g)
        tri=trifecta_table(g,p)

        d=(pd.Timestamp(g['race_date'].iloc[0]).strftime('%Y%m%d')
           if pd.notna(g['race_date'].iloc[0])
           else now.strftime('%Y%m%d'))

        odds,url=odds_cached(d,g['venue'].iloc[0],int(g['race_no'].iloc[0]))

        # 修正点：空のオッズでも combo / odds 列を必ず保証
        if odds is None or odds.empty:
            odds=pd.DataFrame(columns=['combo','odds'])
        else:
            if 'combo' not in odds.columns:
                odds['combo']=pd.Series(dtype='object')
            if 'odds' not in odds.columns:
                odds['odds']=pd.Series(dtype='float64')
            odds=odds[['combo','odds']].copy()

        tri=tri.merge(odds,on='combo',how='left')
        tri['expected_value']=tri['prob']*tri['odds']
        detail[rid]=tri

        top=tri.iloc[0]
        evbest=tri.dropna(subset=['expected_value']).sort_values('expected_value',ascending=False).head(1)
        best=evbest.iloc[0] if len(evbest) else top

        race_rows.append({
            'race_id':rid,
            '場':g['venue'].iloc[0],
            'R':int(g['race_no'].iloc[0]),
            '締切':pd.Timestamp(g['closed_at_jst'].iloc[0]).strftime('%H:%M'),
            '残り分':float(g['minutes_left'].iloc[0]),
            '確率1位':top['combo'],
            '確率1位%':float(top['prob']*100),
            '推奨3連単':best['combo'],
            '予測確率%':float(best['prob']*100),
            '実オッズ':best.get('odds',np.nan),
            '期待値':best.get('expected_value',np.nan),
            'オッズ取得':bool(len(odds)>0),
        })
    except Exception as e:
        race_errors.append({
            'race_id':rid,
            '場':g['venue'].iloc[0] if 'venue' in g else '',
            'R':int(g['race_no'].iloc[0]) if 'race_no' in g else '',
            'エラー':type(e).__name__
        })

races=pd.DataFrame(race_rows)

st.subheader('購入可能レース（締切10分以上）')
if races.empty:
    st.info('現在、条件を満たし、予測処理まで完了したレースはありません。')
else:
    st.dataframe(races,use_container_width=True,hide_index=True)

if race_errors:
    st.warning(f'{len(race_errors)}レースで個別処理エラーがありました。ほかのレースの表示は継続します。')

if not races.empty and not races['オッズ取得'].any():
    st.warning('現在、公式3連単オッズを自動取得できていません。AI確率は表示できますが、期待値は計算できないため購入推奨は出しません。')

if not races.empty:
    safe=races.copy()
    safe['安全スコア']=safe['予測確率%']
    safe=safe[
        (safe['期待値'].isna()) | (safe['期待値']>=0.85)
    ].sort_values(['安全スコア','期待値'],ascending=False).head(10)

    holes=races.dropna(subset=['期待値']).copy()
    holes=holes[
        holes['予測確率%']>=1.5
    ].sort_values('期待値',ascending=False).head(10)

    c1,c2=st.columns(2)
    with c1:
        st.subheader('堅め候補 TOP10')
        st.dataframe(
            safe[['場','R','締切','残り分','推奨3連単','予測確率%','実オッズ','期待値']],
            use_container_width=True,hide_index=True
        )
    with c2:
        st.subheader('穴候補 TOP10（期待値順）')
        if not holes.empty:
            st.dataframe(
                holes[['場','R','締切','残り分','推奨3連単','予測確率%','実オッズ','期待値']],
                use_container_width=True,hide_index=True
            )
        else:
            st.info('実オッズが取れた候補がありません。')

st.subheader('3連単120通りのAI確率・期待値')
if detail:
    rid=st.selectbox('レースを選択',list(detail.keys()))
    t=detail[rid].copy()
    t['prob_pct']=t['prob']*100
    st.dataframe(
        t[['combo','prob_pct','odds','expected_value']]
        .sort_values('expected_value',ascending=False,na_position='last'),
        use_container_width=True,hide_index=True
    )

st.subheader('AIの学習・外れ分析')
if hist.empty or model is None:
    st.info('過去結果が十分に取得できると、ここに検証結果を表示します。')
else:
    recent_ids=hist['race_id'].drop_duplicates().tail(100).tolist()
    checks=[]
    for rid,g in hist[hist['race_id'].isin(recent_ids)].groupby('race_id'):
        try:
            p=race_probs(model,g)
            tri=trifecta_table(g,p)
            pred=tri.iloc[0]['combo']
            actual=g['trifecta_result'].dropna()
            if len(actual):
                checks.append({
                    'race_id':rid,
                    '予想':pred,
                    '結果':actual.iloc[0],
                    '的中':pred==actual.iloc[0],
                    '最大1着確率%':float(max(p)*100)
                })
        except Exception:
            pass

    chk=pd.DataFrame(checks)
    if not chk.empty:
        st.metric(
            '直近100レース 3連単1点一致率（参考）',
            f'{float(chk["的中"].mean()*100):.1f}%'
        )
        miss=chk[~chk['的中']]
        st.write(f'外れ分析対象：{len(miss)}レース。結果は次回再学習時の教師データとして自動的に取り込まれます。')
        st.dataframe(miss.tail(20),use_container_width=True,hide_index=True)
        st.caption('これは簡易監査です。厳密な性能評価には時系列ウォークフォワード検証が必要です。')

st.caption('予測は統計モデルによる推定で、的中・利益を保証しません。オッズ未取得時は期待値を推測しません。')
