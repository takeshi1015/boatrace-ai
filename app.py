from concurrent.futures import ThreadPoolExecutor,as_completed
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st

from src.data import fetch_today,flatten,historical_dataset,fetch_results_for_dates
from src.model import fit_models,trifecta_table,selection_signals
from src.backtest import generate_walk_forward_predictions,nested_selector_backtest,deployment_gate,fit_current_selector,current_buy_score,summarize
from src.calibration import fit_probability_calibrator,calibrate_distribution
from src.tickets import probability_tables_from_trifecta,merge_odds,choose_ticket_candidates,priority_candidate,TICKET_NAMES
from src.odds import fetch_all_ticket_odds
from src.ledger import load_ledger,save_ledger,upsert_predictions,apply_results

JST=ZoneInfo('Asia/Tokyo'); APP_VERSION='v2.11'
DIRECT_REFRESH_MINUTES=30; MIN_EV=1.05
st.set_page_config(page_title='BOAT RACE AI v2.11',layout='wide')
st.title('BOAT RACE AI 購入判断ダッシュボード')
st.caption('v2.11：日次自己学習・確率校正・3連単/3連複/2連単/2連複 自動比較')
now=pd.Timestamp.now(tz=JST)

c1,c2=st.columns([4,1])
c1.write(f'現在時刻：**{now:%Y/%m/%d %H:%M:%S}**')
if c2.button('最新データに更新',use_container_width=True):
    st.cache_data.clear();st.cache_resource.clear();st.rerun()

@st.cache_data(ttl=21600,show_spinner='過去120日データを再構築しています…')
def load_hist():return historical_dataset(120)
@st.cache_resource(ttl=21600,show_spinner='最新結果を含めAIを再学習しています…')
def load_models():return fit_models(load_hist())
@st.cache_data(ttl=21600,show_spinner='未見データ検証と確率校正を更新しています…')
def learning_assets():
    preds=generate_walk_forward_predictions(load_hist(),min_train_days=35,test_days=7,max_test_days=56)
    selected,metrics,folds=nested_selector_backtest(preds,lookback_days=28,step_days=7,min_bets=100)
    gate=deployment_gate(metrics,folds,min_oos_bets=200,min_oos_roi=1.00,min_positive_fold_ratio=0.60,min_recent_fold_ratio=0.50)
    rule,rstats=fit_current_selector(preds,lookback_days=35,min_bets=120)
    calibrator,cstats=fit_probability_calibrator(preds)
    return preds,selected,metrics,folds,gate,rule,rstats,calibrator,cstats

try: today=flatten(fetch_today(),False)
except Exception:
    st.error('本日のレースデータを取得できません。');st.stop()
if today.empty:st.warning('本日のレースデータがありません。');st.stop()

hist=load_hist();models=load_models()
if models is None:st.error('AIモデルを作成できませんでした。');st.stop()
preds,selector_bt,selector_metrics,folds,gate,current_rule,current_stats,calibrator,cal_stats=learning_assets()
base_stats=summarize(preds)
latest_date=str(hist['race_date'].dropna().max()) if not hist.empty else '—'

st.header('自己学習状況')
a,b,c,d,e=st.columns(5)
a.metric('学習レース',f"{hist['race_id'].nunique():,}")
b.metric('最新結果日',latest_date)
c.metric('校正サンプル',f"{cal_stats.get('samples',0):,}")
d.metric('未見検証',f"{gate['oos_bets']:,}/200")
e.metric('未見回収率',f"{gate['oos_roi']*100:.1f}%")
if cal_stats.get('brier_raw') is not None:
    delta=cal_stats['brier_raw']-cal_stats['brier_cal']
    st.caption(f"確率校正 Brier: 校正前 {cal_stats['brier_raw']:.4f} → 校正後 {cal_stats['brier_cal']:.4f}（改善 {delta:+.4f}）。毎回、公開済み結果を再取得して過去を再演算します。")
else:st.caption('確率校正はサンプル蓄積中です。')
if gate['passed']:st.success('実戦投入ゲート：合格')
else:st.warning('実戦投入ゲート：未合格。試験候補は表示しますが、購入判断は慎重にしてください。 理由：'+' / '.join(gate['reasons']))

# Times
dt=pd.to_datetime(today['closed_at'],errors='coerce')
if dt.dt.tz is None:dt=dt.dt.tz_localize(JST,nonexistent='shift_forward',ambiguous='NaT')
else:dt=dt.dt.tz_convert(JST)
today['closed_at_jst']=dt;today['minutes_left']=(dt-now).dt.total_seconds()/60
valid=today[(today['minutes_left']>=10)&(today['minutes_left']<=240)].copy()

base_rows=[];race_groups={};prob_tables={}
for rid,g in valid.groupby('race_id'):
    try:
        raw=trifecta_table(models,g)
        tri=calibrate_distribution(raw,calibrator)
        sig=selection_signals(tri);top=tri.iloc[0]
        score,rule_pass=current_buy_score({'確率1位%':float(top['prob']*100),'確率差':float(sig['prob_margin']),'確信度':float(sig['confidence']),'確率1位':str(top['combo']),'R':int(g['race_no'].iloc[0])},current_rule)
        rid=str(rid);race_groups[rid]=g.copy();prob_tables[rid]=probability_tables_from_trifecta(tri)
        base_rows.append({'race_id':rid,'race_date':g['race_date'].iloc[0],'場':g['venue'].iloc[0],'R':int(g['race_no'].iloc[0]),'締切':pd.Timestamp(g['closed_at_jst'].iloc[0]).strftime('%H:%M'),'残り分':float(g['minutes_left'].iloc[0]),'AI1位':top['combo'],'校正AI確率%':float(top['prob']*100),'利益選別スコア':float(score),'過去条件通過':bool(rule_pass),'確信度':float(sig['confidence'])})
    except Exception:pass
base=pd.DataFrame(base_rows)
if base.empty:st.info('現在、購入時間を確保できる評価対象レースはありません。');st.stop()
# API負荷を抑え、AI条件通過＋締切近い順で最大12レース
preferred=base[base['過去条件通過']].sort_values(['利益選別スコア','残り分'],ascending=[False,True])
fallback=base[~base['過去条件通過']].sort_values('残り分')
ids=list(dict.fromkeys(preferred['race_id'].tolist()+fallback['race_id'].tolist()))[:12]
pool=base[base['race_id'].isin(ids)].copy()

def fetch_one(row):
    d=pd.Timestamp(row['race_date']).strftime('%Y%m%d');timeout=13 if row['残り分']<=30 else 9
    odds,diags=fetch_all_ticket_odds(d,row['場'],int(row['R']),timeout=timeout)
    return row['race_id'],odds,diags,('直前再取得' if row['残り分']<=30 else '通常取得')

odds_map={}
with st.status('4券種の実オッズを取得しています…',expanded=False) as status:
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures=[ex.submit(fetch_one,r) for _,r in pool.iterrows()]
        for fut in as_completed(futures):
            try:
                rid,odds,diags,mode=fut.result();odds_map[str(rid)]=(odds,diags,mode)
            except Exception:pass
    status.update(label='4券種の実オッズ取得完了',state='complete')

race_rows=[];details={}
for _,r in pool.iterrows():
    rid=str(r['race_id']);odds,diags,mode=odds_map.get(rid,({}, {}, '取得失敗'))
    merged={}
    for code,pdf in prob_tables[rid].items():merged[code]=merge_odds(pdf,odds.get(code,pd.DataFrame(columns=['combo','odds'])))
    details[rid]=merged
    safe,value=choose_ticket_candidates(merged);priority,priority_reason=priority_candidate(safe,value)
    if priority is None:continue
    complete=sum(int(merged[c]['odds'].notna().sum()>0) for c in merged)
    reference=bool(r['過去条件通過'] and pd.notna(priority.get('expected_value')) and float(priority['expected_value'])>=MIN_EV)
    final_buy=bool(gate['passed'] and reference)
    def fields(x,prefix):
        if x is None:return {prefix+'券種':'—',prefix+'買い目':'—',prefix+'確率%':np.nan,prefix+'オッズ':np.nan,prefix+'期待値':np.nan}
        return {prefix+'券種':x['ticket'],prefix+'買い目':x['combo'],prefix+'確率%':float(x['prob']*100),prefix+'オッズ':x.get('odds'),prefix+'期待値':x.get('expected_value')}
    row={'race_id':rid,'race_date':r['race_date'],'場':r['場'],'R':r['R'],'締切':r['締切'],'残り分':r['残り分'],'判断':'買い' if final_buy else '見送り','優先券種':priority['ticket'],'優先買い目':priority['combo'],'優先確率%':float(priority['prob']*100),'優先オッズ':priority.get('odds'),'優先期待値':priority.get('expected_value'),'優先理由':priority_reason,'利益選別スコア':r['利益選別スコア'],'過去条件通過':r['過去条件通過'],'オッズ更新':mode,'取得券種数':complete,'参考候補':reference,'実戦候補':final_buy,'確信度':r['確信度']}
    row.update(fields(safe,'高確率'));row.update(fields(value,'高期待値'));race_rows.append(row)
races=pd.DataFrame(race_rows)

st.divider();st.header('本日の券種・買い目判断')
if races.empty:st.warning('実オッズまで揃った候補がありません。');st.stop()
headline=races[races['参考候補']].sort_values(['残り分','優先期待値'],ascending=[True,False])
if headline.empty:st.warning('現在、期待値1.05以上かつ過去選別条件を満たす試験候補はありません。')
else:
    for _,r in headline.head(5).iterrows():
        with st.container(border=True):
            a,b,c,d,e=st.columns([1.0,1.3,1.1,1.2,1.4])
            (a.success if r['判断']=='買い' else a.warning)(f"## {r['判断']}")
            b.markdown(f"### {r['場']} {int(r['R'])}R")
            c.metric('締切まで',f"{r['残り分']:.0f}分")
            d.metric('優先券種',r['優先券種'])
            e.markdown(f"### {r['優先買い目']}")
            st.markdown('#### 高確率寄り')
            x1,x2,x3,x4=st.columns(4);x1.metric('券種',r['高確率券種']);x2.metric('買い目',r['高確率買い目']);x3.metric('AI確率',f"{r['高確率確率%']:.2f}%");x4.metric('期待値',f"{r['高確率期待値']:.2f}" if pd.notna(r['高確率期待値']) else '—')
            st.markdown('#### 高期待値寄り')
            y1,y2,y3,y4=st.columns(4);y1.metric('券種',r['高期待値券種']);y2.metric('買い目',r['高期待値買い目']);y3.metric('AI確率',f"{r['高期待値確率%']:.2f}%");y4.metric('期待値',f"{r['高期待値期待値']:.2f}" if pd.notna(r['高期待値期待値']) else '—')
            st.info(f"優先：{r['優先券種']} {r['優先買い目']} ／ {r['優先理由']} ／ 実オッズ {r['優先オッズ']:.1f}倍 ／ EV {r['優先期待値']:.2f}")
            if r['オッズ更新']=='直前再取得':st.caption('締切30分以内のため直前オッズで再判定')

st.subheader('全評価レース')
st.dataframe(races[['判断','場','R','締切','残り分','優先券種','優先買い目','優先確率%','優先オッズ','優先期待値','高確率券種','高確率買い目','高期待値券種','高期待値買い目','オッズ更新','取得券種数']].sort_values('残り分'),use_container_width=True,hide_index=True)

# Live log: only 3連単 can use existing ledger settlement format without adding result parsing for other ticket types.
# v2.11 therefore logs all displayed recommendations to a separate generic CSV, while legacy 3t ledger remains intact.
log_path='data/v211_recommendations.csv'
try: oldlog=pd.read_csv(log_path)
except Exception: oldlog=pd.DataFrame()
newlog=races.copy();newlog['recorded_at']=now.strftime('%Y-%m-%d %H:%M:%S');newlog['strategy_version']=APP_VERSION
if not oldlog.empty:newlog=pd.concat([oldlog,newlog],ignore_index=True).drop_duplicates(['race_id','優先券種','優先買い目'],keep='first')
try:newlog.to_csv(log_path,index=False,encoding='utf-8-sig')
except Exception:pass

st.divider();st.header('券種別詳細')
rid=st.selectbox('レースを選択',list(details.keys()),format_func=lambda x:f"{x} / {base.loc[base['race_id']==x,'場'].iloc[0]} {int(base.loc[base['race_id']==x,'R'].iloc[0])}R")
for code in ('3t','3f','2t','2f'):
    df=details[rid][code].copy().sort_values(['expected_value','prob'],ascending=False,na_position='last')
    with st.expander(f"{TICKET_NAMES[code]}（{int(df['odds'].notna().sum())}/{len(df)} オッズ取得）",expanded=(code!='3t')):
        show=df.head(30).copy();show['AI確率']=show['prob_pct'].map(lambda x:f'{x:.3f}%');show['実オッズ']=show['odds'].map(lambda x:f'{x:.1f}倍' if pd.notna(x) else '—');show['期待値']=show['expected_value'].map(lambda x:f'{x:.3f}' if pd.notna(x) else '—')
        st.dataframe(show[['combo','AI確率','実オッズ','期待値']],use_container_width=True,hide_index=True)

with st.expander('学習・校正・未見検証の詳細'):
    st.write({'学習最新日':latest_date,'学習レース数':int(hist['race_id'].nunique()),'未見検証数':gate['oos_bets'],'未見回収率%':round(gate['oos_roi']*100,1),'未見的中率%':round(gate['oos_hit_rate']*100,1),'ゲート':gate['passed'],'校正サンプル':cal_stats.get('samples',0),'Brier校正前':cal_stats.get('brier_raw'),'Brier校正後':cal_stats.get('brier_cal')})
    if folds is not None and not folds.empty:st.dataframe(folds,use_container_width=True,hide_index=True)

st.subheader('AI再学習')
if st.button('最新結果で再学習する'):
    st.cache_data.clear();st.cache_resource.clear();st.success('キャッシュを消去しました。再読込すると公開済み最新結果まで含めて再学習します。')
st.caption('v2.11は公開済み過去結果を毎回再取得して学習履歴を再構築するため、Streamlitの一時ファイルが消えても学習本体は失われません。実オッズ未取得項目は推測しません。予測・利益を保証するものではありません。')
