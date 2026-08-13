from concurrent.futures import ThreadPoolExecutor,as_completed
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import streamlit as st

from src.data import fetch_today,flatten,historical_dataset,fetch_results_for_dates
from src.model import fit_models,trifecta_table,selection_signals,race_data_quality
from src.backtest import generate_walk_forward_predictions,nested_selector_backtest,deployment_gate,fit_current_selector,current_buy_score,summarize,temporal_profit_summary
from src.calibration import fit_probability_calibrator,calibrate_distribution,fit_ticket_calibrators,calibrate_ticket_table,ticket_reliability_gate
from src.tickets import probability_tables_from_trifecta,merge_odds,choose_ticket_candidates,priority_candidate,TICKET_NAMES
from src.context_reliability import fit_context_reliability,current_race_context,context_factor,apply_context_factor,apply_market_overlay
from src.odds import fetch_all_ticket_odds
from src.ledger import load_ledger,save_ledger,upsert_predictions,apply_results
from src.persistence import save_pickle,load_pickle,remove_pickle,export_learning_snapshot,import_learning_snapshot,cache_file_exists

JST=ZoneInfo('Asia/Tokyo'); APP_VERSION='v2.13.3b'
DIRECT_REFRESH_MINUTES=30; MIN_EV=1.10
SNAPSHOT_VERSION='v2132-assets-1'; MODEL_SNAPSHOT_VERSION='v2132-model-1'
st.set_page_config(page_title='BOAT RACE AI v2.13.3b',layout='wide')
st.title('BOAT RACE AI 購入判断ダッシュボード')
st.caption('v2.13.3b：①保存・復元＋②新規結果の増分モデル更新')
now=pd.Timestamp.now(tz=JST)

c1,c2,c3,c4=st.columns([2.7,1,1.25,1.45])
c1.write(f'現在時刻：**{now:%Y/%m/%d %H:%M:%S}**')
if c2.button('当日データ・オッズ更新',use_container_width=True):
    st.rerun()
incremental_clicked=c3.button('新規結果だけモデル更新',use_container_width=True,type='secondary')
retrain_clicked=c4.button('完全再学習＋未見検証',use_container_width=True,type='secondary')


with st.expander('① 学習スナップショットの保存・復元', expanded=False):
    st.caption('Streamlit Community Cloudのローカル保存は再起動後まで残る保証がないため、学習済み状態をZIPとしてPCに保存します。次回は同じZIPをアップロードして復元できます。v2.13.3bではモデルの増分学習状態も一緒に保存します。')
    snap_bytes = export_learning_snapshot()
    a1, a2 = st.columns(2)
    if snap_bytes:
        a1.download_button(
            '学習スナップショットをPCへ保存',
            data=snap_bytes,
            file_name=f'boatrace_learning_snapshot_{now:%Y%m%d_%H%M}.zip',
            mime='application/zip',
            use_container_width=True,
        )
        a1.success('保存可能な学習結果があります')
    else:
        a1.info('まだ保存可能な学習結果がありません')

    uploaded_snapshot = a2.file_uploader(
        '以前保存した学習スナップショットを選択',
        type=['zip'],
        key='learning_snapshot_upload',
    )
    if uploaded_snapshot is not None:
        if a2.button('この学習結果を復元', use_container_width=True):
            result = import_learning_snapshot(uploaded_snapshot.getvalue())
            if result.get('restored'):
                st.success('復元しました: ' + ', '.join(result['restored']))
                st.cache_resource.clear()
                st.rerun()
            else:
                st.error('復元できませんでした: ' + ' / '.join(result.get('errors', ['不明なエラー'])))

@st.cache_data(ttl=21600,show_spinner='公開済み結果を確認しています…')
def load_hist():
    return historical_dataset(220)

def _recent_hist(h,days=120):
    if h is None or h.empty:return h
    x=h.copy(); d=pd.to_datetime(x.get('race_date'),errors='coerce')
    if not d.notna().any():return x
    cutoff=d.max()-pd.Timedelta(days=int(days)-1)
    return x[d>=cutoff].copy()


def _training_state_from_hist(h, added_races=0, mode='model_update'):
    ids=[]
    if h is not None and not h.empty and 'race_id' in h:
        ids=sorted(set(h['race_id'].dropna().astype(str)))
    latest='—'
    if h is not None and not h.empty and 'race_date' in h:
        d=pd.to_datetime(h['race_date'],errors='coerce')
        if d.notna().any():
            latest=str(d.max().date())
    return {
        'seen_race_ids': ids,
        'learned_races': len(ids),
        'last_result_date': latest,
        'added_races': int(added_races),
        'updated_at': str(pd.Timestamp.now(tz=JST)),
        'mode': mode,
    }

def _get_training_state():
    snap=load_pickle('training_state',max_age_hours=None,required_version=None)
    if snap is None or not isinstance(snap.get('value'),dict):
        return {}
    return snap['value']

def _new_result_ids(h,state):
    if h is None or h.empty or 'race_id' not in h:
        return set()
    current=set(h['race_id'].dropna().astype(str))
    seen=set((state or {}).get('seen_race_ids') or [])
    return current-seen

@st.cache_resource(ttl=21600,show_spinner='保存済み予測モデルを読み込んでいます…')
def load_models_cached():
    snap=load_pickle('models',max_age_hours=168,required_version=MODEL_SNAPSHOT_VERSION)
    if snap is not None:
        return snap['value'], '保存済み学習モデル'
    # Do not train tens of thousands of races during normal startup. Until the
    # explicit retraining is run once, trifecta_table uses its conservative
    # fallback scores and the deployment gate remains locked.
    return None,'未学習（簡易予測・購入不可）'

def _empty_learning_assets():
    gate={'passed':False,'reasons':['厳密な未見検証は未実施（再学習ボタンで実行）'],
          'oos_bets':0,'oos_roi':0.0,'oos_hit_rate':0.0,
          'positive_fold_ratio':0.0,'recent_positive_ratio':0.0,'valid_folds':0}
    ticket_stats={c:{'samples':0,'positives':0,'safe_factor':0.70,'mean_pred':None,'hit_rate':None} for c in ('3t','3f','2t','2f')}
    ticket_models={c:None for c in ('3t','3f','2t','2f')}
    # Ticket reliability is intentionally false. The UI may still show reference
    # odds/EV, but final BUY is impossible until strict OOS retraining is saved.
    ticket_gate={c:{'passed':False,'reasons':['未見校正未実施'],'samples':0,'safe_factor':0.70} for c in ('3t','3f','2t','2f')}
    return (pd.DataFrame(),pd.DataFrame(),{},pd.DataFrame(),gate,None,{},None,
            {'samples':0,'positives':0,'brier_raw':None,'brier_cal':None},
            ticket_models,ticket_stats,ticket_gate,{})

def _build_learning_assets(full=True):
    h=load_hist()
    # 112 future-side days provide far more than 200 candidate races while
    # requiring only four 28-day model fits. This is strict walk-forward: every
    # evaluated race is predicted by a model trained only on earlier dates.
    preds=generate_walk_forward_predictions(h,min_train_days=35,test_days=28,max_test_days=112)
    selected,metrics,folds=nested_selector_backtest(preds,lookback_days=35,step_days=7,min_bets=80,min_selector_days=35)
    gate=deployment_gate(metrics,folds,min_oos_bets=200,min_oos_roi=1.00,min_positive_fold_ratio=0.60,min_recent_fold_ratio=0.50)
    rule,rstats=fit_current_selector(preds,lookback_days=180,min_bets=100)
    calibrator,cstats=fit_probability_calibrator(preds)
    ticket_calibrators,ticket_stats=fit_ticket_calibrators(preds)
    ticket_gate=ticket_reliability_gate(ticket_stats,min_samples=200)
    context_stats=fit_context_reliability(preds,min_samples=25)
    return (preds,selected,metrics,folds,gate,rule,rstats,calibrator,cstats,
            ticket_calibrators,ticket_stats,ticket_gate,context_stats)

@st.cache_resource(ttl=21600,show_spinner='保存済み学習結果を読み込んでいます…')
def load_learning_assets_cached():
    snap=load_pickle('learning_assets',max_age_hours=168,required_version=SNAPSHOT_VERSION)
    if snap is not None:
        return snap['value'], 'full', '保存済み厳密検証'
    return _empty_learning_assets(),'none','未実施（即時起動）'


if incremental_clicked:
    with st.status('新しく確定した結果だけを確認しています…',expanded=True) as rs:
        st.cache_data.clear()
        h=load_hist()
        old_state=_get_training_state()
        new_ids=_new_result_ids(h,old_state)
        old_model=load_pickle('models',max_age_hours=None,required_version=MODEL_SNAPSHOT_VERSION)

        if old_model is None:
            rs.write('保存済みモデルがないため、直近180日だけで初回モデルを作成します')
            train=_recent_hist(h,180)
            m=fit_models(train)
            if m is None:
                rs.update(label='モデル作成に失敗しました',state='error')
                st.stop()
            added=int(h['race_id'].nunique()) if 'race_id' in h else 0
            save_pickle('models',m,{'version':MODEL_SNAPSHOT_VERSION,'scope':'recent180','update_mode':'bootstrap'})
            state=_training_state_from_hist(h,added_races=added,mode='bootstrap_recent180')
            save_pickle('training_state',state,{'version':'training-state-v1'})
            rs.update(label=f'初回モデル作成完了：{state["learned_races"]:,}レースを記録',state='complete')
        elif not new_ids:
            state=old_state or _training_state_from_hist(h,0,'no_change')
            state=dict(state); state['added_races']=0; state['updated_at']=str(pd.Timestamp.now(tz=JST)); state['mode']='no_change'
            save_pickle('training_state',state,{'version':'training-state-v1'})
            rs.update(label='新しい確定レースはありません。モデル再計算を省略しました',state='complete')
        else:
            rs.write(f'新規確定レースを {len(new_ids):,}件検出')
            rs.write('直近180日を使って、最近の傾向を重視したモデルだけ再適合します')
            # The underlying estimator is not online/partial_fit capable.
            # Operational incremental learning therefore detects only new results,
            # then performs one bounded rolling-window refit instead of replaying
            # all 34k+ historical races or strict OOS validation.
            train=_recent_hist(h,180)
            m=fit_models(train)
            if m is None:
                rs.update(label='増分モデル更新に失敗しました',state='error')
                st.stop()
            save_pickle('models',m,{'version':MODEL_SNAPSHOT_VERSION,'scope':'recent180','update_mode':'incremental_refit','added_races':len(new_ids)})
            state=_training_state_from_hist(h,added_races=len(new_ids),mode='incremental_recent180')
            save_pickle('training_state',state,{'version':'training-state-v1'})
            rs.update(label=f'増分モデル更新完了：新規 {len(new_ids):,}レース',state='complete')
    st.cache_resource.clear()
    st.rerun()

if retrain_clicked:
    with st.status('長期再学習を実行しています。通常更新ではこの処理は走りません…',expanded=True) as rs:
        rs.write('① 220日分の公開結果を確認')
        st.cache_data.clear()
        h=load_hist()
        rs.write('② 予測モデルを再学習')
        m=fit_models(_recent_hist(h,180))
        if m is None:
            rs.update(label='再学習に失敗しました',state='error');st.stop()
        save_pickle('models',m,{'version':MODEL_SNAPSHOT_VERSION,'scope':'full220'})
        rs.write('③ 厳密なWalk-forward未見検証・確率校正を実行')
        assets=_build_learning_assets(full=True)
        save_pickle('learning_assets',assets,{'version':SNAPSHOT_VERSION,'mode':'full'})
        state=_training_state_from_hist(h,added_races=int(h['race_id'].nunique()) if 'race_id' in h else 0,mode='full_retrain')
        save_pickle('training_state',state,{'version':'training-state-v1'})
        remove_pickle('bootstrap_assets')
        rs.write('④ 学習結果と学習状態を保存')
        rs.update(label='再学習完了。上部の「学習スナップショットをPCへ保存」で必ずバックアップしてください',state='complete')
    st.cache_resource.clear()
    st.rerun()

try: today=flatten(fetch_today(),False)
except Exception:
    st.error('本日のレースデータを取得できません。');st.stop()
if today.empty:st.warning('本日のレースデータがありません。');st.stop()

hist=load_hist();models,model_source=load_models_cached()
_assets,learning_mode,learning_source=load_learning_assets_cached()
preds,selector_bt,selector_metrics,folds,gate,current_rule,current_stats,calibrator,cal_stats,ticket_calibrators,ticket_stats,ticket_gate,context_stats=_assets
base_stats=summarize(preds)
temporal_stats=temporal_profit_summary(selector_bt)
if gate.get('oos_bets',0)==0:
    st.warning('未見利益検証は未実施です。「最新結果で再学習」を1回実行すると、厳密な未見検証を保存します。0.0%を実績値としては扱いません。')
latest_date=str(hist['race_date'].dropna().max()) if not hist.empty else '—'
st.caption(f'起動モード：予測モデル={model_source} / 学習検証={learning_source}。保存済み学習が無い場合も即時起動し、実戦ゲートは必ずロックされます。')

training_state=_get_training_state()
if training_state:
    t1,t2,t3,t4=st.columns(4)
    t1.metric('保存済みモデル学習件数',f"{int(training_state.get('learned_races',0)):,}")
    t2.metric('今回追加学習',f"{int(training_state.get('added_races',0)):,}")
    t3.metric('モデル最新結果日',training_state.get('last_result_date','—'))
    upd=str(training_state.get('updated_at','—'))
    t4.metric('モデル更新',upd[:16].replace('T',' ') if upd!='—' else '—')

if learning_mode=='none':
    st.info('高速起動モード：保存済み学習がまだありません。参考予想は表示できますが、実戦の「買い」は出ません。時間のある時に上部の「最新結果で再学習」を1回実行してください。')

st.header('本日の状態')
a,b,c,d=st.columns(4)
a.metric('未見検証',f"{gate['oos_bets']:,}/200" if gate.get('oos_bets',0)>0 else '未実施')
b.metric('未見回収率',f"{gate['oos_roi']*100:.1f}%" if gate.get('oos_bets',0)>0 else '未実施')
c.metric('最新結果日',latest_date)
d.metric('実戦ゲート','合格' if gate['passed'] else '未合格')
if gate['passed']:
    st.success('実戦投入ゲート：合格')
else:
    st.warning('実戦投入ゲート：未合格。現在は「見送り」を優先してください。 理由：'+' / '.join(gate['reasons']))
with st.expander('未見回収率とは？',expanded=False):
    st.write('AIが学習に使っていない未来側の期間だけで、過去時点の予想を再現して100円ずつ購入したと仮定した回収率です。100%が損益分岐、100%未満は赤字、100%超はその未見検証期間では黒字です。')
    st.caption('例：未見回収率85.2%なら、仮に10,000円購入した検証で約8,520円戻った水準を意味します。将来の利益を保証する数字ではありません。')

with st.expander('学習状況・券種別信頼度・条件別弱点（詳細）',expanded=False):
    st.subheader('自己学習状況')
    a,b,c,d,e=st.columns(5)
    a.metric('学習レース',f"{hist['race_id'].nunique():,}")
    b.metric('最新結果日',latest_date)
    c.metric('校正サンプル',f"{cal_stats.get('samples',0):,}")
    d.metric('未見検証',f"{gate['oos_bets']:,}/200" if gate.get('oos_bets',0)>0 else '未実施')
    e.metric('未見回収率',f"{gate['oos_roi']*100:.1f}%" if gate.get('oos_bets',0)>0 else '未実施')
    if cal_stats.get('brier_raw') is not None:
        delta=cal_stats['brier_raw']-cal_stats['brier_cal']
        st.caption(f"確率校正 Brier: 校正前 {cal_stats['brier_raw']:.4f} → 校正後 {cal_stats['brier_cal']:.4f}（改善 {delta:+.4f}）")
    st.subheader('時系列・利益学習')
    if temporal_stats is not None and not temporal_stats.empty:
        ts=temporal_stats.copy()
        ts['的中率']=ts['的中率'].map(lambda x:f'{x*100:.1f}%')
        ts['回収率']=ts['回収率'].map(lambda x:f'{x*100:.1f}%')
        ts['損益']=ts['損益'].map(lambda x:f'{x:,.0f}円')
        st.dataframe(ts,use_container_width=True,hide_index=True)
        st.caption('直近30日を最重視しつつ、90日・180日・長期でも崩れない購入条件を優先します。短期だけ偶然黒字の条件は採用しにくくしています。')
    if isinstance(current_stats,dict) and current_stats.get('window_stats'):
        st.caption('現在ルールの期間別検証: '+ ' / '.join([f"{x['window_days']}日: {x['bets']}件 ROI{x['roi']*100:.1f}%" for x in current_stats['window_stats']]))
    st.subheader('券種別の未見確率信頼度')
    rg=st.columns(4)
    for i,code in enumerate(('3t','3f','2t','2f')):
        g=ticket_gate.get(code,{})
        stat=ticket_stats.get(code,{})
        name=TICKET_NAMES[code]
        hr=stat.get('hit_rate'); mp=stat.get('mean_pred')
        label='信頼可' if g.get('passed') else '試験中'
        rg[i].metric(name,label,delta=f"未見 {g.get('samples',0)}件")
        if hr is not None and mp is not None:
            rg[i].caption(f"予測平均 {mp*100:.1f}% / 実的中 {hr*100:.1f}% / 安全係数 {stat.get('safe_factor',1.0):.2f}")
        if g.get('reasons'): rg[i].caption('・'.join(g['reasons']))
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
        rid=str(rid);race_groups[rid]=g.copy()
        raw_ticket_tables=probability_tables_from_trifecta(tri)
        race_ctx=current_race_context(g,tri)
        dq=race_data_quality(g)
        adjusted={}
        for code,df in raw_ticket_tables.items():
            z=calibrate_ticket_table(df,code,ticket_calibrators,ticket_stats)
            cf,_=context_factor(code,race_ctx,context_stats,min_samples=25)
            z=apply_context_factor(z,cf)
            z['data_quality_factor']=float(dq['factor'])
            z['prob']=np.clip(pd.to_numeric(z['prob'],errors='coerce').fillna(0)*float(dq['factor']),1e-6,1-1e-6)
            z['prob_pct']=z['prob']*100
            adjusted[code]=z
        prob_tables[rid]=adjusted
        race_groups[rid]['context_info']=[race_ctx]*len(race_groups[rid])
        base_rows.append({'race_id':rid,'race_date':g['race_date'].iloc[0],'場':g['venue'].iloc[0],'R':int(g['race_no'].iloc[0]),'締切':pd.Timestamp(g['closed_at_jst'].iloc[0]).strftime('%H:%M'),'残り分':float(g['minutes_left'].iloc[0]),'AI1位':top['combo'],'校正AI確率%':float(top['prob']*100),'利益選別スコア':float(score),'過去条件通過':bool(rule_pass),'確信度':float(sig['confidence']),'データ準備完了':bool(dq['ready']),'データ品質係数':float(dq['factor']),'データ品質理由':str(dq['reason'])})
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
    for code,pdf in prob_tables[rid].items():
        merged[code]=apply_market_overlay(merge_odds(pdf,odds.get(code,pd.DataFrame(columns=['combo','odds']))))
    details[rid]=merged
    reliable={code:df for code,df in merged.items() if ticket_gate.get(code,{}).get('passed')}
    candidate_source=reliable if reliable else (merged if learning_mode=='none' else {})
    safe,value=choose_ticket_candidates(candidate_source);priority,priority_reason=priority_candidate(safe,value)
    if priority is None:
        # No ticket type has enough OOS probability reliability yet. Keep the race in details only.
        continue
    complete=sum(int(merged[c]['odds'].notna().sum()>0) for c in merged)
    pcode=priority.get('ticket_code')
    tgate=ticket_gate.get(pcode,{})
    reference=bool(r['過去条件通過'] and bool(r.get('データ準備完了',False)) and tgate.get('passed') and pd.notna(priority.get('expected_value')) and float(priority['expected_value'])>=MIN_EV)
    final_buy=bool(gate['passed'] and reference)
    def fields(x,prefix):
        if x is None:return {prefix+'券種':'—',prefix+'買い目':'—',prefix+'確率%':np.nan,prefix+'オッズ':np.nan,prefix+'期待値':np.nan}
        return {prefix+'券種':x['ticket'],prefix+'買い目':x['combo'],prefix+'確率%':float(x['prob']*100),prefix+'オッズ':x.get('odds'),prefix+'期待値':x.get('expected_value')}
    row={'race_id':rid,'race_date':r['race_date'],'場':r['場'],'R':r['R'],'締切':r['締切'],'残り分':r['残り分'],'判断':'買い' if final_buy else '見送り','優先券種':priority['ticket'],'優先買い目':priority['combo'],'優先確率%':float(priority['prob']*100),'優先オッズ':priority.get('odds'),'優先期待値':priority.get('expected_value'),'優先理由':priority_reason,'利益選別スコア':r['利益選別スコア'],'過去条件通過':r['過去条件通過'],'オッズ更新':mode,'取得券種数':complete,'参考候補':reference,'実戦候補':final_buy,'確信度':r['確信度'],'データ準備完了':bool(r.get('データ準備完了',False)),'データ品質係数':r.get('データ品質係数',np.nan),'データ品質理由':r.get('データ品質理由','—')}
    row.update(fields(safe,'高確率'));row.update(fields(value,'高期待値'));race_rows.append(row)
races=pd.DataFrame(race_rows)


st.divider();st.header('本日の購入判断')
if races.empty:
    st.warning('実オッズまで揃った候補がありません。');st.stop()

summary=races.sort_values(['判断','残り分','優先期待値'],ascending=[True,True,False]).copy()
summary['場・R']=summary['場'].astype(str)+' '+summary['R'].astype(int).astype(str)+'R'
summary['優先案']=summary['優先券種'].astype(str)+' '+summary['優先買い目'].astype(str)
summary['高確率案']=summary['高確率券種'].astype(str)+' '+summary['高確率買い目'].astype(str)
summary['高期待値案']=summary['高期待値券種'].astype(str)+' '+summary['高期待値買い目'].astype(str)
summary['AI確率']=summary['優先確率%'].map(lambda x:f'{x:.2f}%' if pd.notna(x) else '—')
summary['実オッズ']=summary['優先オッズ'].map(lambda x:f'{x:.1f}倍' if pd.notna(x) else '—')
summary['EV']=summary['優先期待値'].map(lambda x:f'{x:.2f}' if pd.notna(x) else '—')
summary['残り']=summary['残り分'].map(lambda x:f'{x:.0f}分')
summary['最終推奨']=summary['判断'].map(lambda x:'購入' if x=='買い' else '見送り')

buy_rows=summary[summary['判断']=='買い'].sort_values(['残り分','優先期待値'],ascending=[True,False])
watch_rows=summary[summary['判断']=='見送り'].sort_values(['残り分','優先期待値'],ascending=[True,False])

# 最上段：買い候補だけを大きく表示
if buy_rows.empty:
    st.warning('## 現在、買い候補なし')
    st.caption('実戦ゲートまたは安全補正条件を満たすレースがありません。画面が「見送り」の間は購入しない運用です。')
else:
    st.success(f'## 買い候補 {len(buy_rows)}レース')
    for _,r in buy_rows.head(4).iterrows():
        with st.container(border=True):
            a,b,c,d,e,f=st.columns([1.0,1.25,1.0,1.25,1.25,1.0])
            a.success('### 買い')
            b.markdown(f"### {r['場・R']}")
            c.metric('締切まで',r['残り'])
            d.metric('券種',r['優先券種'])
            e.metric('買い目',r['優先買い目'])
            f.metric('EV',r['EV'])
            x1,x2,x3,x4=st.columns(4)
            x1.metric('AI確率',r['AI確率'])
            x2.metric('実オッズ',r['実オッズ'])
            x3.metric('高確率案',r['高確率案'])
            x4.metric('高期待値案',r['高期待値案'])
            st.caption('優先理由：'+str(r['優先理由']))

# 1ページ用の全レース一覧
st.subheader('購入判断一覧')
m1,m2,m3,m4=st.columns(4)
m1.metric('買い',f'{len(buy_rows)}レース')
m2.metric('見送り',f'{len(watch_rows)}レース')
m3.metric('評価対象',f'{len(summary)}レース')
m4.metric('実戦ゲート','合格' if gate['passed'] else '未合格')

compact_cols=['最終推奨','場・R','締切','残り','優先券種','優先買い目','AI確率','実オッズ','EV','高確率案','高期待値案']
compact=summary[compact_cols].copy()
st.dataframe(
    compact,
    use_container_width=True,
    hide_index=True,
    height=min(75+34*len(compact),500),
    column_config={
        '最終推奨':st.column_config.TextColumn('判断',width='small'),
        '場・R':st.column_config.TextColumn('場・R',width='small'),
        '締切':st.column_config.TextColumn('締切',width='small'),
        '残り':st.column_config.TextColumn('残り',width='small'),
        '優先券種':st.column_config.TextColumn('券種',width='small'),
        '優先買い目':st.column_config.TextColumn('買い目',width='small'),
        'AI確率':st.column_config.TextColumn('AI確率',width='small'),
        '実オッズ':st.column_config.TextColumn('実オッズ',width='small'),
        'EV':st.column_config.TextColumn('EV',width='small'),
        '高確率案':st.column_config.TextColumn('高確率案',width='medium'),
        '高期待値案':st.column_config.TextColumn('高期待値案',width='medium'),
    }
)

if not gate['passed']:
    st.info('実戦ゲート未合格のため、現在の最終推奨はすべて「見送り」です。候補のEVが高くても購入対象にはしません。')

with st.expander('見送り理由・候補カードを確認',expanded=False):
    st.caption('必要な場合だけ開いてください。通常の購入判断は上の一覧だけで完結します。')
    for _,r in summary.head(12).iterrows():
        with st.container(border=True):
            a,b,c,d=st.columns([1.0,1.2,1.0,2.2])
            (a.success if r['判断']=='買い' else a.warning)(f"### {r['判断']}")
            b.markdown(f"### {r['場・R']}")
            c.metric('残り',r['残り'])
            d.write('**理由：** '+('購入条件を満たす' if r['判断']=='買い' else ('ゲート未合格' if not gate['passed'] else '安全補正条件未達')))
            st.caption(f"優先 {r['優先案']} / AI {r['AI確率']} / オッズ {r['実オッズ']} / EV {r['EV']} / {r['優先理由']}")

with st.expander('条件別の得意不得意学習',expanded=False):
    ctx_cols=st.columns(4)
    ctx_dims=[('venue_ctx','競艇場別'),('lane1_ctx','1号艇条件'),('race_ctx','レース帯'),('wind_ctx','風条件')]
    for ci,(dim,label) in enumerate(ctx_dims):
        vals=[]
        for key,stat in ((context_stats.get('3t',{}) or {}).get(dim,{}) or {}).items():
            if int(stat.get('samples',0))>=25:
                vals.append((float(stat.get('factor',1.0)),key,int(stat.get('samples',0))))
        if vals:
            f,key,n=sorted(vals)[0]
            ctx_cols[ci].metric(label,f'弱点: {key}',delta=f'補正×{f:.2f} / {n}件')
        else:
            ctx_cols[ci].metric(label,'学習中')
    st.caption('不得意条件は未見予測での実的中率/予測確率を使い、確率を下げる方向にだけ補正します。')
# Live log: only 3連単 can use existing ledger settlement format without adding result parsing for other ticket types.
# v2.11 therefore logs all displayed recommendations to a separate generic CSV, while legacy 3t ledger remains intact.
log_path='data/v211_recommendations.csv'
try: oldlog=pd.read_csv(log_path)
except Exception: oldlog=pd.DataFrame()
newlog=races.copy();newlog['recorded_at']=now.strftime('%Y-%m-%d %H:%M:%S');newlog['strategy_version']=APP_VERSION
if not oldlog.empty:newlog=pd.concat([oldlog,newlog],ignore_index=True).drop_duplicates(['race_id','優先券種','優先買い目'],keep='first')
try:newlog.to_csv(log_path,index=False,encoding='utf-8-sig')
except Exception:pass

st.divider();st.header('券種別詳細（必要なときだけ確認）')
rid=st.selectbox('レースを選択',list(details.keys()),format_func=lambda x:f"{x} / {base.loc[base['race_id']==x,'場'].iloc[0]} {int(base.loc[base['race_id']==x,'R'].iloc[0])}R")
for code in ('3t','3f','2t','2f'):
    df=details[rid][code].copy().sort_values(['expected_value','prob'],ascending=False,na_position='last')
    with st.expander(f"{TICKET_NAMES[code]}（{int(df['odds'].notna().sum())}/{len(df)} オッズ取得）",expanded=(code!='3t')):
        show=df.head(30).copy();show['安全補正AI確率']=show['prob_pct'].map(lambda x:f'{x:.3f}%');show['実オッズ']=show['odds'].map(lambda x:f'{x:.1f}倍' if pd.notna(x) else '—');show['安全補正期待値']=show['expected_value'].map(lambda x:f'{x:.3f}' if pd.notna(x) else '—')
        if 'effective_factor' in show: show['安全係数']=show['effective_factor'].map(lambda x:f'{x:.3f}')
        if 'context_factor' in show: show['条件補正']=show['context_factor'].map(lambda x:f'{x:.3f}')
        if 'market_factor' in show: show['市場補正']=show['market_factor'].map(lambda x:f'{x:.3f}')
        cols=['combo','安全補正AI確率','実オッズ','安全補正期待値']+(['安全係数'] if '安全係数' in show else [])+(['条件補正'] if '条件補正' in show else [])+(['市場補正'] if '市場補正' in show else [])+(['odds_band','popularity_band'] if 'odds_band' in show else [])
        st.dataframe(show[cols],use_container_width=True,hide_index=True)

with st.expander('学習・校正・未見検証の詳細'):
    st.write({'学習最新日':latest_date,'学習レース数':int(hist['race_id'].nunique()),'未見検証数':gate['oos_bets'],'未見回収率%':round(gate['oos_roi']*100,1),'未見的中率%':round(gate['oos_hit_rate']*100,1),'ゲート':gate['passed'],'3連単校正サンプル':ticket_stats.get('3t',{}).get('samples',0),'3連複校正サンプル':ticket_stats.get('3f',{}).get('samples',0),'2連単校正サンプル':ticket_stats.get('2t',{}).get('samples',0),'2連複校正サンプル':ticket_stats.get('2f',{}).get('samples',0)})
    if folds is not None and not folds.empty:st.dataframe(folds,use_container_width=True,hide_index=True)

st.subheader('AI再学習')
st.caption('再学習は画面上部の「最新結果で再学習」から実行します。通常の「当日データ・オッズ更新」では保存済みモデルを再利用するため、長期学習は走りません。')
