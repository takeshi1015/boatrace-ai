from __future__ import annotations
from datetime import date, timedelta
import requests, pandas as pd
BASE='https://boatraceopenapi.github.io/api/v1'
NAMES={1:'桐生',2:'戸田',3:'江戸川',4:'平和島',5:'多摩川',6:'浜名湖',7:'蒲郡',8:'常滑',9:'津',10:'三国',11:'びわこ',12:'住之江',13:'尼崎',14:'鳴門',15:'丸亀',16:'児島',17:'宮島',18:'徳山',19:'下関',20:'若松',21:'芦屋',22:'福岡',23:'唐津',24:'大村'}
JCD={v:f'{k:02d}' for k,v in NAMES.items()}
HEADERS={'User-Agent':'BoatRaceAI-FreeDashboard/2.0'}
def get_json(url,timeout=30):
    r=requests.get(url,timeout=timeout,headers=HEADERS)
    if r.status_code==404:return None
    r.raise_for_status(); return r.json()
def fetch_today(): return get_json(f'{BASE}/today.json')
def fetch_day(d):
    d=pd.Timestamp(d).date(); return get_json(f'{BASE}/{d:%Y}/{d:%Y%m%d}.json')
def _pct(x):
    try:return float(x)/100.0 if x is not None else None
    except:return None
def flatten(payload, include_result=True):
    rows=[]; stadiums=(((payload or {}).get('programs') or {}).get('stadiums') or {})
    for sk,stadium in stadiums.items():
      for rk,race in ((stadium or {}).get('races') or {}).items():
        preview=race.get('preview') or {}; pr=preview.get('racers') or {}; result=race.get('result') or {}; rr=result.get('racers') or {}; tri=((result.get('payouts') or {}).get('trifecta') or []); tri_combo=tri[0].get('combination') if tri else None; tri_pay=tri[0].get('amount') if tri else None
        for ek,racer in (race.get('racers') or {}).items():
          p=pr.get(ek) or {}; z=rr.get(ek) or {}
          rows.append({'race_id':f"{race.get('date','')}-{int(sk):02d}-{int(rk):02d}",'race_date':race.get('date'),'stadium_no':int(sk),'venue':NAMES.get(int(sk),str(sk)),'race_no':int(rk),'closed_at':race.get('closed_at'),'lane':int(ek),'racer_id':racer.get('number'),'racer_name':racer.get('name'),'rank_number':racer.get('rank_number'),'age':racer.get('age'),'weight':racer.get('weight'),'flying_count':racer.get('flying_count'),'late_count':racer.get('late_count'),'avg_start':racer.get('average_start_timing'),'racer_win_rate':racer.get('national_win_rate'),'racer_2ren_rate':_pct(racer.get('national_top_2_percent')),'racer_3ren_rate':_pct(racer.get('national_top_3_percent')),'local_win_rate':racer.get('local_win_rate'),'local_2ren_rate':_pct(racer.get('local_top_2_percent')),'motor_no':racer.get('motor_number'),'motor_2ren_rate':_pct(racer.get('motor_top_2_percent')),'boat_no':racer.get('boat_number'),'boat_2ren_rate':_pct(racer.get('boat_top_2_percent')),'course':p.get('course_number'),'preview_start_timing':p.get('start_timing'),'exhibition_time':p.get('exhibition_time'),'wind_speed':preview.get('wind_speed'),'wave_height_cm':preview.get('wave_height'),'air_temperature':preview.get('air_temperature'),'water_temperature':preview.get('water_temperature'),'finish_num':z.get('place_number') if include_result else None,'win':1 if include_result and z.get('place_number')==1 else 0,'trifecta_result':tri_combo if include_result else None,'trifecta_payout':tri_pay if include_result else None})
    return pd.DataFrame(rows)
def historical_dataset(days=60):
    today=pd.Timestamp.now(tz='Asia/Tokyo').date(); frames=[]
    for i in range(days,0,-1):
        d=today-timedelta(days=i)
        if d < date(2026,1,1): continue
        try:
            obj=fetch_day(d)
            if obj:
                x=flatten(obj,True)
                if not x.empty and x['finish_num'].notna().any(): frames.append(x)
        except Exception: pass
    return pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
