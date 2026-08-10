import requests
import pandas as pd

BASE = "https://boatraceopenapi.github.io/api/v1"
NAMES = {1:"桐生",2:"戸田",3:"江戸川",4:"平和島",5:"多摩川",6:"浜名湖",7:"蒲郡",8:"常滑",
9:"津",10:"三国",11:"びわこ",12:"住之江",13:"尼崎",14:"鳴門",15:"丸亀",16:"児島",
17:"宮島",18:"徳山",19:"下関",20:"若松",21:"芦屋",22:"福岡",23:"唐津",24:"大村"}

def fetch_today():
    r = requests.get(f"{BASE}/today.json", timeout=30,
                     headers={"User-Agent":"BoatRaceAI-FreeDashboard/1.0"})
    r.raise_for_status()
    return r.json()

def flatten(payload):
    rows=[]
    stadiums=(((payload or {}).get("programs") or {}).get("stadiums") or {})
    for sk, stadium in stadiums.items():
        for rk, race in ((stadium or {}).get("races") or {}).items():
            preview=race.get("preview") or {}
            pr=preview.get("racers") or {}
            for ek, racer in (race.get("racers") or {}).items():
                p=pr.get(ek) or {}
                rows.append({
                    "venue":NAMES.get(int(sk),str(sk)),
                    "race_no":int(rk),
                    "closed_at":race.get("closed_at"),
                    "lane":int(ek),
                    "racer_name":racer.get("name"),
                    "rank_number":racer.get("rank_number"),
                    "national_win_rate":racer.get("national_win_rate"),
                    "local_win_rate":racer.get("local_win_rate"),
                    "motor_number":racer.get("motor_number"),
                    "motor_top_2_percent":racer.get("motor_top_2_percent"),
                    "boat_number":racer.get("boat_number"),
                    "boat_top_2_percent":racer.get("boat_top_2_percent"),
                    "exhibition_time":p.get("exhibition_time"),
                    "preview_course":p.get("course_number"),
                    "preview_start_timing":p.get("start_timing"),
                    "wind_speed":preview.get("wind_speed"),
                    "wave_height_cm":preview.get("wave_height"),
                    "air_temperature":preview.get("air_temperature"),
                    "water_temperature":preview.get("water_temperature"),
                })
    return pd.DataFrame(rows)
