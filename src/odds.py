import re
import requests
import pandas as pd
from bs4 import BeautifulSoup
from .data import JCD, HEADERS

ODDS_COLUMNS=['combo','odds']

def official_url(date_yyyymmdd,venue,race_no):
    jcd=JCD.get(venue)
    if not jcd:
        return None
    return f'https://www.boatrace.jp/owpc/pc/race/odds3t?hd={date_yyyymmdd}&jcd={jcd}&rno={int(race_no)}'

def empty_odds():
    return pd.DataFrame(columns=ODDS_COLUMNS)

def fetch_trifecta_odds(date_yyyymmdd,venue,race_no,timeout=20):
    url=official_url(date_yyyymmdd,venue,race_no)
    if not url:
        return empty_odds(),None
    try:
        r=requests.get(url,timeout=timeout,headers=HEADERS)
        r.raise_for_status()
        soup=BeautifulSoup(r.text,'lxml')
        found={}

        for tag in soup.find_all(attrs={'data-odds':True}):
            comb=tag.get('data-combination') or tag.get('data-combo') or ''
            val=tag.get('data-odds')
            if re.fullmatch(r'[1-6]-[1-6]-[1-6]',comb or ''):
                try:
                    found[comb]=float(val)
                except Exception:
                    pass

        alltext=soup.get_text(' ',strip=True)
        pat=r'([1-6])\s*[-－]\s*([1-6])\s*[-－]\s*([1-6])\s+([0-9]+(?:\.[0-9]+)?)'
        for m in re.finditer(pat,alltext):
            a,b,c,o=m.groups()
            if len({a,b,c})==3:
                try:
                    found[f'{a}-{b}-{c}']=float(o)
                except Exception:
                    pass

        if not found:
            return empty_odds(),url

        df=pd.DataFrame(
            [{'combo':k,'odds':v} for k,v in found.items()],
            columns=ODDS_COLUMNS
        )
        df['combo']=df['combo'].astype(str)
        df['odds']=pd.to_numeric(df['odds'],errors='coerce')
        df=df.dropna(subset=['odds']).drop_duplicates('combo',keep='last')
        return df,url
    except Exception:
        return empty_odds(),url
