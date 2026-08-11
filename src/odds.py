import re, requests, pandas as pd
from bs4 import BeautifulSoup
from .data import JCD, HEADERS
def official_url(date_yyyymmdd,venue,race_no):
    jcd=JCD.get(venue)
    if not jcd:return None
    return f'https://www.boatrace.jp/owpc/pc/race/odds3t?hd={date_yyyymmdd}&jcd={jcd}&rno={int(race_no)}'
def fetch_trifecta_odds(date_yyyymmdd,venue,race_no,timeout=20):
    url=official_url(date_yyyymmdd,venue,race_no)
    if not url:return pd.DataFrame(columns=['combo','odds']),None
    try:
        r=requests.get(url,timeout=timeout,headers=HEADERS); r.raise_for_status(); soup=BeautifulSoup(r.text,'lxml'); found={}
        for tag in soup.find_all(attrs={'data-odds':True}):
            comb=tag.get('data-combination') or tag.get('data-combo') or ''; val=tag.get('data-odds')
            if re.fullmatch(r'[1-6]-[1-6]-[1-6]',comb or ''):
                try:found[comb]=float(val)
                except:pass
        alltext=soup.get_text(' ',strip=True)
        for m in re.finditer(r'([1-6])\s*[-－]\s*([1-6])\s*[-－]\s*([1-6])\s+([0-9]+(?:\.[0-9]+)?)',alltext):
            a,b,c,o=m.groups()
            if len({a,b,c})==3:found[f'{a}-{b}-{c}']=float(o)
        return pd.DataFrame([{'combo':k,'odds':v} for k,v in found.items()]),url
    except Exception:return pd.DataFrame(columns=['combo','odds']),url
