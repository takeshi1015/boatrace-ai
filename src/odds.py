from __future__ import annotations
import re
from io import StringIO
from urllib.parse import urlencode

import requests
import pandas as pd
from bs4 import BeautifulSoup
from .data import JCD

ODDS_COLUMNS = ["combo", "odds"]

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/151.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ja-JP,ja;q=0.9,en-US;q=0.7,en;q=0.5",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
    "Connection": "keep-alive",
}

def official_url(date_yyyymmdd, venue, race_no, host="www.boatrace.jp"):
    jcd = JCD.get(venue)
    if not jcd:
        return None
    qs = urlencode({"hd": date_yyyymmdd, "jcd": jcd, "rno": int(race_no)})
    return f"https://{host}/owpc/pc/race/odds3t?{qs}"

def empty_odds():
    return pd.DataFrame(columns=ODDS_COLUMNS)

def _clean_text(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def _boat(x):
    s = _clean_text(x)
    return int(s) if re.fullmatch(r"[1-6]", s) else None

def _odd(x):
    s = _clean_text(x).replace(",", "")
    if s in {"", "-", "—", "‐", "―", "欠", "返還", "None", "nan"}:
        return None
    if not re.fullmatch(r"\d+(?:\.\d+)?", s):
        return None
    try:
        v = float(s)
        return v if v > 0 else None
    except Exception:
        return None

def _expand_table(table):
    rows = table.find_all("tr")
    grid, active = [], {}
    for tr in rows:
        cells = tr.find_all(["th","td"], recursive=False)
        row, col = [], 0

        def consume_active():
            nonlocal col
            while col in active:
                remain, txt = active[col]
                row.append(txt)
                remain -= 1
                if remain <= 0:
                    del active[col]
                else:
                    active[col] = [remain, txt]
                col += 1

        consume_active()
        for cell in cells:
            consume_active()
            txt = _clean_text(cell.get_text(" ", strip=True))
            try:
                rs = max(1, int(cell.get("rowspan", 1)))
            except Exception:
                rs = 1
            try:
                cs = max(1, int(cell.get("colspan", 1)))
            except Exception:
                cs = 1
            for _ in range(cs):
                row.append(txt)
                if rs > 1:
                    active[col] = [rs-1, txt]
                col += 1

        if active:
            max_col = max(active)
            while col <= max_col:
                if col in active:
                    remain, txt = active[col]
                    row.append(txt)
                    remain -= 1
                    if remain <= 0:
                        del active[col]
                    else:
                        active[col] = [remain, txt]
                else:
                    row.append("")
                col += 1
        grid.append(row)

    width = max((len(r) for r in grid), default=0)
    return [r + [""]*(width-len(r)) for r in grid]

def _matrix_to_odds(matrix):
    if not matrix:
        return {}
    width = max((len(r) for r in matrix), default=0)
    best = {}
    # 6 first-place blocks x 3 columns = 18 columns
    for start in range(max(1, width-18+1)):
        found = {}
        for row in matrix:
            if len(row) < start+18:
                continue
            for block in range(6):
                base = start + block*3
                second = _boat(row[base])
                third = _boat(row[base+1])
                odd = _odd(row[base+2])
                first = block+1
                if second and third and odd and len({first,second,third}) == 3:
                    found[f"{first}-{second}-{third}"] = odd
        if len(found) > len(best):
            best = found
    return best

def _parse_html(html):
    soup = BeautifulSoup(html, "lxml")
    page_text = _clean_text(soup.get_text(" ", strip=True))
    diag = {
        "has_3t_title": "3連単オッズ" in page_text,
        "no_data": "データはありません" in page_text,
        "tables": len(soup.find_all("table")),
        "html_chars": len(html),
    }
    if diag["no_data"]:
        return {}, diag

    best = {}
    for table in soup.find_all("table"):
        found = _matrix_to_odds(_expand_table(table))
        if len(found) > len(best):
            best = found
        if len(best) >= 120:
            break

    # pandas read_html fallback; rowspan is often expanded automatically
    if len(best) < 100:
        try:
            for df in pd.read_html(StringIO(html), displayed_only=False):
                mat = df.fillna("").astype(str).values.tolist()
                found = _matrix_to_odds(mat)
                if len(found) > len(best):
                    best = found
        except Exception:
            pass

    diag["parsed_count"] = len(best)
    return best, diag

def _parse_reader_markdown(text):
    """
    Reader fallback returns the official page as Markdown/text.
    The official odds table appears as 20 body rows:
      first row of each 4-row group has 18 tokens,
      next 3 rows have 12 tokens.
    Reconstruct each of six first-place blocks.
    """
    lines = text.splitlines()
    start = 0
    for i, line in enumerate(lines):
        if "3連単オッズ" in line:
            start = i
            break

    body = lines[start:start+120]
    rows = []
    for line in body:
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        # ignore headers/separators and racer-name header rows
        numeric = []
        for c in cells:
            c2 = re.sub(r"[*_`]", "", c).strip()
            if re.fullmatch(r"\d+(?:\.\d+)?", c2):
                numeric.append(c2)
        if len(numeric) in (12,18):
            rows.append(numeric)

    # We expect 20 odds rows; find any 20-row window with 5 group starts (18 tokens)
    best = {}
    for s in range(max(1, len(rows)-20+1)):
        window = rows[s:s+20]
        if len(window) < 20:
            continue
        seconds = {k: None for k in range(1,7)}
        found = {}
        for ri, nums in enumerate(window):
            pos = 0
            for first in range(1,7):
                # every fourth row carries second-place value for each block
                if len(nums) == 18:
                    second = _boat(nums[pos]); third = _boat(nums[pos+1]); odd = _odd(nums[pos+2]); pos += 3
                    seconds[first] = second
                else:
                    second = seconds[first]
                    third = _boat(nums[pos]); odd = _odd(nums[pos+1]); pos += 2
                if second and third and odd and len({first,second,third}) == 3:
                    found[f"{first}-{second}-{third}"] = odd
        if len(found) > len(best):
            best = found
    return best, {"reader_lines": len(lines), "reader_numeric_rows": len(rows), "parsed_count": len(best)}

def _df_from_found(found):
    if not found:
        return empty_odds()
    df = pd.DataFrame([{"combo":k, "odds":v} for k,v in found.items()], columns=ODDS_COLUMNS)
    df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
    return (
        df.dropna(subset=["odds"])
          .drop_duplicates("combo", keep="last")
          .sort_values("combo")
          .reset_index(drop=True)
    )

def _direct_fetch(url, timeout=7):
    """
    Normal public-page access only. No CAPTCHA solving or access-control bypass.
    Prime a normal HTTP session first because BOAT RACE may set cookies on initial pages.
    """
    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)
    diagnostics = {"route":"official-direct", "url":url}
    try:
        # lightweight cookie/session priming
        session.get("https://www.boatrace.jp/", timeout=min(timeout,5), allow_redirects=True)
    except Exception:
        pass
    try:
        r = session.get(url, timeout=timeout, allow_redirects=True, headers={"Referer":"https://www.boatrace.jp/"})
        diagnostics.update({
            "http_status": r.status_code,
            "final_url": r.url,
            "content_type": r.headers.get("content-type",""),
            "bytes": len(r.content),
        })
        r.raise_for_status()
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding or "utf-8"
        found, pdiag = _parse_html(r.text)
        diagnostics.update(pdiag)
        return _df_from_found(found), diagnostics
    except Exception as e:
        diagnostics["error"] = type(e).__name__
        return empty_odds(), diagnostics

def _reader_fetch(url, timeout=10):
    """
    Free fallback transport that reads the public BOAT RACE official page.
    Data source remains the official page; transport is labeled in diagnostics.
    """
    diagnostics = {"route":"official-via-reader", "source_url":url}
    try:
        reader_url = "https://r.jina.ai/" + url
        r = requests.get(
            reader_url,
            timeout=timeout,
            headers={
                "User-Agent":"Mozilla/5.0",
                "Accept":"text/plain,text/markdown,*/*",
                "X-No-Cache":"true",
            },
        )
        diagnostics.update({
            "http_status":r.status_code,
            "bytes":len(r.content),
        })
        r.raise_for_status()
        found, pdiag = _parse_reader_markdown(r.text)
        diagnostics.update(pdiag)
        return _df_from_found(found), diagnostics
    except Exception as e:
        diagnostics["error"] = type(e).__name__
        return empty_odds(), diagnostics

def fetch_trifecta_odds(date_yyyymmdd, venue, race_no, timeout=8):
    # Try both canonical host variants.
    urls = [
        official_url(date_yyyymmdd, venue, race_no, "www.boatrace.jp"),
        official_url(date_yyyymmdd, venue, race_no, "boatrace.jp"),
    ]
    diagnostics = []

    for url in [u for u in urls if u]:
        df, diag = _direct_fetch(url, timeout=timeout)
        diagnostics.append(diag)
        if len(df) >= 20:
            return df, url, diagnostics

    # Fallback: reader transport for the exact official page.
    if urls[0]:
        df, diag = _reader_fetch(urls[0], timeout=max(10, timeout))
        diagnostics.append(diag)
        if len(df) >= 20:
            return df, urls[0], diagnostics

    return empty_odds(), urls[0] if urls else None, diagnostics

# ===== v2.11 multi-ticket odds =====
TICKET_TARGETS={'3f':20,'2t':30,'2f':15}

def _official_ticket_url(date_yyyymmdd,venue,race_no,path,host='www.boatrace.jp'):
    jcd=JCD.get(venue)
    if not jcd: return None
    qs=urlencode({'hd':date_yyyymmdd,'jcd':jcd,'rno':int(race_no)})
    return f'https://{host}/owpc/pc/race/{path}?{qs}'

def _matrix_to_3f(matrix):
    if not matrix:return {}
    width=max((len(r) for r in matrix),default=0); best={}
    for start in range(max(1,width-12+1)):
        found={}
        for row in matrix:
            if len(row)<start+12:continue
            for block in range(4):
                base=start+block*3; first=block+1
                second=_boat(row[base]); third=_boat(row[base+1]); odd=_odd(row[base+2])
                if second and third and odd and first<second<third:
                    found[f'{first}-{second}-{third}']=odd
        if len(found)>len(best):best=found
    return best

def _matrix_to_2t(matrix):
    if not matrix:return {}
    width=max((len(r) for r in matrix),default=0); best={}
    for start in range(max(1,width-12+1)):
        found={}
        for row in matrix:
            if len(row)<start+12:continue
            for block in range(6):
                base=start+block*2; first=block+1
                second=_boat(row[base]); odd=_odd(row[base+1])
                if second and odd and first!=second:
                    found[f'{first}-{second}']=odd
        if len(found)>len(best):best=found
    return best

def _matrix_to_2f(matrix):
    if not matrix:return {}
    width=max((len(r) for r in matrix),default=0); best={}
    for start in range(max(1,width-10+1)):
        found={}
        for row in matrix:
            if len(row)<start+10:continue
            for block in range(5):
                base=start+block*2; first=block+1
                second=_boat(row[base]); odd=_odd(row[base+1])
                if second and odd and first<second:
                    found[f'{first}-{second}']=odd
        if len(found)>len(best):best=found
    return best

def _heading_table(soup,heading):
    for tag in soup.find_all(['h2','h3','h4','div','p']):
        if heading in _clean_text(tag.get_text(' ',strip=True)):
            return tag.find_next('table')
    return None

def _parse_ticket_html(html,code):
    soup=BeautifulSoup(html,'lxml')
    heading={'3f':'3連複オッズ','2t':'2連単オッズ','2f':'2連複オッズ'}[code]
    table=_heading_table(soup,heading)
    parser={'3f':_matrix_to_3f,'2t':_matrix_to_2t,'2f':_matrix_to_2f}[code]
    best={}
    candidates=[table] if table is not None else []
    candidates+=soup.find_all('table')
    seen=set()
    for t in candidates:
        if t is None or id(t) in seen:continue
        seen.add(id(t)); found=parser(_expand_table(t))
        if len(found)>len(best):best=found
        if len(best)>=TICKET_TARGETS[code]:break
    return best,{'heading':heading,'tables':len(soup.find_all('table')),'parsed_count':len(best)}

def _markdown_matrix(text,heading):
    lines=text.splitlines(); start=None
    for i,line in enumerate(lines):
        if heading in line:start=i;break
    if start is None:return []
    rows=[]
    for line in lines[start+1:start+100]:
        if line.startswith('### ') and rows:break
        if '|' not in line:continue
        cells=[re.sub(r'[*_`]','',c).strip() for c in line.strip().strip('|').split('|')]
        if cells and not all(set(c)<=set('-: ') for c in cells if c):rows.append(cells)
    width=max((len(r) for r in rows),default=0)
    return [r+['']*(width-len(r)) for r in rows]

def _reader_ticket(url,code,timeout=10):
    diag={'route':'official-via-reader','source_url':url,'ticket':code}
    try:
        r=requests.get('https://r.jina.ai/'+url,timeout=timeout,headers={'User-Agent':'Mozilla/5.0','Accept':'text/plain,text/markdown,*/*','X-No-Cache':'true'})
        diag.update({'http_status':r.status_code,'bytes':len(r.content)});r.raise_for_status()
        heading={'3f':'3連複オッズ','2t':'2連単オッズ','2f':'2連複オッズ'}[code]
        parser={'3f':_matrix_to_3f,'2t':_matrix_to_2t,'2f':_matrix_to_2f}[code]
        found=parser(_markdown_matrix(r.text,heading));diag['parsed_count']=len(found)
        return _df_from_found(found),diag
    except Exception as e:
        diag['error']=type(e).__name__;return empty_odds(),diag

def fetch_ticket_odds(date_yyyymmdd,venue,race_no,code,timeout=8):
    if code=='3t':return fetch_trifecta_odds(date_yyyymmdd,venue,race_no,timeout)
    path='odds3f' if code=='3f' else 'odds2tf'
    url=_official_ticket_url(date_yyyymmdd,venue,race_no,path)
    diagnostics=[]
    if not url:return empty_odds(),None,diagnostics
    try:
        s=requests.Session();s.headers.update(BROWSER_HEADERS)
        try:s.get('https://www.boatrace.jp/',timeout=min(timeout,5))
        except Exception:pass
        r=s.get(url,timeout=timeout,headers={'Referer':'https://www.boatrace.jp/'})
        diag={'route':'official-direct','url':url,'ticket':code,'http_status':r.status_code,'bytes':len(r.content)}
        r.raise_for_status()
        if not r.encoding or r.encoding.lower()=='iso-8859-1':r.encoding=r.apparent_encoding or 'utf-8'
        found,pdiag=_parse_ticket_html(r.text,code);diag.update(pdiag);diagnostics.append(diag)
        df=_df_from_found(found)
        if len(df)>=max(5,int(TICKET_TARGETS[code]*0.75)):return df,url,diagnostics
    except Exception as e:
        diagnostics.append({'route':'official-direct','url':url,'ticket':code,'error':type(e).__name__})
    df,diag=_reader_ticket(url,code,max(timeout,10));diagnostics.append(diag)
    return df,url,diagnostics

def fetch_all_ticket_odds(date_yyyymmdd,venue,race_no,timeout=8):
    out={};diags={}
    for code in ('3t','3f','2t','2f'):
        df,url,diag=fetch_ticket_odds(date_yyyymmdd,venue,race_no,code,timeout)
        out[code]=df;diags[code]=diag
    return out,diags
