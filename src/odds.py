from __future__ import annotations
import re
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
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}

def official_url(date_yyyymmdd, venue, race_no):
    jcd = JCD.get(venue)
    if not jcd:
        return None
    return (
        "https://www.boatrace.jp/owpc/pc/race/odds3t"
        f"?rno={int(race_no)}&jcd={jcd}&hd={date_yyyymmdd}"
    )

def empty_odds():
    return pd.DataFrame(columns=ODDS_COLUMNS)

def _clean_text(x):
    return re.sub(r"\s+", " ", str(x or "")).strip()

def _to_int_boat(x):
    s = _clean_text(x)
    m = re.fullmatch(r"[1-6]", s)
    return int(s) if m else None

def _to_float_odds(x):
    s = _clean_text(x).replace(",", "")
    # 発売前/未発売/欠場等の記号は数値にしない
    if s in {"", "-", "—", "‐", "―", "欠", "返還"}:
        return None
    m = re.search(r"(?<!\d)(\d+(?:\.\d+)?)(?!\d)", s)
    if not m:
        return None
    try:
        v = float(m.group(1))
        return v if v > 0 else None
    except Exception:
        return None

def _expand_html_table(table):
    """
    rowspan/colspan を展開して2次元の文字列行列へ変換する。
    BOAT RACE公式3連単表は、2着艇にrowspanが使われるため、
    これを展開しないと120通りを復元できない。
    """
    rows = table.find_all("tr")
    grid = []
    carry = {}  # col -> [remaining_rows, text]

    for tr in rows:
        row = []
        col = 0

        def fill_carry_until_free():
            nonlocal col
            while col in carry:
                remaining, text = carry[col]
                row.append(text)
                remaining -= 1
                if remaining <= 0:
                    del carry[col]
                else:
                    carry[col] = [remaining, text]
                col += 1

        fill_carry_until_free()

        for cell in tr.find_all(["th", "td"], recursive=False):
            fill_carry_until_free()
            text = _clean_text(cell.get_text(" ", strip=True))
            try:
                rowspan = max(1, int(cell.get("rowspan", 1)))
            except Exception:
                rowspan = 1
            try:
                colspan = max(1, int(cell.get("colspan", 1)))
            except Exception:
                colspan = 1

            for _ in range(colspan):
                row.append(text)
                if rowspan > 1:
                    carry[col] = [rowspan - 1, text]
                col += 1

        # trailing rowspan cells
        if carry:
            max_col = max(carry)
            while col <= max_col:
                if col in carry:
                    remaining, text = carry[col]
                    row.append(text)
                    remaining -= 1
                    if remaining <= 0:
                        del carry[col]
                    else:
                        carry[col] = [remaining, text]
                else:
                    row.append("")
                col += 1

        grid.append(row)

    width = max((len(r) for r in grid), default=0)
    return [r + [""] * (width - len(r)) for r in grid]

def _extract_120_from_matrix(matrix):
    """
    公式表の本体は6つの1着艇ブロックが横並びで、
    各ブロックは [2着, 3着, オッズ] の3列。
    rowspan展開後、20行 x 18列から120通りを復元する。
    """
    if not matrix:
        return {}

    width = max(len(r) for r in matrix)
    found_best = {}

    # 余分な先頭列等があってもよいよう18列窓を走査
    for start in range(0, max(1, width - 18 + 1)):
        found = {}
        for row in matrix:
            if len(row) < start + 18:
                continue
            for block in range(6):
                base = start + block * 3
                second = _to_int_boat(row[base])
                third = _to_int_boat(row[base + 1])
                odd = _to_float_odds(row[base + 2])
                first = block + 1

                if (
                    second is not None
                    and third is not None
                    and odd is not None
                    and len({first, second, third}) == 3
                ):
                    found[f"{first}-{second}-{third}"] = odd

        if len(found) > len(found_best):
            found_best = found

    return found_best

def _extract_from_official_tables(soup):
    best = {}

    # 「3連単オッズ」の直後にある表を最優先
    heading = None
    for tag in soup.find_all(["h2", "h3", "h4", "div", "p"]):
        if "3連単オッズ" in _clean_text(tag.get_text(" ", strip=True)):
            heading = tag
            break

    candidates = []
    if heading is not None:
        nxt = heading.find_next("table")
        if nxt is not None:
            candidates.append(nxt)

    # DOM変更に備え全tableも候補にする
    candidates.extend(soup.find_all("table"))

    seen = set()
    for table in candidates:
        ident = id(table)
        if ident in seen:
            continue
        seen.add(ident)
        matrix = _expand_html_table(table)
        found = _extract_120_from_matrix(matrix)
        if len(found) > len(best):
            best = found
        if len(best) >= 120:
            break

    return best

def _fallback_flat_text(soup):
    """
    将来DOMが変わった場合の補助。
    明示的に「1-2-3 12.3」の形で出た場合のみ拾う。
    """
    found = {}
    alltext = soup.get_text(" ", strip=True)
    pat = (
        r"([1-6])\s*[-－]\s*([1-6])\s*[-－]\s*([1-6])"
        r"\s+([0-9]+(?:\.[0-9]+)?)"
    )
    for m in re.finditer(pat, alltext):
        a, b, c, o = m.groups()
        if len({a, b, c}) == 3:
            try:
                found[f"{a}-{b}-{c}"] = float(o)
            except Exception:
                pass
    return found

def fetch_trifecta_odds(date_yyyymmdd, venue, race_no, timeout=8):
    url = official_url(date_yyyymmdd, venue, race_no)
    if not url:
        return empty_odds(), None

    try:
        r = requests.get(
            url,
            timeout=timeout,
            headers={**BROWSER_HEADERS, "Referer": "https://www.boatrace.jp/"},
        )
        r.raise_for_status()

        # BOAT RACE公式は日本語HTML。requests推定が外れた場合に備える
        if not r.encoding or r.encoding.lower() == "iso-8859-1":
            r.encoding = r.apparent_encoding or "utf-8"

        soup = BeautifulSoup(r.text, "lxml")

        # 公式が「データはありません」と返した場合は無理に推測しない
        page_text = _clean_text(soup.get_text(" ", strip=True))
        if "データはありません" in page_text:
            return empty_odds(), url

        found = _extract_from_official_tables(soup)

        if len(found) < 20:
            fallback = _fallback_flat_text(soup)
            if len(fallback) > len(found):
                found = fallback

        if not found:
            return empty_odds(), url

        df = pd.DataFrame(
            [{"combo": k, "odds": v} for k, v in found.items()],
            columns=ODDS_COLUMNS,
        )
        df["odds"] = pd.to_numeric(df["odds"], errors="coerce")
        df = (
            df.dropna(subset=["odds"])
              .drop_duplicates("combo", keep="last")
              .sort_values("combo")
              .reset_index(drop=True)
        )
        return df, url

    except Exception:
        return empty_odds(), url
