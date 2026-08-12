from __future__ import annotations
import itertools
import pandas as pd
from .model import fit_models, trifecta_table, selection_signals
from .tickets import probability_tables_from_trifecta

def generate_walk_forward_predictions(hist, min_train_days=35, test_days=7, max_test_days=49):
    if hist is None or hist.empty:
        return pd.DataFrame()
    h = hist.copy()
    h["race_date_dt"] = pd.to_datetime(h["race_date"], errors="coerce")
    h = h[h["race_date_dt"].notna() & h["finish_num"].notna()].copy()
    dates = sorted(h["race_date_dt"].dt.normalize().drop_duplicates())
    if len(dates) < min_train_days + 7:
        return pd.DataFrame()

    test_dates = dates[-min(max_test_days, len(dates)-min_train_days):]
    rows = []
    for start_i in range(0, len(test_dates), test_days):
        window = test_dates[start_i:start_i+test_days]
        if not window:
            continue
        cutoff = window[0]
        train = h[h["race_date_dt"] < cutoff]
        test = h[h["race_date_dt"].dt.normalize().isin(window)]
        if train["race_id"].nunique() < 300:
            continue
        models = fit_models(train)
        if models is None:
            continue

        for rid, g in test.groupby("race_id"):
            actual = g["trifecta_result"].dropna()
            payout = pd.to_numeric(g["trifecta_payout"], errors="coerce").dropna()
            if not len(actual):
                continue
            try:
                tri = trifecta_table(models, g)
                top = tri.iloc[0]
                sig = selection_signals(tri)
                pred = str(top["combo"])
                act = str(actual.iloc[0])
                hit = pred == act
                pay = float(payout.iloc[0]) if len(payout) else 0.0
                ticket_tables = probability_tables_from_trifecta(tri)
                aa = tuple(int(x) for x in act.split("-"))
                actual_map = {
                    "3t": act,
                    "3f": "-".join(map(str, sorted(aa))),
                    "2t": f"{aa[0]}-{aa[1]}",
                    "2f": "-".join(map(str, sorted(aa[:2]))),
                }
                # Context fields are stored on every walk-forward prediction so
                # venue / lane-1 / race-number / weather strengths and weaknesses
                # are learned only from races the model had not seen.
                lane1_first_prob=float(tri.loc[tri['combo'].astype(str).str.startswith('1-'),'prob'].sum())
                def _first_num(col):
                    try:
                        return float(pd.to_numeric(g[col],errors='coerce').dropna().iloc[0])
                    except Exception:
                        return None
                context_extra={
                    'lane1_first_prob':lane1_first_prob,
                    'wind_speed':_first_num('wind_speed'),
                    'wave_height_cm':_first_num('wave_height_cm'),
                }
                ticket_extra = {}
                for code, tdf in ticket_tables.items():
                    if tdf is None or tdf.empty:
                        continue
                    tt = tdf.iloc[0]
                    tpred = str(tt["combo"])
                    ticket_extra[f"{code}_pred"] = tpred
                    ticket_extra[f"{code}_prob"] = float(tt["prob"])
                    ticket_extra[f"{code}_hit"] = bool(tpred == actual_map[code])

                rows.append({
                    "race_id": rid,
                    "race_date": str(g["race_date"].iloc[0])[:10],
                    "venue": g["venue"].iloc[0],
                    "stadium_no": int(g["stadium_no"].iloc[0]),
                    "race_no": int(g["race_no"].iloc[0]),
                    "pred": pred,
                    "prob": float(top["prob"]),
                    "prob_margin": float(sig["prob_margin"]),
                    "confidence": float(sig["confidence"]),
                    "first_lane": sig["first_lane"],
                    "actual": act,
                    "hit": bool(hit),
                    "payout": pay,
                    "stake": 100.0,
                    "return_yen": pay if hit else 0.0,
                    "profit_yen": (pay if hit else 0.0) - 100.0,
                    **context_extra,
                    **ticket_extra,
                })
            except Exception:
                pass
    return pd.DataFrame(rows)

def summarize(df):
    if df is None or df.empty:
        return {"races":0, "hit_rate":0.0, "roi":0.0, "profit":0.0}
    stake = 100.0 * len(df)
    ret = pd.to_numeric(df["return_yen"], errors="coerce").fillna(0).sum()
    return {
        "races": int(len(df)),
        "hit_rate": float(df["hit"].mean()),
        "roi": float(ret/stake) if stake else 0.0,
        "profit": float(ret-stake),
    }

def _apply_rule(df, rule):
    m = pd.Series(True, index=df.index)
    if rule.get("min_prob") is not None:
        m &= df["prob"] >= rule["min_prob"]
    if rule.get("min_margin") is not None:
        m &= df["prob_margin"] >= rule["min_margin"]
    if rule.get("min_conf") is not None:
        m &= df["confidence"] >= rule["min_conf"]
    if rule.get("first_lane") is not None:
        m &= df["first_lane"] == rule["first_lane"]
    if rule.get("race_no_min") is not None:
        m &= df["race_no"] >= rule["race_no_min"]
    if rule.get("race_no_max") is not None:
        m &= df["race_no"] <= rule["race_no_max"]
    return df[m].copy()

def _candidate_rules(train):
    if train.empty:
        return []
    def qs(col, vals):
        return sorted(set([0.0] + [float(train[col].quantile(q)) for q in vals if train[col].notna().any()]))

    probs = qs("prob", (.35,.50,.65,.75,.82,.88,.92))
    margins = qs("prob_margin", (.35,.50,.65,.75,.85))
    confs = qs("confidence", (.35,.50,.65,.75,.85))
    rules = []

    for p, m, c in itertools.product(probs, margins, confs):
        rules.append({
            "min_prob":p, "min_margin":m, "min_conf":c,
            "first_lane":None, "race_no_min":None, "race_no_max":None
        })

    for p, c, lane in itertools.product(probs, confs, (1, None)):
        for rmin, rmax in ((None,6), (7,None), (None,None)):
            rules.append({
                "min_prob":p, "min_margin":0.0, "min_conf":c,
                "first_lane":lane, "race_no_min":rmin, "race_no_max":rmax
            })
    return rules

def optimize_rule(train, min_bets=100):
    if train is None or train.empty:
        return None, {}
    best = None
    best_stats = {}
    best_score = -1e9

    for rule in _candidate_rules(train):
        x = _apply_rule(train, rule)
        n = len(x)
        if n < min_bets:
            continue
        s = summarize(x)
        shrunk = (s["roi"]*n + 1.0*250) / (n+250)
        coverage = min(n/max(len(train),1), 0.30) / 0.30
        hit_quality = min(s["hit_rate"]/0.12, 1.0)
        score = (shrunk-1.0)*100 + 1.5*coverage + 0.5*hit_quality
        if score > best_score:
            best_score = score
            best = rule
            best_stats = s | {"shrunk_roi":shrunk, "score":score}

    return best, best_stats

def _date_window(df, days=None):
    if df is None or df.empty or days is None:
        return df.copy() if df is not None else pd.DataFrame()
    x=df.copy()
    d=pd.to_datetime(x.get("race_date"),errors="coerce")
    if not d.notna().any():
        return x
    cutoff=d.max()-pd.Timedelta(days=int(days)-1)
    return x[d>=cutoff].copy()


def _rule_key(rule):
    if rule is None:return None
    return tuple((k,rule.get(k)) for k in sorted(rule))


def optimize_rule_temporal(train, windows=(30,90,180,None), min_bets=100):
    """Choose a rule that survives multiple historical horizons.

    Recent windows receive more weight, but a rule is penalized when it only
    works in one short slice.  All decisions are based on past data only.
    """
    if train is None or train.empty:
        return None, {}
    t=train.copy()
    t["date_dt"]=pd.to_datetime(t.get("race_date"),errors="coerce")
    t=t[t["date_dt"].notna()].sort_values("date_dt")
    if t.empty:return None,{}

    # Find promising rules independently in each horizon, then compare the
    # small candidate set across every available horizon. This is materially
    # cheaper and less overfit than searching the full grid four times jointly.
    candidates=[]
    horizon_defs=[]
    # Full-grid search is expensive. Search only a compact set of horizons:
    # recent 30d, medium 90d, and all available history. 180d is still used
    # for evaluation below when enough history exists, but does not trigger an
    # additional full grid search.
    search_windows=(30,90,None)
    for w in windows:
        x=_date_window(t,w)
        if x.empty:continue
        horizon_defs.append((w,x))
        if w not in search_windows:
            continue
        mb=max(35,min(min_bets,int(max(35,len(x)*0.06))))
        rule,stats=optimize_rule(x,min_bets=mb)
        if rule is not None:
            candidates.append(rule)
    if not candidates:
        return optimize_rule(t,min_bets=min_bets)

    unique={_rule_key(r):r for r in candidates}
    weights={30:0.45,90:0.30,180:0.15,None:0.10}
    best=None;best_stats={};best_score=-1e9
    for rule in unique.values():
        rows=[];score=0.0;used_w=0.0;loss_penalty=0.0
        for w,x in horizon_defs:
            chosen=_apply_rule(x,rule)
            n=len(chosen)
            if n<20:
                continue
            ss=summarize(chosen)
            # Shrink strongly toward break-even to prevent a small lucky sample
            # from dominating the current strategy.
            prior=180.0 if w in (30,90) else 250.0
            shr=(ss["roi"]*n+1.0*prior)/(n+prior)
            ww=weights.get(w,0.10)
            used_w+=ww
            score += ww*((shr-1.0)*100.0)
            if ss["roi"]<0.90:
                loss_penalty += ww*(0.90-ss["roi"])*60.0
            rows.append({"window_days":"all" if w is None else int(w),"bets":n,"roi":ss["roi"],"hit_rate":ss["hit_rate"],"shrunk_roi":shr,"profit":ss["profit"]})
        if used_w<=0:continue
        score=score/used_w-loss_penalty
        # reward stability, not isolated jackpots
        profitable=sum(1 for r in rows if r["roi"]>1.0)
        score += 0.7*profitable
        if score>best_score:
            best_score=score;best=rule
            allstats=summarize(_apply_rule(t,rule))
            best_stats=allstats|{"temporal_score":score,"window_stats":rows,"profitable_windows":profitable,"available_windows":len(rows)}
    if best is None:
        return optimize_rule(t,min_bets=min_bets)
    return best,best_stats


def temporal_profit_summary(df):
    """ROI/hit/profit by 30/90/180/all unseen periods for UI and gating."""
    if df is None or df.empty:return pd.DataFrame()
    rows=[]
    for w in (30,90,180,None):
        x=_date_window(df,w)
        if x.empty:continue
        s=summarize(x)
        rows.append({"期間":"全期間" if w is None else f"直近{w}日","件数":s["races"],"的中率":s["hit_rate"],"回収率":s["roi"],"損益":s["profit"]})
    return pd.DataFrame(rows)

def nested_selector_backtest(preds, lookback_days=35, step_days=7, min_bets=90, min_selector_days=35):
    """Strict walk-forward selector validation.

    The selector for each test fold is fitted only on dates before that fold.
    v2.13.2 intentionally uses the robust single-horizon optimizer here; the
    richer temporal optimizer remains available for the *current* rule. This
    avoids the v2.13.0/1 failure mode where an overly strict multi-horizon rule
    selected zero bets in every future fold.
    """
    if preds is None or preds.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    p = preds.copy()
    p["date_dt"] = pd.to_datetime(p["race_date"], errors="coerce")
    p = p[p["date_dt"].notna()].sort_values("date_dt")
    dates = sorted(p["date_dt"].dt.normalize().drop_duplicates())
    if len(dates) < min_selector_days + step_days:
        return pd.DataFrame(), {}, pd.DataFrame()

    eval_rows = []
    fold_rows = []
    for i in range(min_selector_days, len(dates), step_days):
        test_dates = dates[i:i+step_days]
        if not test_dates:
            break
        train_dates = dates[max(0, i-lookback_days):i]
        train = p[p["date_dt"].dt.normalize().isin(train_dates)]
        test = p[p["date_dt"].dt.normalize().isin(test_dates)]
        if train.empty or test.empty:
            continue

        # Scale the minimum count to the actual training window so a compact
        # 35-day fold cannot become impossible merely because global history grew.
        fold_min = max(35, min(int(min_bets), int(max(35, len(train)*0.08))))
        rule, train_stats = optimize_rule(train, min_bets=fold_min)
        if rule is None:
            continue
        chosen = _apply_rule(test, rule)
        test_stats = summarize(chosen)
        fold_rows.append({
            "test_start": str(pd.Timestamp(test_dates[0]).date()),
            "test_end": str(pd.Timestamp(test_dates[-1]).date()),
            **rule,
            "train_bets": train_stats.get("races",0),
            "train_roi": train_stats.get("roi",0),
            "train_shrunk_roi": train_stats.get("shrunk_roi",0),
            "test_bets": test_stats.get("races",0),
            "test_hit_rate": test_stats.get("hit_rate",0),
            "test_roi": test_stats.get("roi",0),
            "test_profit": test_stats.get("profit",0),
        })
        if len(chosen):
            chosen = chosen.copy()
            chosen["fold_test_start"] = str(pd.Timestamp(test_dates[0]).date())
            eval_rows.append(chosen)

    out = pd.concat(eval_rows, ignore_index=True) if eval_rows else pd.DataFrame()
    folds = pd.DataFrame(fold_rows)
    return out, summarize(out), folds

def deployment_gate(metrics, folds,
                    min_oos_bets=200,
                    min_oos_roi=1.00,
                    min_positive_fold_ratio=0.60,
                    min_recent_fold_ratio=0.50):
    reasons = []

    if not metrics or metrics.get("races",0) < min_oos_bets:
        reasons.append(f"未見検証件数が{min_oos_bets}件未満")
    if not metrics or metrics.get("roi",0) <= min_oos_roi:
        reasons.append("未見データ総回収率が100%以下")

    positive_ratio = 0.0
    recent_ratio = 0.0
    valid_folds = pd.DataFrame()

    if folds is not None and not folds.empty:
        valid_folds = folds[folds["test_bets"] >= 15].copy()
        if len(valid_folds):
            positive_ratio = float((valid_folds["test_roi"] > 1.0).mean())
            recent = valid_folds.tail(min(3, len(valid_folds)))
            recent_ratio = float((recent["test_roi"] > 1.0).mean())
        else:
            reasons.append("各未見期間の購入件数が少なすぎる")
    else:
        reasons.append("未見期間別の検証結果がない")

    if positive_ratio < min_positive_fold_ratio:
        reasons.append(f"未見期間の黒字率が{min_positive_fold_ratio*100:.0f}%未満")
    if recent_ratio < min_recent_fold_ratio:
        reasons.append("直近未見期間の安定性不足")

    return {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "oos_bets": metrics.get("races",0) if metrics else 0,
        "oos_roi": metrics.get("roi",0) if metrics else 0,
        "oos_hit_rate": metrics.get("hit_rate",0) if metrics else 0,
        "positive_fold_ratio": positive_ratio,
        "recent_positive_ratio": recent_ratio,
        "valid_folds": len(valid_folds),
    }

def fit_current_selector(preds, lookback_days=180, min_bets=120):
    if preds is None or preds.empty:
        return None, {}
    p = preds.copy()
    p["date_dt"] = pd.to_datetime(p["race_date"], errors="coerce")
    p = p[p["date_dt"].notna()].sort_values("date_dt")
    dates = sorted(p["date_dt"].dt.normalize().drop_duplicates())
    if not dates:
        return None, {}
    keep = dates[-min(lookback_days, len(dates)):]
    train = p[p["date_dt"].dt.normalize().isin(keep)]
    return optimize_rule_temporal(train, min_bets=min_bets)

def current_buy_score(race_row, rule):
    if rule is None:
        return 0.0, False

    p = float(race_row.get("確率1位%",0))/100.0
    m = float(race_row.get("確率差",0))
    c = float(race_row.get("確信度",0))

    checks = [
        p >= float(rule.get("min_prob") or 0),
        m >= float(rule.get("min_margin") or 0),
        c >= float(rule.get("min_conf") or 0),
    ]

    if rule.get("first_lane") is not None:
        try:
            first = int(str(race_row.get("確率1位","")).split("-")[0])
            checks.append(first == int(rule["first_lane"]))
        except Exception:
            checks.append(False)

    rno = int(race_row.get("R",0) or 0)
    if rule.get("race_no_min") is not None:
        checks.append(rno >= int(rule["race_no_min"]))
    if rule.get("race_no_max") is not None:
        checks.append(rno <= int(rule["race_no_max"]))

    passed = all(checks)

    def ratio(x,t):
        t = float(t or 0)
        if t <= 0:
            return 1.0
        return min(x/t, 2.0)/2.0

    score = 100*(
        0.40*ratio(p,rule.get("min_prob")) +
        0.30*ratio(m,rule.get("min_margin")) +
        0.30*ratio(c,rule.get("min_conf"))
    )
    if not passed:
        score *= 0.45

    return float(score), bool(passed)
