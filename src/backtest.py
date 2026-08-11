from __future__ import annotations
import itertools
import pandas as pd
from .model import fit_models, trifecta_table, selection_signals

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

def nested_selector_backtest(preds, lookback_days=28, step_days=7, min_bets=100):
    if preds is None or preds.empty:
        return pd.DataFrame(), {}, pd.DataFrame()

    p = preds.copy()
    p["date_dt"] = pd.to_datetime(p["race_date"], errors="coerce")
    p = p[p["date_dt"].notna()].sort_values("date_dt")
    dates = sorted(p["date_dt"].dt.normalize().drop_duplicates())
    if len(dates) < lookback_days + step_days:
        return pd.DataFrame(), {}, pd.DataFrame()

    eval_rows = []
    fold_rows = []

    for i in range(lookback_days, len(dates), step_days):
        test_dates = dates[i:i+step_days]
        if not test_dates:
            break
        train_dates = dates[max(0, i-lookback_days):i]
        train = p[p["date_dt"].dt.normalize().isin(train_dates)]
        test = p[p["date_dt"].dt.normalize().isin(test_dates)]

        rule, train_stats = optimize_rule(train, min_bets=min_bets)
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

def fit_current_selector(preds, lookback_days=35, min_bets=120):
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
    return optimize_rule(train, min_bets=min_bets)

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
