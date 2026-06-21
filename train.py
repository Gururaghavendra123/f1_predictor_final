"""
Offline training pipeline. Run:  python train.py [--collect]

Steps:
  1. (optional) collect new races into the parquet store  (--collect)
  2. build the leakage-free feature table
  3. temporal validation: train on <= 2024, test on 2025  (honest accuracy)
  4. walk-forward validation on 2026 if present            (regime-change proof)
  5. fit the final live model on ALL data (recency-weighted) + save snapshot

The API never trains; it only loads what this script saves.
"""
import argparse

import numpy as np
import pandas as pd

from f1_data_collector import load_raw, collect
from f1_features import build_training_table, build_snapshot
from f1_ml_predictor import F1Predictor

HALFLIFE_YEARS = 1.5  # a race this old counts half as much in the live model


def recency_weights(table, halflife_years=HALFLIFE_YEARS):
    dates = pd.to_datetime(table['race_date'])
    age_years = (dates.max() - dates).dt.days / 365.25
    return np.power(0.5, age_years / halflife_years).values


def evaluate(train_tbl, test_tbl, sample_weight=None):
    """Fit on train_tbl, score every race in test_tbl. Returns a metrics dict."""
    p = F1Predictor().fit(train_tbl, sample_weight=sample_weight)

    winner_hits = 0
    podium_overlap = 0.0
    abs_err = []
    n_races = 0

    for _, g in test_tbl.groupby(['year', 'round']):
        ranked = p.rank_race(g.to_dict('records'))
        actual = dict(zip(g['driver'], g['target_position']))

        # winner accuracy
        if actual.get(ranked[0]['driver']) == 1:
            winner_hits += 1

        # podium overlap (top-3 set intersection / 3)
        pred_top3 = {r['driver'] for r in ranked[:3]}
        actual_top3 = {d for d, pos in actual.items() if pos <= 3}
        podium_overlap += len(pred_top3 & actual_top3) / 3.0

        # position MAE on classified finishers only (ignore DNF == 20)
        for r in ranked:
            a = actual.get(r['driver'])
            if a is not None and a < 20:
                abs_err.append(abs(r['predicted_position'] - a))

        n_races += 1

    return {
        'races': n_races,
        'winner_accuracy': winner_hits / n_races if n_races else 0.0,
        'podium_overlap': podium_overlap / n_races if n_races else 0.0,
        'position_mae': float(np.mean(abs_err)) if abs_err else None,
    }


def _print_metrics(title, m):
    print(f"\n{title}")
    print(f"  races evaluated : {m['races']}")
    print(f"  winner accuracy : {m['winner_accuracy']:.1%}")
    print(f"  podium overlap  : {m['podium_overlap']:.1%}")
    mae = m['position_mae']
    print(f"  position MAE    : {mae:.2f}" if mae is not None else "  position MAE    : n/a")


def walk_forward(table, year=2026):
    """Train on everything before each `year` round, predict that round."""
    sub = table[table['year'] == year]
    if sub.empty:
        print(f"\n[walk-forward {year}] no {year} data in store yet — skipping.")
        return None

    rounds = sorted(sub['round'].unique())
    agg = {'races': 0, 'winner_hits': 0, 'podium_overlap': 0.0, 'abs_err': []}

    for rnd in rounds:
        race_date = sub[sub['round'] == rnd]['race_date'].iloc[0]
        train = table[table['race_date'] < race_date]
        test = sub[sub['round'] == rnd]
        if len(train) < 200:   # too little history to be meaningful
            continue
        m = evaluate(train, test)
        agg['races'] += m['races']
        agg['winner_hits'] += int(m['winner_accuracy'] * m['races'])
        agg['podium_overlap'] += m['podium_overlap'] * m['races']
        if m['position_mae'] is not None:
            agg['abs_err'].append(m['position_mae'])

    if agg['races'] == 0:
        print(f"\n[walk-forward {year}] not enough history yet — skipping.")
        return None

    result = {
        'races': agg['races'],
        'winner_accuracy': agg['winner_hits'] / agg['races'],
        'podium_overlap': agg['podium_overlap'] / agg['races'],
        'position_mae': float(np.mean(agg['abs_err'])) if agg['abs_err'] else None,
    }
    _print_metrics(f"[walk-forward {year}] step-by-step through the season", result)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--collect', action='store_true',
                    help='fetch any new/missing races before training (uses network)')
    args = ap.parse_args()

    if args.collect:
        print("Collecting new races first...")
        collect()

    raw = load_raw()
    if raw.empty:
        print("No raw data. Run `python f1_data_collector.py` first.")
        return

    table = build_training_table(raw)
    years = sorted(table['year'].unique())
    print(f"Feature table: {len(table)} rows, seasons {years}")

    # 3) temporal validation: past -> most recent complete season
    if 2025 in years and any(y <= 2024 for y in years):
        train_tbl = table[table['year'] <= 2024]
        test_tbl = table[table['year'] == 2025]
        _print_metrics("[validation] train 2022-2024 -> test 2025", evaluate(train_tbl, test_tbl))

    # 4) regime-change proof
    walk_forward(table, year=2026)

    # 5) final live model on ALL data, recency-weighted
    print("\nFitting final live model on all data (recency-weighted)...")
    weights = recency_weights(table)
    predictor = F1Predictor().fit(table, sample_weight=weights)
    snapshot = build_snapshot(raw)
    predictor.save(snapshot)
    print(f"Snapshot as_of {snapshot['as_of']}, "
          f"{len(snapshot['drivers'])} drivers, {len(snapshot['teams'])} teams.")
    print("\nDone. Restart the API to load the new model.")


if __name__ == "__main__":
    main()
