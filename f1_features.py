"""
Feature builder — the single source of truth for how raw race results become
model features. Used by BOTH training (train.py) and serving (the API), so the
two can never drift.

Leakage-free by construction: features for race R are computed only from races
that happened strictly BEFORE R. We do one chronological pass, emit each race's
feature rows from the accumulated history, THEN fold that race into the history.

Three signal groups:
  * driver priors  — career + recent form (stable skill, transfers across eras)
  * team / car pace — career + recent form (the part that RESETS each rule era;
                      recent form tracks the current pecking order)
  * track history  — driver's past results at this specific circuit

Plus per-race known-before-the-race inputs: grid position, weather, track type,
era flag.
"""
from collections import defaultdict

import numpy as np
import pandas as pd

FORM_WINDOW = 5            # races counted as "recent form" for a driver
DNF_FILL = 20.0           # finishing position assigned to a DNF (back of field)
DEFAULT_FINISH = 11.0     # midfield prior when a driver/team has no history
DEFAULT_GRID = 11.0

# Numeric columns fed to the model. Order matters — keep stable across train/serve.
FEATURE_COLUMNS = [
    'grid_position', 'era',
    'track_temp', 'air_temp', 'humidity', 'rainfall',
    'track_is_street', 'track_is_highspeed', 'track_is_technical',
    'driver_career_races', 'driver_career_avg_finish', 'driver_career_win_rate',
    'driver_career_podium_rate', 'driver_career_dnf_rate', 'driver_career_avg_points',
    'driver_form_avg_finish', 'driver_form_avg_grid', 'driver_form_avg_points',
    'team_career_avg_finish', 'team_career_avg_points',
    'team_form_avg_finish', 'team_form_avg_grid', 'team_form_avg_points',
    'driver_track_races', 'driver_track_avg_finish',
]

# Circuit categorisation by event-name keyword (flags are independent / may overlap).
_STREET = ('Monaco', 'Singapore', 'Azerbaijan', 'Saudi', 'Miami', 'Las Vegas')
_HIGHSPEED = ('Italian', 'Belgian', 'British', 'Austrian')
_TECHNICAL = ('Hungarian', 'Japanese', 'Monaco', 'Singapore', 'Dutch', 'Emilia')


def _track_type(event_name):
    name = str(event_name)
    return {
        'track_is_street': int(any(k in name for k in _STREET)),
        'track_is_highspeed': int(any(k in name for k in _HIGHSPEED)),
        'track_is_technical': int(any(k in name for k in _TECHNICAL)),
    }


def _summary(records):
    """Aggregate a list of prior race records into rate/average stats.

    Each record: {'finish': float|nan (nan == DNF), 'grid': float, 'points': float}
    """
    n = len(records)
    if n == 0:
        return {'avg_finish': DEFAULT_FINISH, 'avg_grid': DEFAULT_GRID,
                'avg_points': 0.0, 'win_rate': 0.0, 'podium_rate': 0.0, 'dnf_rate': 0.0}

    finishes = [r['finish'] for r in records if pd.notna(r['finish'])]
    grids = [r['grid'] for r in records if pd.notna(r['grid']) and r['grid'] > 0]
    pts = [r['points'] if pd.notna(r['points']) else 0.0 for r in records]
    wins = sum(1 for r in records if pd.notna(r['finish']) and r['finish'] == 1)
    podiums = sum(1 for r in records if pd.notna(r['finish']) and r['finish'] <= 3)
    dnfs = sum(1 for r in records if pd.isna(r['finish']))

    return {
        'avg_finish': float(np.mean(finishes)) if finishes else DEFAULT_FINISH,
        'avg_grid': float(np.mean(grids)) if grids else DEFAULT_GRID,
        'avg_points': float(np.mean(pts)),
        'win_rate': wins / n,
        'podium_rate': podiums / n,
        'dnf_rate': dnfs / n,
    }


def _driver_feats(recs, window=FORM_WINDOW):
    c = _summary(recs)
    f = _summary(recs[-window:])
    return {
        'driver_career_races': len(recs),
        'driver_career_avg_finish': c['avg_finish'],
        'driver_career_win_rate': c['win_rate'],
        'driver_career_podium_rate': c['podium_rate'],
        'driver_career_dnf_rate': c['dnf_rate'],
        'driver_career_avg_points': c['avg_points'],
        'driver_form_avg_finish': f['avg_finish'],
        'driver_form_avg_grid': f['avg_grid'],
        'driver_form_avg_points': f['avg_points'],
    }


def _team_feats(recs, window=FORM_WINDOW):
    c = _summary(recs)
    f = _summary(recs[-window * 2:])  # 2 cars per team -> wider window for same race count
    return {
        'team_career_avg_finish': c['avg_finish'],
        'team_career_avg_points': c['avg_points'],
        'team_form_avg_finish': f['avg_finish'],
        'team_form_avg_grid': f['avg_grid'],
        'team_form_avg_points': f['avg_points'],
    }


def _track_feats(finishes):
    cls = [f for f in finishes if pd.notna(f)]
    return {
        'driver_track_races': len(finishes),
        'driver_track_avg_finish': float(np.mean(cls)) if cls else DEFAULT_FINISH,
    }


def _race_inputs(row):
    """Known-before-the-race inputs from a raw row."""
    inp = {
        'grid_position': row['grid_position'] if pd.notna(row['grid_position']) else DEFAULT_GRID,
        'era': int(row['era']),
        'track_temp': row['track_temp'] if pd.notna(row['track_temp']) else 25.0,
        'air_temp': row['air_temp'] if pd.notna(row['air_temp']) else 20.0,
        'humidity': row['humidity'] if pd.notna(row['humidity']) else 50.0,
        'rainfall': int(bool(row['rainfall'])),
    }
    inp.update(_track_type(row['event']))
    return inp


def _rec(row):
    return {'finish': row['finish_position'], 'grid': row['grid_position'],
            'points': row['points']}


def build_training_table(raw, form_window=FORM_WINDOW):
    """Chronological, leakage-free feature matrix with target_position.

    Returns a DataFrame with FEATURE_COLUMNS + meta (driver/team/event/year/round/
    race_date) + target_position (DNF -> DNF_FILL).
    """
    raw = raw.sort_values(['race_date', 'year', 'round']).reset_index(drop=True)

    d_hist = defaultdict(list)            # driver -> [rec]
    t_hist = defaultdict(list)            # team   -> [rec]
    dt_hist = defaultdict(list)           # (driver, event) -> [finish]

    out = []
    for (yr, rnd), g in raw.groupby(['year', 'round'], sort=False):
        # 1) emit features for every driver using history BEFORE this race
        for _, r in g.iterrows():
            drv, team, ev = r['driver'], r['team'], r['event']
            feats = {}
            feats.update(_driver_feats(d_hist[drv], form_window))
            feats.update(_team_feats(t_hist[team], form_window))
            feats.update(_track_feats(dt_hist[(drv, ev)]))
            feats.update(_race_inputs(r))
            feats.update({
                'driver': drv, 'team': team, 'event': ev,
                'year': yr, 'round': rnd, 'race_date': r['race_date'],
                'target_position': r['finish_position'] if pd.notna(r['finish_position']) else DNF_FILL,
            })
            out.append(feats)

        # 2) fold this race into history (so the NEXT race sees it, this one didn't)
        for _, r in g.iterrows():
            rec = _rec(r)
            d_hist[r['driver']].append(rec)
            t_hist[r['team']].append(rec)
            dt_hist[(r['driver'], r['event'])].append(r['finish_position'])

    return pd.DataFrame(out)


def build_snapshot(raw, form_window=FORM_WINDOW):
    """Final per-driver / per-team stats AFTER all races — used by the API to
    build features for an upcoming race. JSON-serialisable."""
    raw = raw.sort_values(['race_date', 'year', 'round']).reset_index(drop=True)

    d_hist = defaultdict(list)
    t_hist = defaultdict(list)
    dt_hist = defaultdict(list)
    d_team = {}                            # driver -> latest team

    for _, r in raw.iterrows():
        rec = _rec(r)
        d_hist[r['driver']].append(rec)
        t_hist[r['team']].append(rec)
        dt_hist[(r['driver'], r['event'])].append(r['finish_position'])
        d_team[r['driver']] = r['team']

    drivers = {}
    for drv, recs in d_hist.items():
        feats = _driver_feats(recs, form_window)
        feats['current_team'] = d_team[drv]
        drivers[drv] = feats

    teams = {team: _team_feats(recs, form_window) for team, recs in t_hist.items()}

    driver_track = defaultdict(dict)
    for (drv, ev), finishes in dt_hist.items():
        driver_track[drv][ev] = _track_feats(finishes)

    as_of = raw['race_date'].max()
    return {
        'as_of': str(as_of.date()) if pd.notna(as_of) else None,
        'form_window': form_window,
        'feature_columns': FEATURE_COLUMNS,
        'drivers': drivers,
        'teams': teams,
        'driver_track': {k: v for k, v in driver_track.items()},
    }


def features_for_prediction(snapshot, driver, grid_position, event, era,
                            track_temp=25.0, air_temp=20.0, humidity=50.0,
                            rainfall=False, team=None):
    """Build one model-ready feature dict for an upcoming race from the snapshot.

    Unknown driver -> midfield defaults. Team defaults to the driver's last-known
    team in the snapshot (overridable for mid-season moves).
    """
    drv = snapshot['drivers'].get(driver, {})
    team = team or drv.get('current_team')
    tstats = snapshot['teams'].get(team, {})

    # track history — tolerant match so UI names ("Monaco Grand Prix") or short
    # names ("Monaco") both resolve to the stored event key.
    drv_tracks = snapshot.get('driver_track', {}).get(driver, {})
    track = drv_tracks.get(event)
    if track is None:
        track = next((v for k, v in drv_tracks.items()
                      if event in k or k in event), {})

    feats = {}
    # driver priors (default to neutral midfield when missing)
    feats['driver_career_races'] = drv.get('driver_career_races', 0)
    feats['driver_career_avg_finish'] = drv.get('driver_career_avg_finish', DEFAULT_FINISH)
    feats['driver_career_win_rate'] = drv.get('driver_career_win_rate', 0.0)
    feats['driver_career_podium_rate'] = drv.get('driver_career_podium_rate', 0.0)
    feats['driver_career_dnf_rate'] = drv.get('driver_career_dnf_rate', 0.0)
    feats['driver_career_avg_points'] = drv.get('driver_career_avg_points', 0.0)
    feats['driver_form_avg_finish'] = drv.get('driver_form_avg_finish', DEFAULT_FINISH)
    feats['driver_form_avg_grid'] = drv.get('driver_form_avg_grid', DEFAULT_GRID)
    feats['driver_form_avg_points'] = drv.get('driver_form_avg_points', 0.0)
    # team / car pace
    feats['team_career_avg_finish'] = tstats.get('team_career_avg_finish', DEFAULT_FINISH)
    feats['team_career_avg_points'] = tstats.get('team_career_avg_points', 0.0)
    feats['team_form_avg_finish'] = tstats.get('team_form_avg_finish', DEFAULT_FINISH)
    feats['team_form_avg_grid'] = tstats.get('team_form_avg_grid', DEFAULT_GRID)
    feats['team_form_avg_points'] = tstats.get('team_form_avg_points', 0.0)
    # track history
    feats['driver_track_races'] = track.get('driver_track_races', 0)
    feats['driver_track_avg_finish'] = track.get('driver_track_avg_finish', DEFAULT_FINISH)
    # race inputs
    feats['grid_position'] = grid_position
    feats['era'] = int(era)
    feats['track_temp'] = track_temp
    feats['air_temp'] = air_temp
    feats['humidity'] = humidity
    feats['rainfall'] = int(bool(rainfall))
    feats.update(_track_type(event))

    return feats
