"""
F1 data collector — incremental, rate-limit friendly.

Builds a raw per-race results store under data/races/{year}_R{round}.parquet.
One row per driver per race with the raw fields needed for feature building
(grid, finish, team, weather, era). Feature engineering lives in f1_features.py.

Key properties:
  * Incremental  — only fetches races not already stored (re-runs are cheap/offline).
  * Future-safe  — never network-loads a race whose session is in the future.
  * Lean         — no separate qualifying fetch (grid is already in race results),
                   which roughly halves API calls vs the old collector.
  * Resilient    — backs off and retries on rate-limit errors.

Run:  python f1_data_collector.py
"""
import os
import time
import warnings

import fastf1
import pandas as pd

warnings.filterwarnings('ignore')

CACHE_DIR = 'cache'
DATA_DIR = os.path.join('data', 'races')
DEFAULT_YEARS = [2022, 2023, 2024, 2025, 2026]
ERA_BOUNDARY_YEAR = 2026  # year >= this -> regulation era 1 (new 2026 rules)

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)


def _race_path(year, rnd):
    return os.path.join(DATA_DIR, f"{year}_R{int(rnd):02d}.parquet")


def _utc_now_naive():
    return pd.Timestamp.now(tz='UTC').tz_localize(None)


def _race_date_utc(event):
    """Race session start as naive-UTC Timestamp, or None if unknown."""
    try:
        d = pd.Timestamp(event.get_session_date('Race', utc=True))
        if d.tzinfo is not None:
            d = d.tz_convert('UTC').tz_localize(None)
        return d
    except Exception:
        return None


def _load_race(year, rnd, max_retries=3):
    """Load a race session with rate-limit backoff. Returns session or None."""
    delay = 5
    for attempt in range(1, max_retries + 1):
        try:
            session = fastf1.get_session(year, rnd, 'R')
            session.load(laps=False, telemetry=False, weather=True, messages=False)
            return session
        except Exception as e:
            msg = str(e).lower()
            if 'rate limit' in msg or '429' in msg or 'too many requests' in msg:
                print(f"    rate limited, backing off {delay}s "
                      f"(attempt {attempt}/{max_retries})")
                time.sleep(delay)
                delay *= 2
                continue
            # data genuinely not available (e.g. not published yet)
            print(f"    load failed: {str(e)[:90]}")
            return None
    print(f"    gave up after {max_retries} retries")
    return None


def _mean_weather(session):
    wx = {'track_temp': None, 'air_temp': None, 'humidity': None, 'rainfall': False}
    try:
        wd = session.weather_data
        if wd is not None and len(wd):
            if 'TrackTemp' in wd:
                wx['track_temp'] = float(wd['TrackTemp'].mean())
            if 'AirTemp' in wd:
                wx['air_temp'] = float(wd['AirTemp'].mean())
            if 'Humidity' in wd:
                wx['humidity'] = float(wd['Humidity'].mean())
            if 'Rainfall' in wd:
                wx['rainfall'] = bool(wd['Rainfall'].any())
    except Exception:
        pass
    return wx


def _extract(session, year, rnd, event):
    """Flatten a loaded race session into a per-driver DataFrame."""
    res = session.results
    if res is None or len(res) == 0:
        return None

    wx = _mean_weather(session)
    race_date = _race_date_utc(event)
    era = 1 if year >= ERA_BOUNDARY_YEAR else 0

    rows = []
    for _, r in res.iterrows():
        rows.append({
            'year': year,
            'round': int(rnd),
            'event': event['EventName'],
            'country': event['Country'],
            'race_date': race_date,
            'era': era,
            'driver': r.get('Abbreviation'),
            'driver_number': str(r.get('DriverNumber')),
            'team': r.get('TeamName'),
            'grid_position': r.get('GridPosition'),
            'finish_position': r.get('Position'),          # NaN if DNF
            'classified_position': str(r.get('ClassifiedPosition')),
            'status': r.get('Status'),                     # 'Finished', '+1 Lap', DNF reason...
            'points': r.get('Points'),
            'track_temp': wx['track_temp'],
            'air_temp': wx['air_temp'],
            'humidity': wx['humidity'],
            'rainfall': wx['rainfall'],
        })
    return pd.DataFrame(rows)


def collect(years=DEFAULT_YEARS, force=False, throttle=1.0):
    """Fetch any missing, already-run races into the parquet store.

    years    : seasons to consider.
    force    : re-fetch even if a race parquet already exists.
    throttle : seconds to sleep between network fetches (rate-limit hygiene).
    """
    now = _utc_now_naive()
    valid_years = [y for y in years if y <= now.year]
    future_years = [y for y in years if y > now.year]
    if future_years:
        print(f"skipping future years: {future_years}")

    total_new = 0
    for year in valid_years:
        print(f"\n=== {year} ===")
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:
            print(f"schedule {year} failed: {e}")
            continue

        for _, event in sched.iterrows():
            rnd = event['RoundNumber']
            if rnd == 0 or str(event.get('EventFormat', '')).lower() == 'testing':
                continue  # pre-season testing, not a race

            path = _race_path(year, rnd)
            if os.path.exists(path) and not force:
                continue  # already stored -> no network call

            # future gate: never load a race that hasn't happened yet
            rdate = _race_date_utc(event)
            if rdate is not None and rdate >= now:
                print(f"  skip future: R{rnd} {event['EventName']} ({rdate.date()})")
                continue

            print(f"  fetch: R{rnd} {event['EventName']}")
            session = _load_race(year, rnd)
            if session is None:
                continue
            df = _extract(session, year, rnd, event)
            if df is None or df.empty:
                print("    no results, skip")
                continue

            df.to_parquet(path, index=False)
            total_new += 1
            time.sleep(throttle)

    print(f"\nDONE. new races stored: {total_new}")
    return total_new


def load_raw(years=None):
    """Concat the stored race parquet files into one time-ordered DataFrame."""
    if not os.path.isdir(DATA_DIR):
        return pd.DataFrame()

    frames = []
    for fname in sorted(os.listdir(DATA_DIR)):
        if not fname.endswith('.parquet'):
            continue
        if years is not None and int(fname.split('_')[0]) not in years:
            continue
        frames.append(pd.read_parquet(os.path.join(DATA_DIR, fname)))

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(['race_date', 'year', 'round']).reset_index(drop=True)


if __name__ == "__main__":
    print("=" * 60)
    print("F1 DATA COLLECTOR — incremental raw store")
    print("=" * 60)

    collect()

    raw = load_raw()
    n_races = raw[['year', 'round']].drop_duplicates().shape[0] if len(raw) else 0
    print(f"\nRaw store: {len(raw)} driver-rows across {n_races} races")
    if len(raw):
        cols = ['year', 'round', 'event', 'driver', 'team',
                'grid_position', 'finish_position', 'era']
        print(raw[cols].head(8).to_string(index=False))
