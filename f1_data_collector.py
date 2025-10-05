import fastf1
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
import os
import time

warnings.filterwarnings('ignore')

# Create cache directory if it doesn't exist
os.makedirs('cache', exist_ok=True)

# Enable fastf1 cache for faster data loading
fastf1.Cache.enable_cache('cache')

class F1DataCollector:
    def __init__(self):
        self.sessions_data = []
        self.driver_stats = {}
        
    def get_race_sessions(self, years=[2020, 2021, 2022, 2023, 2024]):
        """Download race session data from multiple years"""
        print("Downloading F1 session data...")
        
        current_date = datetime.now()
        current_year = current_date.year
        
        # Filter out future years
        valid_years = [year for year in years if year <= current_year]
        
        if len(valid_years) < len(years):
            future_years = [year for year in years if year > current_year]
            print(f"⚠️ Warning: Skipping future years: {future_years}")
        
        if not valid_years:
            print(f"❌ Error: No valid years to process. Current year is {current_year}")
            return []
        
        print(f"📅 Processing years: {valid_years}")
        
        for year in valid_years:
            try:
                print(f"\n--- Processing {year} season ---")
                
                # Add delay to avoid rate limiting
                time.sleep(2)
                
                schedule = fastf1.get_event_schedule(year)
                
                races_processed = 0
                races_skipped = 0
                
                for _, event in schedule.iterrows():
                    # Skip if event hasn't happened yet
                    if pd.notna(event['EventDate']):
                        event_date = pd.to_datetime(event['EventDate'])
                        if event_date > current_date:
                            print(f"⊘ Skipping: {year} {event['EventName']} (scheduled for {event_date.date()}, hasn't happened yet)")
                            races_skipped += 1
                            continue
                    
                    # Skip non-conventional race formats (sprint-only, etc.)
                    if event['EventFormat'] != 'conventional':
                        continue
                        
                    try:
                        # Add small delay between race downloads
                        time.sleep(1)
                        
                        # Get race session with error handling
                        session = fastf1.get_session(year, event['EventName'], 'R')
                        
                        # Try to load with laps and telemetry first
                        try:
                            session.load(laps=True, telemetry=False, weather=True, messages=False)
                        except:
                            # If that fails, try minimal load
                            session.load(laps=False, telemetry=False, weather=False, messages=False)
                        
                        # Check if we actually got results
                        if session.results is None or len(session.results) == 0:
                            print(f"⊘ Skipping {year} {event['EventName']}: No results data available")
                            races_skipped += 1
                            continue
                        
                        # Get qualifying session for grid positions
                        try:
                            quali = fastf1.get_session(year, event['EventName'], 'Q')
                            quali.load(laps=False, telemetry=False, weather=False, messages=False)
                        except:
                            # If quali fails, use race grid positions
                            quali = session
                        
                        # Collect session data
                        session_info = {
                            'year': year,
                            'event_name': event['EventName'],
                            'country': event['Country'],
                            'location': event['Location'],
                            'date': event['EventDate'],
                            'session_type': 'Race',
                            'results': session.results,
                            'quali_results': quali.results if hasattr(quali, 'results') else None,
                            'track_temp': None,
                            'air_temp': None,
                            'humidity': None,
                            'rainfall': False
                        }
                        
                        # Try to get weather data if available
                        try:
                            if hasattr(session, 'weather_data') and session.weather_data is not None:
                                if 'TrackTemp' in session.weather_data.columns:
                                    session_info['track_temp'] = session.weather_data['TrackTemp'].mean()
                                if 'AirTemp' in session.weather_data.columns:
                                    session_info['air_temp'] = session.weather_data['AirTemp'].mean()
                                if 'Humidity' in session.weather_data.columns:
                                    session_info['humidity'] = session.weather_data['Humidity'].mean()
                                if 'Rainfall' in session.weather_data.columns:
                                    session_info['rainfall'] = session.weather_data['Rainfall'].any()
                        except:
                            pass  # Weather data is optional
                        
                        self.sessions_data.append(session_info)
                        races_processed += 1
                        print(f"✓ Downloaded: {year} {event['EventName']}")
                        
                    except KeyboardInterrupt:
                        print("\n⚠️ Training interrupted by user")
                        raise
                    except Exception as e:
                        error_msg = str(e).lower()
                        # Check if it's a "data not available" type error
                        if any(keyword in error_msg for keyword in ['not found', 'no data', 'invalid', 'does not exist', 'no such', 'cannot find']):
                            print(f"⊘ Skipping {year} {event['EventName']}: Data not available yet")
                            races_skipped += 1
                        else:
                            print(f"✗ Failed to load {year} {event['EventName']}: {str(e)[:100]}")
                            races_skipped += 1
                        continue
                
                print(f"--- {year}: Downloaded {races_processed} races, skipped {races_skipped} races ---")
                        
            except Exception as e:
                print(f"❌ Failed to get schedule for {year}: {str(e)}")
                continue
                
        print(f"\n{'='*60}")
        print(f"✅ Total sessions collected: {len(self.sessions_data)}")
        print(f"{'='*60}\n")
        
        return self.sessions_data
    
    def create_driver_features(self):
        """Create driver-specific features based on historical performance"""
        print("Creating driver features...")
        
        driver_performance = {}
        
        for session in self.sessions_data:
            results = session['results']
            
            for _, result in results.iterrows():
                driver = result['Abbreviation']
                
                if driver not in driver_performance:
                    driver_performance[driver] = {
                        'races': 0,
                        'wins': 0,
                        'podiums': 0,
                        'points_total': 0,
                        'avg_finish_position': [],
                        'avg_grid_position': [],
                        'dnf_rate': 0,
                        'dnfs': 0
                    }
                
                # Update driver stats
                driver_performance[driver]['races'] += 1
                
                if result['Position'] == 1:
                    driver_performance[driver]['wins'] += 1
                    
                if result['Position'] <= 3 and pd.notna(result['Position']):
                    driver_performance[driver]['podiums'] += 1
                    
                if pd.notna(result['Points']):
                    driver_performance[driver]['points_total'] += result['Points']
                
                if pd.notna(result['Position']):
                    driver_performance[driver]['avg_finish_position'].append(result['Position'])
                else:
                    driver_performance[driver]['dnfs'] += 1
                
                if pd.notna(result['GridPosition']):
                    driver_performance[driver]['avg_grid_position'].append(result['GridPosition'])
        
        # Calculate averages and rates
        for driver in driver_performance:
            perf = driver_performance[driver]
            perf['win_rate'] = perf['wins'] / perf['races'] if perf['races'] > 0 else 0
            perf['podium_rate'] = perf['podiums'] / perf['races'] if perf['races'] > 0 else 0
            perf['avg_finish'] = np.mean(perf['avg_finish_position']) if perf['avg_finish_position'] else 20
            perf['avg_grid'] = np.mean(perf['avg_grid_position']) if perf['avg_grid_position'] else 20
            perf['dnf_rate'] = perf['dnfs'] / perf['races'] if perf['races'] > 0 else 0
            perf['avg_points_per_race'] = perf['points_total'] / perf['races'] if perf['races'] > 0 else 0
        
        self.driver_stats = driver_performance
        print(f"✓ Created features for {len(driver_performance)} drivers")
        return driver_performance
    
    def create_training_dataset(self):
        """Create the final training dataset"""
        print("Creating training dataset...")
        
        training_data = []
        
        for session in self.sessions_data:
            event_name = session['event_name']
            country = session['country']
            year = session['year']
            results = session['results']
            
            # Track characteristics (simplified)
            track_features = self.get_track_features(event_name, country)
            
            for _, result in results.iterrows():
                driver = result['Abbreviation']
                
                if driver in self.driver_stats:
                    driver_perf = self.driver_stats[driver]
                    
                    # Create feature row
                    feature_row = {
                        'driver': driver,
                        'year': year,
                        'event': event_name,
                        'country': country,
                        'grid_position': result['GridPosition'] if pd.notna(result['GridPosition']) else 20,
                        'driver_win_rate': driver_perf['win_rate'],
                        'driver_podium_rate': driver_perf['podium_rate'],
                        'driver_avg_finish': driver_perf['avg_finish'],
                        'driver_avg_grid': driver_perf['avg_grid'],
                        'driver_dnf_rate': driver_perf['dnf_rate'],
                        'driver_avg_points': driver_perf['avg_points_per_race'],
                        'track_temp': session.get('track_temp', 25),
                        'air_temp': session.get('air_temp', 20),
                        'humidity': session.get('humidity', 50),
                        'rainfall': int(session.get('rainfall', False)),
                        **track_features,
                        'finished_position': result['Position'] if pd.notna(result['Position']) else 21,
                        'points_scored': result['Points'] if pd.notna(result['Points']) else 0,
                        'won_race': 1 if result['Position'] == 1 else 0,
                        'podium_finish': 1 if (pd.notna(result['Position']) and result['Position'] <= 3) else 0
                    }
                    
                    training_data.append(feature_row)
        
        df = pd.DataFrame(training_data)
        print(f"✓ Training dataset created with {len(df)} rows")
        return df
    
    def get_track_features(self, event_name, country):
        """Get simplified track characteristics"""
        # Simplified track categorization
        street_circuits = ['Monaco', 'Singapore', 'Baku', 'Jeddah', 'Miami', 'Las Vegas']
        high_speed = ['Monza', 'Spa', 'Silverstone', 'Suzuka']
        technical = ['Hungary', 'Monaco', 'Singapore', 'Mexico']
        
        return {
            'track_is_street': 1 if any(track in event_name for track in street_circuits) else 0,
            'track_is_highspeed': 1 if any(track in event_name for track in high_speed) else 0,
            'track_is_technical': 1 if any(track in event_name for track in technical) else 0
        }

# Usage example
if __name__ == "__main__":
    # Initialize collector
    collector = F1DataCollector()
    
    # Download session data (this will take some time)
    print("="*60)
    print("F1 DATA COLLECTOR - Starting data collection")
    print("="*60)
    print()
    
    sessions = collector.get_race_sessions([2022, 2023, 2024, 2025])  # Protected against future dates
    
    if not sessions:
        print("❌ No sessions were collected. Exiting.")
        exit(1)
    
    # Create driver features
    driver_stats = collector.create_driver_features()
    
    # Create training dataset
    training_df = collector.create_training_dataset()
    
    # Save the dataset
    training_df.to_csv('f1_training_data.csv', index=False)
    print("\n" + "="*60)
    print(f"✅ Training data saved to f1_training_data.csv")
    print("="*60)
    
    # Display sample
    print("\nSample of training data:")
    print(training_df.head())
    print(f"\nDataset shape: {training_df.shape}")
    print(f"Features: {list(training_df.columns)}")