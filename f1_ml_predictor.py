import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, mean_absolute_error
import joblib
import json

class F1Predictor:
    def __init__(self):
        self.win_model = None
        self.podium_model = None
        self.position_model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_columns = []
        self.trained = False
        
    def prepare_features(self, df):
        """Prepare features for training"""
        print("Preparing features for training...")
        
        # Create a copy of the dataframe
        features_df = df.copy()
        
        # Encode categorical variables
        categorical_cols = ['driver', 'country']
        for col in categorical_cols:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    features_df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(features_df[col].astype(str))
                else:
                    # Handle new categories during prediction
                    known_classes = set(self.label_encoders[col].classes_)
                    features_df[f'{col}_encoded'] = features_df[col].apply(
                        lambda x: self.label_encoders[col].transform([str(x)])[0] if str(x) in known_classes else -1
                    )
        
        # Select numerical features
        numerical_features = [
            'grid_position', 'driver_win_rate', 'driver_podium_rate', 
            'driver_avg_finish', 'driver_avg_grid', 'driver_dnf_rate', 
            'driver_avg_points', 'track_temp', 'air_temp', 'humidity', 
            'rainfall', 'track_is_street', 'track_is_highspeed', 'track_is_technical'
        ]
        
        # Add encoded categorical features
        encoded_features = [f'{col}_encoded' for col in categorical_cols if col in features_df.columns]
        
        # Combine all features
        self.feature_columns = numerical_features + encoded_features
        
        # Select only available columns
        available_features = [col for col in self.feature_columns if col in features_df.columns]
        
        X = features_df[available_features].fillna(0)
        
        print(f"Features prepared: {len(available_features)} features")
        return X, available_features
    
    def train_models(self, training_data_path='f1_training_data.csv'):
        """Train the prediction models"""
        print("Loading training data...")
        df = pd.read_csv(training_data_path)
        
        print(f"Training data shape: {df.shape}")
        
        # Prepare features
        X, self.feature_columns = self.prepare_features(df)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Prepare targets
        y_win = df['won_race'].fillna(0)
        y_podium = df['podium_finish'].fillna(0)
        y_position = df['finished_position'].fillna(21)
        
        print("Training models...")
        
        # Train win prediction model (XGBoost for better performance)
        print("Training win prediction model...")
        self.win_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.win_model.fit(X_scaled, y_win)
        
        # Train podium prediction model
        print("Training podium prediction model...")
        self.podium_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.podium_model.fit(X_scaled, y_podium)
        
        # Train position prediction model
        print("Training position prediction model...")
        self.position_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        self.position_model.fit(X_scaled, y_position)
        
        self.trained = True
        
        # Evaluate models
        X_train, X_test, y_win_train, y_win_test = train_test_split(
            X_scaled, y_win, test_size=0.2, random_state=42
        )
        
        win_pred = self.win_model.predict(X_test)
        win_accuracy = accuracy_score(y_win_test, win_pred)
        
        _, _, y_pos_train, y_pos_test = train_test_split(
            X_scaled, y_position, test_size=0.2, random_state=42
        )
        pos_pred = self.position_model.predict(X_test)
        pos_mae = mean_absolute_error(y_pos_test, pos_pred)
        
        print(f"Win prediction accuracy: {win_accuracy:.3f}")
        print(f"Position prediction MAE: {pos_mae:.3f}")
        
        # Save models
        self.save_models()
        
        return {
            'win_accuracy': win_accuracy,
            'position_mae': pos_mae,
            'features_count': len(self.feature_columns)
        }
    
    def predict_race_outcome(self, race_data):
        """Predict race outcomes for given race data"""
        if not self.trained:
            self.load_models()
        
        # Convert single prediction to DataFrame
        if isinstance(race_data, dict):
            race_data = [race_data]
        
        prediction_df = pd.DataFrame(race_data)
        
        # Prepare features
        X, _ = self.prepare_features(prediction_df)
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        win_prob = self.win_model.predict_proba(X_scaled)[:, 1]
        podium_prob = self.podium_model.predict_proba(X_scaled)[:, 1]
        predicted_position = self.position_model.predict(X_scaled)
        
        results = []
        for i in range(len(prediction_df)):
            result = {
                'driver': prediction_df.iloc[i]['driver'],
                'win_probability': float(win_prob[i]),
                'podium_probability': float(podium_prob[i]),
                'predicted_position': max(1, min(20, int(round(predicted_position[i])))),
                'confidence': float((win_prob[i] + podium_prob[i]) / 2)
            }
            results.append(result)
        
        # Sort by win probability
        results.sort(key=lambda x: x['win_probability'], reverse=True)
        
        return results
    
    def save_models(self):
        """Save trained models and preprocessors"""
        print("Saving models...")
        joblib.dump(self.win_model, 'models/win_model.pkl')
        joblib.dump(self.podium_model, 'models/podium_model.pkl')
        joblib.dump(self.position_model, 'models/position_model.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.label_encoders, 'models/label_encoders.pkl')
        
        # Save feature columns
        with open('models/feature_columns.json', 'w') as f:
            json.dump(self.feature_columns, f)
        
        print("Models saved successfully!")
    
    def load_models(self):
        """Load trained models and preprocessors"""
        try:
            self.win_model = joblib.load('models/win_model.pkl')
            self.podium_model = joblib.load('models/podium_model.pkl')
            self.position_model = joblib.load('models/position_model.pkl')
            self.scaler = joblib.load('models/scaler.pkl')
            self.label_encoders = joblib.load('models/label_encoders.pkl')
            
            with open('models/feature_columns.json', 'r') as f:
                self.feature_columns = json.load(f)
            
            self.trained = True
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
    
    def get_driver_list(self):
        """Get list of available drivers"""
        if 'driver' in self.label_encoders:
            return list(self.label_encoders['driver'].classes_)
        return []
    
    def create_race_prediction_template(self):
        """Create a template for race prediction input"""
        return {
            'driver': 'VER',  # Driver abbreviation
            'grid_position': 1,
            'track_temp': 25,
            'air_temp': 20,
            'humidity': 50,
            'rainfall': 0,  # 0 or 1
            'track_is_street': 0,  # 0 or 1
            'track_is_highspeed': 0,  # 0 or 1
            'track_is_technical': 0,  # 0 or 1
            'country': 'Monaco'
        }

# Example usage and testing
if __name__ == "__main__":
    import os
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize predictor
    predictor = F1Predictor()
    
    # Train models (assumes training data exists)
    if os.path.exists('f1_training_data.csv'):
        results = predictor.train_models()
        print("Training completed!")
        print(f"Results: {results}")
    else:
        print("Training data not found. Please run the data collection script first.")
        
    # Example prediction
    sample_race_data = [
        {
            'driver': 'VER',
            'grid_position': 1,
            'driver_win_rate': 0.4,
            'driver_podium_rate': 0.7,
            'driver_avg_finish': 3.2,
            'driver_avg_grid': 2.1,
            'driver_dnf_rate': 0.1,
            'driver_avg_points': 18.5,
            'track_temp': 25,
            'air_temp': 20,
            'humidity': 50,
            'rainfall': 0,
            'track_is_street': 0,
            'track_is_highspeed': 1,
            'track_is_technical': 0,
            'country': 'Italy',
            'driver_encoded': 0,
            'country_encoded': 0
        },
        {
            'driver': 'HAM',
            'grid_position': 3,
            'driver_win_rate': 0.25,
            'driver_podium_rate': 0.6,
            'driver_avg_finish': 4.1,
            'driver_avg_grid': 3.5,
            'driver_dnf_rate': 0.05,
            'driver_avg_points': 15.2,
            'track_temp': 25,
            'air_temp': 20,
            'humidity': 50,
            'rainfall': 0,
            'track_is_street': 0,
            'track_is_highspeed': 1,
            'track_is_technical': 0,
            'country': 'Italy',
            'driver_encoded': 1,
            'country_encoded': 0
        }
    ]
    
    if predictor.trained:
        predictions = predictor.predict_race_outcome(sample_race_data)
        print("\nSample predictions:")
        for pred in predictions:
            print(f"{pred['driver']}: {pred['win_probability']:.3f} win prob, pos {pred['predicted_position']}")