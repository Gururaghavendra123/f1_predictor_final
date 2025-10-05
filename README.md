# 🏎️ F1 Race Predictor

<div align="center">

![F1 Race Predictor](https://img.shields.io/badge/F1-Race%20Predictor-red?style=for-the-badge&logo=formula1)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688?style=for-the-badge&logo=fastapi)

---

## 👤 Author

**Gururaghavendra P**

- LinkedIn: [My Linkedin](https://www.linkedin.com/in/gururaghavendra-padmanaban-455867290/)
- GitHub: [Gururaghavendra123](https://github.com/YourUsername)
- Email: pgururaghavandra1@gmail.com

---

**AI-Powered Formula 1 Race Outcome Prediction using Machine Learning**

[Live Demo](https://your-vercel-url.vercel.app) • [Report Bug](https://github.com/Gururaghavendra123/f1_predictor_final/issues) • [Request Feature](https://github.com/Gururaghavendra123/f1_predictor_final/issues)

</div>

---

## 📖 About The Project

F1 Race Predictor is an intelligent machine learning system that predicts Formula 1 race outcomes based on historical data, driver performance, weather conditions, and track characteristics. The system uses XGBoost and Random Forest algorithms to provide accurate predictions with confidence scores.

### ✨ Key Features

- 🤖 **AI-Powered Predictions** - Uses XGBoost & Random Forest for accurate race outcome predictions
- 📊 **Driver-Centric Model** - Focuses on individual driver performance, not team/car performance
- 🌤️ **Weather Integration** - Accounts for track temperature, air temperature, humidity, and rainfall
- 🏁 **Track Classification** - Analyzes street circuits, high-speed tracks, and technical tracks
- 🎨 **Modern UI** - Beautiful, responsive React interface with real-time predictions
- 📈 **Confidence Scores** - Provides win probability, podium probability, and overall confidence

### 🎯 Live Demo

**Frontend:** [https://your-app.vercel.app](https://your-app.vercel.app)  
**API Docs:** [https://your-api.render.com/docs](https://your-api.render.com/docs)

---

## 🛠️ Built With

### Backend

- **Python 3.8+** - Core programming language
- **FastAPI** - High-performance web framework
- **FastF1** - F1 data collection API
- **XGBoost** - Gradient boosting for predictions
- **scikit-learn** - Machine learning algorithms
- **Pandas & NumPy** - Data processing

### Frontend

- **React 18** - UI framework
- **Tailwind CSS** - Styling
- **Lucide React** - Icons
- **Vercel** - Deployment platform

### ML Models

- **Win Prediction** - XGBoost Classifier
- **Podium Prediction** - Random Forest Classifier
- **Position Prediction** - XGBoost Regressor

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Node.js 14 or higher
- npm or yarn
- 8GB+ RAM (for data processing)

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/f1-race-predictor.git
cd f1-race-predictor
```

#### 2. Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir models cache

# Start the backend server
python f1_api_backend.py
```

The API will be available at `http://localhost:8000`

#### 3. Train the Models

```bash
# Train models with historical data (2022-2025)
curl -X POST "http://localhost:8000/train" \
     -H "Content-Type: application/json" \
     -d '{"years": [2022, 2023, 2024, 2025], "retrain": false}'
```

Training takes approximately 15-20 minutes for 3-4 years of data.

#### 4. Frontend Setup

```bash
# Navigate to frontend directory
cd f1-frontend

# Install dependencies
npm install

# Start development server
npm start
```

The app will open at `http://localhost:3000`

---

## 💻 Usage

### Making Predictions

1. **Select Track** - Choose from 20+ F1 circuits
2. **Set Weather Conditions**
   - Track temperature (°C)
   - Air temperature (°C)
   - Humidity (%)
   - Rainfall (yes/no)
3. **Add Drivers** - Select drivers and their grid positions
4. **Predict Race** - Click the button to get AI predictions

### API Usage

```bash
# Get prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "race_conditions": {
         "track_name": "Monaco",
         "country": "Monaco",
         "track_temp": 25.0,
         "air_temp": 20.0,
         "humidity": 50.0,
         "rainfall": false,
         "track_type": "street"
       },
       "drivers": [
         {"driver_code": "VER", "grid_position": 1},
         {"driver_code": "HAM", "grid_position": 2}
       ]
     }'
```

---

## 📊 Model Performance

| Metric                  | Value               |
| ----------------------- | ------------------- |
| Win Prediction Accuracy | 77-82%              |
| Position MAE            | 2.0-2.5 positions   |
| Training Data           | 1,700+ race results |
| Years Covered           | 2022-2025           |
| Drivers Analyzed        | 25+ active drivers  |

---

## 🏗️ Project Structure

```
f1-race-predictor/
├── f1_data_collector.py      # Data collection from FastF1 API
├── f1_ml_predictor.py         # ML model training & prediction
├── f1_api_backend.py          # FastAPI server
├── requirements.txt           # Python dependencies
├── models/                    # Trained ML models
│   ├── win_model.pkl
│   ├── podium_model.pkl
│   └── position_model.pkl
├── cache/                     # FastF1 data cache
└── f1-frontend/              # React application
    ├── src/
    │   └── App.js            # Main React component
    ├── public/
    └── package.json
```

---

## 🎓 How It Works

### 1. Data Collection

- Downloads historical F1 race data using FastF1 API
- Processes race results, qualifying positions, and weather data
- Creates driver performance profiles (win rate, podium rate, average finish, DNF rate)

### 2. Feature Engineering

- **Driver Features**: Historical performance metrics independent of teams
- **Track Features**: Circuit classification (street/highspeed/technical)
- **Weather Features**: Temperature, humidity, rainfall conditions
- **Grid Position**: Starting position from qualifying

### 3. Model Training

Three separate models work together:

- **Win Predictor** (XGBoost): Predicts probability of winning
- **Podium Predictor** (Random Forest): Predicts top-3 finish probability
- **Position Predictor** (XGBoost Regressor): Predicts exact finishing position

### 4. Prediction

Combines all three models to provide:

- Predicted finishing position (1-20)
- Win probability (0-100%)
- Podium probability (0-100%)
- Overall confidence score

---

## 🌟 Key Design Decisions

### Driver-Centric Approach

Unlike traditional models that focus on teams/cars, this system prioritizes **individual driver performance**. This makes predictions robust to driver team changes.

**Benefits:**

- Predictions remain accurate when drivers switch teams
- Captures driver skill independent of car performance
- More reliable for long-term predictions

### Weather Integration

Real weather conditions significantly impact race outcomes:

- Hot temperatures favor certain drivers/teams
- Rain creates unpredictable conditions
- Humidity affects tire performance

### Track Classification

Different track types favor different driving styles:

- **Street Circuits** (Monaco, Singapore) - Precision and qualifying crucial
- **High-Speed** (Monza, Spa) - Top speed and slipstream important
- **Technical** (Hungary, Suzuka) - Downforce and cornering key

---

## 🚀 Deployment

### Frontend (Vercel)

```bash
cd f1-frontend
vercel --prod
```

### Backend (Render)

1. Create `render.yaml`
2. Connect GitHub repository
3. Deploy with one click

See [Deployment Guide](./DEPLOYMENT.md) for detailed instructions.

---

## 🤝 Contributing

Contributions are what make the open-source community amazing! Any contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 🙏 Acknowledgments

- [FastF1](https://github.com/theOehrly/Fast-F1) - Amazing F1 data API
- [XGBoost](https://xgboost.readthedocs.io/) - High-performance ML library
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://react.dev/) - UI framework
- Formula 1 community for inspiration

---

## 📈 Future Enhancements

- [ ] Tire strategy prediction
- [ ] Safety car probability
- [ ] Championship points prediction
- [ ] Historical race replay & analysis
- [ ] Mobile app (React Native)
- [ ] Real-time race updates
- [ ] Driver head-to-head comparisons
- [ ] Constructor championship predictions

---

<div align="center">

**⭐ Star this repo if you found it helpful! ⭐**

Made with ❤️ and ☕ by Gururaghavendra

</div>
