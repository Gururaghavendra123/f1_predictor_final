import React, { useState, useEffect } from 'react';
import { Trophy, Zap, CloudRain, Thermometer, Users, MapPin, Settings, Play, RotateCcw } from 'lucide-react';

const F1Predictor = () => {
  const [isTraining, setIsTraining] = useState(false);
  const [isPredicting, setIsPredicting] = useState(false);
  const [modelsTrained, setModelsTrained] = useState(false);
  const [predictions, setPredictions] = useState([]);
  const [drivers, setDrivers] = useState([]);
  const [tracks, setTracks] = useState([]);
  
  // Form state
  const [raceConditions, setRaceConditions] = useState({
    track_name: 'Monaco',
    country: 'Monaco',
    track_temp: 25,
    air_temp: 20,
    humidity: 50,
    rainfall: false,
    track_type: 'street'
  });
  
  const [selectedDrivers, setSelectedDrivers] = useState([
    { driver_code: 'VER', grid_position: 1 },
    { driver_code: 'HAM', grid_position: 2 },
    { driver_code: 'LEC', grid_position: 3 },
    { driver_code: 'RUS', grid_position: 4 },
    { driver_code: 'NOR', grid_position: 5 }
  ]);

  const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  useEffect(() => {
    checkModelStatus();
    loadDrivers();
    loadTracks();
  }, []);

  const checkModelStatus = async () => {
    try {
      const response = await fetch(`${API_BASE}/model-info`);
      const data = await response.json();
      setModelsTrained(data.status === 'trained');
    } catch (error) {
      console.error('Error checking model status:', error);
    }
  };

  const loadDrivers = async () => {
    try {
      const response = await fetch(`${API_BASE}/drivers`);
      const data = await response.json();
      setDrivers(data.drivers);
    } catch (error) {
      console.error('Error loading drivers:', error);
    }
  };

  const loadTracks = async () => {
    try {
      const response = await fetch(`${API_BASE}/tracks`);
      const data = await response.json();
      setTracks(data.tracks);
    } catch (error) {
      console.error('Error loading tracks:', error);
    }
  };

  const handleTrain = async () => {
    setIsTraining(true);
    try {
      const response = await fetch(`${API_BASE}/train`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ years: [2022, 2023, 2024], retrain: false })
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log('Training completed:', data);
        setModelsTrained(true);
        alert('Models trained successfully!');
      } else {
        const error = await response.json();
        alert(`Training failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Training error:', error);
      alert('Training failed. Please check the console.');
    } finally {
      setIsTraining(false);
    }
  };

  const handlePredict = async () => {
    if (!modelsTrained) {
      alert('Please train models first!');
      return;
    }

    setIsPredicting(true);
    try {
      const requestData = {
        race_conditions: raceConditions,
        drivers: selectedDrivers
      };

      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestData)
      });

      if (response.ok) {
        const data = await response.json();
        
        // Sort by predicted position (1, 2, 3, 4, etc.)
        const sortedPredictions = data.sort((a, b) => a.predicted_position - b.predicted_position);
        
        setPredictions(sortedPredictions);
      } else {
        const error = await response.json();
        alert(`Prediction failed: ${error.detail}`);
      }
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Prediction failed. Please check the console.');
    } finally {
      setIsPredicting(false);
    }
  };

  const handleTrackChange = (trackName) => {
    const track = tracks.find(t => t.name === trackName);
    if (track) {
      setRaceConditions(prev => ({
        ...prev,
        track_name: trackName,
        country: track.country,
        track_type: track.type
      }));
    }
  };

  const updateDriver = (index, field, value) => {
    const updated = [...selectedDrivers];
    updated[index] = { ...updated[index], [field]: value };
    setSelectedDrivers(updated);
  };

  const addDriver = () => {
    if (selectedDrivers.length < 10) {
      setSelectedDrivers([...selectedDrivers, { driver_code: 'ALO', grid_position: selectedDrivers.length + 1 }]);
    }
  };

  const removeDriver = (index) => {
    setSelectedDrivers(selectedDrivers.filter((_, i) => i !== index));
  };

  const getPodiumColor = (position) => {
    const pos = parseInt(position);
    if (pos === 1) return 'bg-yellow-400 text-yellow-900';
    if (pos === 2) return 'bg-gray-400 text-gray-900';
    if (pos === 3) return 'bg-orange-500 text-white';
    return 'bg-gray-700 text-gray-300';
  };

  const getBorderColor = (position) => {
    const pos = parseInt(position);
    if (pos === 1) return 'border-yellow-500';
    if (pos === 2) return 'border-gray-400';
    if (pos === 3) return 'border-orange-500';
    return 'border-gray-600';
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-red-900 via-gray-900 to-black text-white">
      {/* Header */}
      <div className="bg-black/50 backdrop-blur-sm border-b border-red-600/30">
        <div className="max-w-7xl mx-auto px-4 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <Trophy className="h-8 w-8 text-red-500" />
              <h1 className="text-3xl font-bold bg-gradient-to-r from-red-500 to-white bg-clip-text text-transparent">
                F1 Race Predictor
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`px-3 py-1 rounded-full text-sm ${modelsTrained ? 'bg-green-600' : 'bg-yellow-600'}`}>
                {modelsTrained ? 'Models Ready' : 'Needs Training'}
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          
          {/* Training Panel */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Settings className="h-5 w-5 mr-2 text-red-500" />
                Model Training
              </h2>
              
              <div className="space-y-4">
                <p className="text-gray-300 text-sm">
                  Train the AI models using historical F1 data to make accurate predictions.
                </p>
                
                <button
                  onClick={handleTrain}
                  disabled={isTraining}
                  className="w-full bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-3 rounded-lg font-medium transition-colors flex items-center justify-center"
                >
                  {isTraining ? (
                    <>
                      <RotateCcw className="h-4 w-4 mr-2 animate-spin" />
                      Training Models...
                    </>
                  ) : (
                    <>
                      <Zap className="h-4 w-4 mr-2" />
                      {modelsTrained ? 'Retrain Models' : 'Train Models'}
                    </>
                  )}
                </button>
                
                {isTraining && (
                  <div className="text-sm text-yellow-400 bg-yellow-400/10 p-3 rounded-lg">
                    ⚠️ Training may take 5-10 minutes. Please wait...
                  </div>
                )}
              </div>
            </div>

            {/* Race Conditions */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-6 mt-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <MapPin className="h-5 w-5 mr-2 text-blue-500" />
                Race Conditions
              </h2>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium mb-2">Track</label>
                  <select
                    value={raceConditions.track_name}
                    onChange={(e) => handleTrackChange(e.target.value)}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
                  >
                    {tracks.map(track => (
                      <option key={track.name} value={track.name}>
                        {track.name} ({track.country})
                      </option>
                    ))}
                  </select>
                </div>

                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <label className="block text-sm font-medium mb-2">
                      <Thermometer className="h-4 w-4 inline mr-1" />
                      Track Temp (°C)
                    </label>
                    <input
                      type="number"
                      value={raceConditions.track_temp}
                      onChange={(e) => setRaceConditions(prev => ({...prev, track_temp: parseFloat(e.target.value)}))}
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
                    />
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium mb-2">Air Temp (°C)</label>
                    <input
                      type="number"
                      value={raceConditions.air_temp}
                      onChange={(e) => setRaceConditions(prev => ({...prev, air_temp: parseFloat(e.target.value)}))}
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-3 py-2 text-white"
                    />
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium mb-2">Humidity (%)</label>
                  <input
                    type="range"
                    min="0"
                    max="100"
                    value={raceConditions.humidity}
                    onChange={(e) => setRaceConditions(prev => ({...prev, humidity: parseInt(e.target.value)}))}
                    className="w-full"
                  />
                  <div className="text-center text-sm text-gray-400">{raceConditions.humidity}%</div>
                </div>

                <div className="flex items-center space-x-3">
                  <input
                    type="checkbox"
                    checked={raceConditions.rainfall}
                    onChange={(e) => setRaceConditions(prev => ({...prev, rainfall: e.target.checked}))}
                    className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded"
                  />
                  <label className="text-sm font-medium flex items-center">
                    <CloudRain className="h-4 w-4 mr-1 text-blue-400" />
                    Rain Expected
                  </label>
                </div>
              </div>
            </div>
          </div>

          {/* Drivers Selection */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-6">
              <div className="flex items-center justify-between mb-4">
                <h2 className="text-xl font-semibold flex items-center">
                  <Users className="h-5 w-5 mr-2 text-green-500" />
                  Drivers ({selectedDrivers.length})
                </h2>
                <button
                  onClick={addDriver}
                  disabled={selectedDrivers.length >= 10}
                  className="bg-green-600 hover:bg-green-700 disabled:bg-gray-600 px-3 py-1 rounded text-sm"
                >
                  + Add
                </button>
              </div>

              <div className="space-y-3 max-h-96 overflow-y-auto">
                {selectedDrivers.map((driver, index) => (
                  <div key={index} className="bg-gray-700/50 rounded-lg p-3">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-sm font-medium">Driver {index + 1}</span>
                      <button
                        onClick={() => removeDriver(index)}
                        className="text-red-400 hover:text-red-300 text-sm"
                      >
                        Remove
                      </button>
                    </div>
                    
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Driver</label>
                        <select
                          value={driver.driver_code}
                          onChange={(e) => updateDriver(index, 'driver_code', e.target.value)}
                          className="w-full bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm"
                        >
                          {drivers.map(code => (
                            <option key={code} value={code}>{code}</option>
                          ))}
                        </select>
                      </div>
                      
                      <div>
                        <label className="block text-xs text-gray-400 mb-1">Grid Pos</label>
                        <input
                          type="number"
                          min="1"
                          max="20"
                          value={driver.grid_position}
                          onChange={(e) => updateDriver(index, 'grid_position', parseInt(e.target.value))}
                          className="w-full bg-gray-600 border border-gray-500 rounded px-2 py-1 text-sm"
                        />
                      </div>
                    </div>
                  </div>
                ))}
              </div>

              <button
                onClick={handlePredict}
                disabled={isPredicting || !modelsTrained || selectedDrivers.length === 0}
                className="w-full mt-6 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed px-4 py-3 rounded-lg font-medium transition-colors flex items-center justify-center"
              >
                {isPredicting ? (
                  <>
                    <RotateCcw className="h-4 w-4 mr-2 animate-spin" />
                    Predicting...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4 mr-2" />
                    Predict Race
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Predictions Results */}
          <div className="lg:col-span-1">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700 p-6">
              <h2 className="text-xl font-semibold mb-4 flex items-center">
                <Trophy className="h-5 w-5 mr-2 text-yellow-500" />
                Race Predictions
              </h2>

              {predictions.length === 0 ? (
                <div className="text-center text-gray-400 py-8">
                  <Trophy className="h-12 w-12 mx-auto mb-3 opacity-50" />
                  <p>Run a prediction to see results</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {predictions.map((pred, index) => (
                    <div
                      key={pred.driver}
                      className={`p-4 rounded-lg border-l-4 ${getBorderColor(pred.predicted_position)} ${
                        pred.predicted_position <= 3 
                          ? 'bg-gradient-to-r from-gray-800 to-gray-900' 
                          : 'bg-gray-800/50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center space-x-3">
                          <div className={`w-10 h-10 rounded-full flex items-center justify-center text-lg font-bold ${getPodiumColor(pred.predicted_position)}`}>
                            {pred.predicted_position}
                          </div>
                          <span className="font-semibold text-lg">{pred.driver}</span>
                        </div>
                        <div className="text-right">
                          <div className="text-sm font-medium text-white">
                            {(pred.win_probability * 100).toFixed(1)}% win
                          </div>
                          <div className="text-xs text-gray-400">
                            {(pred.podium_probability * 100).toFixed(1)}% podium
                          </div>
                        </div>
                      </div>
                      
                      <div className="w-full bg-gray-700 rounded-full h-2">
                        <div
                          className={`h-2 rounded-full transition-all duration-500 ${
                            pred.predicted_position === 1 ? 'bg-yellow-500' :
                            pred.predicted_position === 2 ? 'bg-gray-400' :
                            pred.predicted_position === 3 ? 'bg-orange-500' :
                            'bg-gray-500'
                          }`}
                          style={{ width: `${pred.confidence * 100}%` }}
                        ></div>
                      </div>
                      <div className="text-xs text-gray-400 mt-1">
                        Confidence: {(pred.confidence * 100).toFixed(1)}%
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default F1Predictor;