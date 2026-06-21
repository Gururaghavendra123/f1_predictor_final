import React, { useState, useEffect, useMemo } from 'react';
import {
  Flag, Gauge, Thermometer, Droplets, CloudRain, Radio, ChevronDown,
  Plus, X, Activity, Cpu, AlertTriangle, Wind,
} from 'lucide-react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const PREFERRED = ['VER', 'NOR', 'LEC', 'PIA', 'RUS', 'HAM', 'ANT', 'ALO'];

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

export default function App() {
  const [modelInfo, setModelInfo] = useState(null);
  const [drivers, setDrivers] = useState([]);
  const [tracks, setTracks] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [isPredicting, setIsPredicting] = useState(false);
  const [error, setError] = useState('');

  const [conditions, setConditions] = useState({
    track_name: '', country: '', track_temp: 30, air_temp: 24,
    humidity: 55, rainfall: false, track_type: 'normal', era: 1,
  });
  const [grid, setGrid] = useState([]);

  // ---- boot ----
  useEffect(() => {
    (async () => {
      try {
        const [mi, dr, tk] = await Promise.all([
          fetch(`${API_BASE}/model-info`).then((r) => r.json()).catch(() => null),
          fetch(`${API_BASE}/drivers`).then((r) => r.json()).catch(() => ({ drivers: [] })),
          fetch(`${API_BASE}/tracks`).then((r) => r.json()).catch(() => ({ tracks: [] })),
        ]);
        setModelInfo(mi);
        const dlist = dr?.drivers || [];
        const tlist = tk?.tracks || [];
        setDrivers(dlist);
        setTracks(tlist);

        const def = tlist.find((t) => /Monaco/i.test(t.name)) || tlist[0];
        if (def) setConditions((c) => ({ ...c, track_name: def.name, country: def.country, track_type: def.type }));

        const picks = PREFERRED.filter((d) => dlist.includes(d)).slice(0, 8);
        const seed = (picks.length ? picks : dlist.slice(0, 8));
        setGrid(seed.map((code, i) => ({ driver_code: code, grid_position: i + 1 })));
      } catch (e) {
        setError('Could not reach the API. Is the backend running on :8000?');
      }
    })();
  }, []);

  const ready = modelInfo?.status === 'trained';

  // ---- handlers ----
  const onTrack = (name) => {
    const t = tracks.find((x) => x.name === name);
    if (t) setConditions((c) => ({ ...c, track_name: t.name, country: t.country, track_type: t.type }));
  };
  const setCond = (k, v) => setConditions((c) => ({ ...c, [k]: v }));
  const setRow = (i, k, v) => setGrid((g) => g.map((r, idx) => (idx === i ? { ...r, [k]: v } : r)));
  const addRow = () => setGrid((g) => (g.length < 20 ? [...g, { driver_code: drivers.find((d) => !g.some((r) => r.driver_code === d)) || drivers[0], grid_position: g.length + 1 }] : g));
  const delRow = (i) => setGrid((g) => g.filter((_, idx) => idx !== i));

  const predict = async () => {
    if (!ready) { setError('Model not loaded. Run `python train.py`, then restart the API.'); return; }
    if (!grid.length) { setError('Add at least one driver to the grid.'); return; }
    setError('');
    setIsPredicting(true);
    setPredictions([]);
    try {
      const body = { race_conditions: conditions, drivers: grid };
      const [res] = await Promise.all([
        fetch(`${API_BASE}/predict`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        }),
        sleep(1500), // let the lights sequence breathe
      ]);
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      data.sort((a, b) => a.predicted_position - b.predicted_position);
      setPredictions(data);
    } catch (e) {
      setError(e.message || 'Prediction failed.');
    } finally {
      setIsPredicting(false);
    }
  };

  const podium = useMemo(() => predictions.filter((p) => p.predicted_position <= 3), [predictions]);
  const rest = useMemo(() => predictions.filter((p) => p.predicted_position > 3), [predictions]);
  const maxWin = useMemo(() => Math.max(0.01, ...predictions.map((p) => p.win_probability)), [predictions]);

  const humidityFill = `${conditions.humidity}%`;

  return (
    <div className="oracle">
      {/* ---------------- TOP BAR ---------------- */}
      <header className="topbar reveal">
        <div className="brand">
          <div className="brand-mark"><Flag size={24} strokeWidth={2.5} /></div>
          <div>
            <h1>Ghost <em>Lap</em></h1>
            <div className="sub">Predicting F1 race outcomes with machine learning and real data</div>
          </div>
        </div>
        <div className={`status-pill ${ready ? 'live' : ''}`}>
          <span className="dot" />
          {ready ? 'Model Online' : 'Model Offline'}
          {ready && modelInfo?.trained_through && (
            <span className="meta">· thru {modelInfo.trained_through}</span>
          )}
        </div>
      </header>

      <div className="stage">
        {/* ---------------- COLUMN 1: CIRCUIT + CONDITIONS ---------------- */}
        <section className="col-setup">
          <div className="panel reveal" style={{ animationDelay: '0.05s' }}>
            <div className="panel-head">
              <span className="panel-num">01</span>
              <h2 className="panel-title">Circuit</h2>
              <Radio className="panel-icon" size={16} />
            </div>
            <div className="panel-body">
              <div className="field-label"><Flag size={11} /> Grand Prix</div>
              <div className="select-wrap">
                <select className="input" value={conditions.track_name} onChange={(e) => onTrack(e.target.value)}>
                  {tracks.map((t) => <option key={t.name} value={t.name}>{t.name}</option>)}
                </select>
                <ChevronDown className="chev" size={16} />
              </div>
              <div className="circuit-flag">
                {conditions.country || '—'}
                {conditions.track_type && conditions.track_type !== 'normal' && (
                  <span className="tag">{conditions.track_type}</span>
                )}
              </div>

              <div className="cond-grid">
                <div className="cond">
                  <div className="k"><Thermometer size={11} /> Track</div>
                  <div className="v">{conditions.track_temp}<small>°C</small></div>
                  <input type="number" value={conditions.track_temp}
                    onChange={(e) => setCond('track_temp', parseFloat(e.target.value) || 0)} />
                </div>
                <div className="cond">
                  <div className="k"><Wind size={11} /> Air</div>
                  <div className="v">{conditions.air_temp}<small>°C</small></div>
                  <input type="number" value={conditions.air_temp}
                    onChange={(e) => setCond('air_temp', parseFloat(e.target.value) || 0)} />
                </div>
              </div>

              <div className="slider-row">
                <div className="slider-top">
                  <span className="field-label" style={{ margin: 0 }}><Droplets size={11} /> Humidity</span>
                  <span className="val">{conditions.humidity}%</span>
                </div>
                <input type="range" min="0" max="100" value={conditions.humidity}
                  style={{ '--fill': humidityFill }}
                  onChange={(e) => setCond('humidity', parseInt(e.target.value, 10))} />
              </div>

              <div className="toggle-row">
                <div className={`toggle ${conditions.rainfall ? 'on' : ''}`}
                  onClick={() => setCond('rainfall', !conditions.rainfall)}>
                  <CloudRain className="tg-icon" size={18} />
                  <div className="tg-text"><b>{conditions.rainfall ? 'Wet' : 'Dry'}</b><span>Conditions</span></div>
                </div>
                <div className={`toggle era ${conditions.era === 1 ? 'on' : ''}`}
                  onClick={() => setCond('era', conditions.era === 1 ? 0 : 1)}>
                  <Cpu className="tg-icon" size={18} />
                  <div className="tg-text"><b>{conditions.era === 1 ? '2026+' : '2022–25'}</b><span>Rules Era</span></div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ---------------- COLUMN 2: STARTING GRID ---------------- */}
        <section className="col-grid">
          <div className="panel reveal" style={{ animationDelay: '0.12s' }}>
            <div className="panel-head">
              <span className="panel-num">02</span>
              <h2 className="panel-title">Starting Grid</h2>
              <Gauge className="panel-icon" size={16} />
            </div>
            <div className="panel-body">
              <div className="grid-meta">
                <span className="grid-count">{grid.length} / 20 cars</span>
                <button className="add-btn" onClick={addRow} disabled={grid.length >= 20 || !drivers.length}>
                  <Plus size={13} /> Add Car
                </button>
              </div>

              <div className="grid-list">
                {grid.map((row, i) => (
                  <div className="grid-row" key={i}>
                    <div className="grid-slot">{row.grid_position}</div>
                    <div className="select-wrap">
                      <select className="drv" value={row.driver_code}
                        onChange={(e) => setRow(i, 'driver_code', e.target.value)}>
                        {drivers.map((d) => <option key={d} value={d}>{d}</option>)}
                      </select>
                    </div>
                    <input className="gp" type="number" min="1" max="20" value={row.grid_position}
                      onChange={(e) => setRow(i, 'grid_position', parseInt(e.target.value, 10) || 1)} />
                    <button className="row-x" onClick={() => delRow(i)} aria-label="remove"><X size={15} /></button>
                  </div>
                ))}
              </div>

              <button className={`launch ${isPredicting ? 'armed' : ''}`} onClick={predict}
                disabled={isPredicting || !ready}>
                <div className="lights">
                  {[0, 1, 2, 3, 4].map((n) => <span className="light" key={n} />)}
                </div>
                <div className="launch-label">
                  <Activity size={18} />
                  {isPredicting ? 'Lights Out…' : 'Run Prediction'}
                </div>
                <div className="launch-hint">
                  {ready ? 'Simulate finishing order' : 'Awaiting model'}
                </div>
              </button>

              {error && (
                <div className="banner"><AlertTriangle size={14} /> {error}</div>
              )}
            </div>
          </div>
        </section>

        {/* ---------------- COLUMN 3: RESULTS ---------------- */}
        <section className="col-results">
          <div className="panel reveal" style={{ animationDelay: '0.19s' }}>
            <div className="panel-head">
              <span className="panel-num">03</span>
              <h2 className="panel-title">Predicted Classification</h2>
              <Flag className="panel-icon" size={16} />
            </div>
            <div className="panel-body">
              {predictions.length === 0 ? (
                <div className="empty">
                  <div className="big">P?</div>
                  <p>Run a prediction to see the grid</p>
                </div>
              ) : (
                <>
                  {podium.length > 0 && (
                    <div className="podium">
                      {podium.map((p, i) => (
                        <div className={`pod pod--${p.predicted_position}`} key={p.driver}
                          style={{ animationDelay: `${0.1 + i * 0.12}s` }}>
                          <div className="cap">
                            <span className="medal">{p.predicted_position === 1 ? '🏆' : p.predicted_position === 2 ? '🥈' : '🥉'}</span>
                            <div className="drv">{p.driver}</div>
                            <div className="wl">{(p.win_probability * 100).toFixed(0)}% win</div>
                          </div>
                          <div className="pos">{p.predicted_position}</div>
                        </div>
                      ))}
                    </div>
                  )}

                  {rest.length > 0 && (
                    <div className="tower">
                      {rest.map((p, i) => (
                        <div className={`tower-row p${p.predicted_position}`} key={p.driver}
                          style={{ animationDelay: `${0.3 + i * 0.06}s` }}>
                          <div className="tr-pos">{p.predicted_position}</div>
                          <div className="tr-main">
                            <div className="tr-name">{p.driver}</div>
                            <div className="tr-bar">
                              <i style={{ '--w': `${(p.win_probability / maxWin) * 100}%` }} />
                            </div>
                          </div>
                          <div className="tr-stats">
                            <div className="tr-win">{(p.win_probability * 100).toFixed(0)}<small>% WIN</small></div>
                            <div className="tr-sub">EXP P{p.expected_position?.toFixed?.(1)} · {(p.podium_probability * 100).toFixed(0)}% POD</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                  <div className="res-note">
                    <span>MODEL · {modelInfo?.model_type || 'position_ranking'}</span>
                    <span>{modelInfo?.features_count || 25} FEATURES</span>
                  </div>
                </>
              )}
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
