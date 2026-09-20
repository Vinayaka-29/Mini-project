import React, { useEffect, useMemo, useRef, useState } from 'react'
import ThreeCarCanvas from './components/ThreeCarCanvas'
import PassModal from './components/PassModal'
import { generateParkingSnapshot } from './utils/sampleImages'

const API_BASE = 'http://localhost:8000/api'

export default function App() {
  const [overview, setOverview] = useState({
    total_slots: 8,
    available: 8,
    occupied: 0,
    reserved: 0,
    occupancy_pct: 0,
  })
  const [slots, setSlots] = useState([])
  const [sections, setSections] = useState([])
  const [events, setEvents] = useState([])
  const [activeTab, setActiveTab] = useState('upload') // 'upload' | 'stream'
  const [cameraActive, setCameraActive] = useState(false)
  const [annotatedImage, setAnnotatedImage] = useState(null)
  const [isScanning, setIsScanning] = useState(false)
  const [telemetry, setTelemetry] = useState(null)
  const [sectionFilter, setSectionFilter] = useState('ALL')
  const [licensePlate, setLicensePlate] = useState('')
  const [vehicleType, setVehicleType] = useState('car')
  const [issuedTicket, setIssuedTicket] = useState(null)
  const [statusMsg, setStatusMsg] = useState('SYSTEM ONLINE • YOLOv8 READY')
  const [isCarBoosted, setIsCarBoosted] = useState(true)

  const fileInputRef = useRef(null)

  // Fetch initial data
  const fetchData = async () => {
    try {
      const [ovRes, slRes, evRes, secRes] = await Promise.all([
        fetch(`${API_BASE}/overview`),
        fetch(`${API_BASE}/slots`),
        fetch(`${API_BASE}/events`),
        fetch(`${API_BASE}/sections`),
      ])
      if (ovRes.ok) setOverview(await ovRes.json())
      if (slRes.ok) {
        const slData = await slRes.json()
        setSlots(slData.slots || [])
      }
      if (evRes.ok) {
        const evData = await evRes.json()
        setEvents(evData.events || [])
      }
      if (secRes.ok) {
        const sectionData = await secRes.json()
        setSections(sectionData.sections || [])
      }
    } catch (err) {
      console.warn('Backend polling error:', err)
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 4000)
    return () => clearInterval(interval)
  }, [])

  // Handle uploaded image file
  const handleFileUpload = async (file) => {
    if (!file) return
    setIsScanning(true)
    setStatusMsg('SCANNING IMAGE WITH YOLOv8 NEURAL NETWORK...')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/detect/auto`, {
        method: 'POST',
        body: formData,
      })
      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || 'Automatic calibration failed')
      }

      const data = await res.json()
      if (!data.bays?.length) {
        setAnnotatedImage(null)
        setSlots([])
        setOverview({ total_slots: 0, available: 0, occupied: 0, reserved: 0, occupancy_pct: 0 })
        setTelemetry({ detectedCount: data.vehicles?.length || 0, inferenceTime: null, detections: data.vehicles || [] })
        setStatusMsg(data.message || 'NO PARKING BAYS FOUND IN IMAGE')
        return
      }

      setAnnotatedImage(URL.createObjectURL(file))
      const calibratedSlots = data.bays.map((bay) => ({
        slot_id: bay.id,
        section_id: bay.id.match(/^[A-Z]+/)?.[0] || 'A',
        polygon: [[bay.box[0], bay.box[1]], [bay.box[2], bay.box[1]], [bay.box[2], bay.box[3]], [bay.box[0], bay.box[3]]],
        status: bay.status === 'occupied' ? 'OCCUPIED' : 'AVAILABLE',
        confidence: 1 - bay.overlap,
      }))
      setSlots(calibratedSlots)
      const occupiedCount = calibratedSlots.filter((slot) => slot.status === 'OCCUPIED').length
      setOverview({
        total_slots: calibratedSlots.length,
        available: calibratedSlots.length - occupiedCount,
        occupied: occupiedCount,
        reserved: 0,
        occupancy_pct: Math.round((occupiedCount / calibratedSlots.length) * 1000) / 10,
      })
      setTelemetry({
        detectedCount: data.vehicles?.length || 0,
        inferenceTime: null,
        detections: data.vehicles || [],
      })
      setStatusMsg(`AUTO CALIBRATION COMPLETE: ${calibratedSlots.length} BAYS, ${data.vehicles?.length || 0} VEHICLES LOCATED`)
    } catch (err) {
      console.error(err)
      setStatusMsg(`CALIBRATION ERROR: ${err.message}`)
    } finally {
      setIsScanning(false)
    }
  }

  // Quick Preset Sample Generator
  const runPresetSample = async (scenario) => {
    setIsScanning(true)
    const blob = await generateParkingSnapshot(scenario)
    const testFile = new File([blob], `sample_${scenario}.jpg`, { type: 'image/jpeg' })
    await handleFileUpload(testFile)
  }

  // Camera Toggle
  const toggleCamera = async () => {
    const nextState = !cameraActive
    try {
      const res = await fetch(`${API_BASE}/camera/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: nextState }),
      })
      if (res.ok) {
        setCameraActive(nextState)
        setStatusMsg(nextState ? 'LIVE CCTV CAMERA STREAM ACTIVE' : 'CCTV STREAM PAUSED (STANDBY)')
      }
    } catch (err) {
      console.error(err)
    }
  }

  // Smart Slot Allocation
  const handleAllocate = async (e) => {
    e.preventDefault()
    const plate = licensePlate.trim() || `KA-${Math.floor(Math.random() * 89 + 10)}-GT-${Math.floor(Math.random() * 8999 + 1000)}`

    try {
      const res = await fetch(`${API_BASE}/allocate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vehicle_id: plate, vehicle_type: vehicleType }),
      })
      if (!res.ok) {
        const errData = await res.json()
        alert(errData.detail || 'Parking is currently full!')
        return
      }
      const ticketData = await res.json()
      setIssuedTicket(ticketData)
      setLicensePlate('')
      fetchData()
    } catch (err) {
      alert('Allocation error: Ensure backend is running.')
    }
  }

  // Simulate Traffic Flow
  const simulateTraffic = async () => {
    try {
      const res = await fetch(`${API_BASE}/simulate/event`, { method: 'POST' })
      if (res.ok) {
        const ov = await res.json()
        setOverview(ov)
        fetchData()
        setStatusMsg('SIMULATED RANDOM VEHICLE MOVEMENT')
      }
    } catch (err) {
      console.error(err)
    }
  }

  // Reset All Slots
  const resetBays = async () => {
    try {
      const res = await fetch(`${API_BASE}/simulate/reset`, { method: 'POST' })
      if (res.ok) {
        setOverview(await res.json())
        setAnnotatedImage(null)
        setTelemetry(null)
        fetchData()
        setStatusMsg('ALL PARKING BAYS RESET TO VACANT')
      }
    } catch (err) {
      console.error(err)
    }
  }

  // Filtered Slots
  const filteredSlots = useMemo(() => {
    if (sectionFilter === 'ALL') return slots
    return slots.filter((s) => (s.section_id || 'A') === sectionFilter)
  }, [slots, sectionFilter])

  return (
    <div className="app-shell">
      {/* TOPBAR */}
      <header className="topbar">
        <div className="brand-wrapper">
          <div className="brand-icon-box">🏎️</div>
          <div>
            <span className="eyebrow">AUTONOMOUS VISION SYSTEM</span>
            <h1 className="app-title">AI-PARK INTELLIGENCE</h1>
          </div>
        </div>

        <div className="topbar-actions">
          <div className="status-pill">
            <span className="status-dot" />
            <span>{statusMsg}</span>
          </div>
        </div>
      </header>

      {/* HERO SECTION: 3D CYBER CAR + SMART ALLOCATION */}
      <section className="hero-section">
        {/* 3D Interactive Luxury Hypercar */}
        <div className="three-car-panel">
          <ThreeCarCanvas isRunning={isCarBoosted} carColor="#00f0ff" />
        </div>

        {/* Smart Parking Ticket Pass Allocation Panel */}
        <div className="action-hero-panel">
          <div>
            <div className="action-hero-title">
              <span>⚡</span>
              <h3>VIP SMART ALLOCATION</h3>
            </div>
            <p className="eyebrow" style={{ marginTop: 4 }}>
              Instant AI Nearest-Bay Reservation & Digital Pass
            </p>
          </div>

          <form className="allocation-form" onSubmit={handleAllocate}>
            <div className="cyber-input-group">
              <label>License Plate Number</label>
              <input
                type="text"
                className="cyber-input"
                placeholder="e.g. KA-05-MK-9999"
                value={licensePlate}
                onChange={(e) => setLicensePlate(e.target.value)}
              />
            </div>

            <div className="cyber-input-group">
              <label>Vehicle Classification</label>
              <select
                className="cyber-input"
                value={vehicleType}
                onChange={(e) => setVehicleType(e.target.value)}
              >
                <option value="car">Hypercar / Sedan</option>
                <option value="suv">Luxury SUV</option>
                <option value="ev">Electric Vehicle (EV)</option>
                <option value="bike">Superbike</option>
              </select>
            </div>

            <button type="submit" className="cyber-btn primary">
              RESERVE OPTIMAL BAY ➔
            </button>
          </form>

          <div className="quick-tools">
            <button className="cyber-btn secondary" onClick={simulateTraffic} style={{ flex: 1 }}>
              🎲 SIMULATE TRAFFIC
            </button>
            <button className="cyber-btn secondary" onClick={resetBays} style={{ flex: 1 }}>
              🔄 RESET ALL BAYS
            </button>
          </div>
        </div>
      </section>

      {/* STATS TELEMETRY GRID */}
      <section className="stats-grid">
        <div className="stat-card cyan">
          <span className="stat-label">Total Parking Bays</span>
          <strong className="stat-val">{overview.total_slots}</strong>
          <span className="stat-sub">High-Definition Monitored</span>
        </div>

        <div className="stat-card emerald">
          <span className="stat-label">Available Slots</span>
          <strong className="stat-val" style={{ color: 'var(--neon-emerald)' }}>
            {overview.available}
          </strong>
          <span className="stat-sub">Ready for immediate parking</span>
        </div>

        <div className="stat-card crimson">
          <span className="stat-label">Occupied Slots</span>
          <strong className="stat-val" style={{ color: 'var(--neon-crimson)' }}>
            {overview.occupied}
          </strong>
          <span className="stat-sub">Vehicles detected</span>
        </div>

        <div className="stat-card amber">
          <span className="stat-label">Reserved Passes</span>
          <strong className="stat-val" style={{ color: 'var(--neon-amber)' }}>
            {overview.reserved}
          </strong>
          <span className="stat-sub">VIP assigned</span>
        </div>

        <div className="stat-card cyan">
          <span className="stat-label">Occupancy Rate</span>
          <strong className="stat-val">{overview.occupancy_pct}%</strong>
          <span className="stat-sub">Real-time facility capacity</span>
        </div>
      </section>

      {/* DUAL WORKSPACE: VISION SCANNER / LIVE STREAM + 3D PARKING GRID */}
      <section className="workspace-grid">
        {/* LEFT PANEL: VISION ENGINE */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>👁️</span>
              <h3>YOLOv8 VISION INTELLIGENCE</h3>
            </div>
            <div className="tab-group">
              <button
                className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`}
                onClick={() => setActiveTab('upload')}
              >
                📸 IMAGE SCAN
              </button>
              <button
                className={`tab-btn ${activeTab === 'stream' ? 'active' : ''}`}
                onClick={() => setActiveTab('stream')}
              >
                📹 LIVE CCTV STREAM
              </button>
            </div>
          </div>

          {activeTab === 'upload' ? (
            <>
              {/* IMAGE UPLOAD & SCANNER VIEWPORT */}
              <div
                className="scanner-viewport"
                onClick={() => fileInputRef.current && fileInputRef.current.click()}
              >
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  capture="environment"
                  onChange={(e) => {
                    if (e.target.files && e.target.files[0]) {
                      handleFileUpload(e.target.files[0])
                    }
                  }}
                />

                {isScanning && <div className="scanner-beam" />}

                {annotatedImage ? (
                  <img
                    src={annotatedImage}
                    alt="YOLO Detection Preview"
                    className="scanner-image-preview"
                  />
                ) : (
                  <div className="dropzone-empty">
                    <span className="dropzone-icon">📷</span>
                    <div className="dropzone-text">
                      <h4>Drop Parking Photo or Tap to Snap Camera</h4>
                      <p>Supports JPG, PNG, Mobile Camera Snap, CCTV Stills</p>
                    </div>
                    <button
                      className="cyber-btn secondary"
                      onClick={(e) => {
                        e.stopPropagation()
                        fileInputRef.current && fileInputRef.current.click()
                      }}
                    >
                      SELECT LOCAL IMAGE
                    </button>
                  </div>
                )}
              </div>

              {/* QUICK DEMO PRESETS */}
              <div className="preset-strip">
                <span>⚡ Instant Demos:</span>
                <button
                  className="preset-chip"
                  onClick={() => runPresetSample('empty')}
                >
                  🟢 Empty Lot (All Free)
                </button>
                <button
                  className="preset-chip"
                  onClick={() => runPresetSample('half')}
                >
                  🟡 50% Occupied
                </button>
                <button
                  className="preset-chip"
                  onClick={() => runPresetSample('crowded')}
                >
                  🔴 High Density (Busy)
                </button>
              </div>

              {/* TELEMETRY */}
              {telemetry && (
                <div className="telemetry-bar">
                  <span>
                    Detected: <strong>{telemetry.detectedCount} Vehicles</strong>
                  </span>
                  <span>
                    Inference: <strong>{telemetry.inferenceTime} ms</strong>
                  </span>
                  <span>
                    Vision Model: <strong>YOLOv8-Nano</strong>
                  </span>
                </div>
              )}
            </>
          ) : (
            <>
              {/* LIVE CCTV STREAM VIEWPORT */}
              <div className="scanner-viewport">
                <img
                  src={`${API_BASE}/camera/feed`}
                  alt="Live CCTV Camera Feed"
                  className="scanner-image-preview"
                  style={{ maxHeight: 360, width: '100%', background: '#000' }}
                />
              </div>

              <div style={{ display: 'flex', gap: 10 }}>
                <button
                  className={`cyber-btn ${cameraActive ? 'secondary' : 'primary'}`}
                  style={{ flex: 1 }}
                  onClick={toggleCamera}
                >
                  {cameraActive ? '⏸️ PAUSE LIVE DETECTION' : '▶️ START LIVE SCANNING'}
                </button>
              </div>

              <div className="telemetry-bar">
                <span>
                  Source: <strong>WEBCAM / CCTV STREAM</strong>
                </span>
                <span>
                  Status: <strong>{cameraActive ? 'ACTIVE SCANNING' : 'STANDBY'}</strong>
                </span>
                <span>
                  Target FPS: <strong>25 FPS</strong>
                </span>
              </div>
            </>
          )}
        </div>

        {/* RIGHT PANEL: HOLOGRAPHIC 3D PARKING GRID */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>🅿️</span>
              <h3>FACILITY PARKING BAYS</h3>
            </div>
            <div className="tab-group">
              <button
                className={`tab-btn ${sectionFilter === 'ALL' ? 'active' : ''}`}
                onClick={() => setSectionFilter('ALL')}
              >
                ALL
              </button>
              <button
                className={`tab-btn ${sectionFilter === 'A' ? 'active' : ''}`}
                onClick={() => setSectionFilter('A')}
              >
                SEC A
              </button>
              <button
                className={`tab-btn ${sectionFilter === 'B' ? 'active' : ''}`}
                onClick={() => setSectionFilter('B')}
              >
                SEC B
              </button>
            </div>
          </div>

          <div className="parking-grid">
            {filteredSlots.map((slot) => {
              const statusClass = slot.status.toLowerCase()
              return (
                <div
                  key={slot.slot_id}
                  className={`slot-card ${statusClass}`}
                  onClick={() => {
                    if (slot.status === 'AVAILABLE') {
                      setLicensePlate(`VIP-${slot.slot_id}`)
                    }
                  }}
                  title={slot.status === 'AVAILABLE' ? 'Click to Reserve this slot' : 'Occupied'}
                >
                  <div className="slot-header">
                    <span className="slot-id-badge">{slot.slot_id}</span>
                    <span className="slot-badge">{slot.status}</span>
                  </div>

                  <div className="slot-visual-center">
                    {slot.status === 'OCCUPIED' ? (
                      <span role="img" aria-label="Car" style={{ filter: 'drop-shadow(0 0 10px rgba(255,0,85,0.6))' }}>
                        🏎️
                      </span>
                    ) : slot.status === 'RESERVED' ? (
                      <span role="img" aria-label="Reserved">
                        🎫
                      </span>
                    ) : (
                      <span role="img" aria-label="Available" style={{ filter: 'drop-shadow(0 0 10px rgba(0,255,157,0.5))' }}>
                        🅿️
                      </span>
                    )}
                  </div>

                  <div className="slot-meta">
                    <span>{slot.vehicle_id || (slot.status === 'AVAILABLE' ? 'TAP TO RESERVE' : 'IN USE')}</span>
                    <span>{slot.confidence ? `${Math.round(slot.confidence * 100)}% CONF` : '100%'}</span>
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </section>

      {/* BOTTOM SECTION: RECENT EVENT STREAM & SYSTEM HEALTH */}
      <section className="bottom-grid">
        {/* RECENT EVENTS */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>📡</span>
              <h3>REAL-TIME AUDIT LOG & EVENTS</h3>
            </div>
            <span className="hud-badge">{events.length} LOGGED</span>
          </div>

          <ul className="event-list">
            {events.length ? (
              events
                .slice()
                .reverse()
                .map((ev, idx) => (
                  <li key={ev.event_id || idx} className="event-item">
                    <span className="event-time">
                      {ev.timestamp ? ev.timestamp.slice(11, 19) : 'LIVE'}
                    </span>
                    <span className="event-slot">{ev.slot_id}</span>
                    <span className={`event-status ${ev.status?.toLowerCase()}`}>
                      {ev.event_type || ev.status}
                    </span>
                    <span style={{ color: 'var(--text-muted)' }}>
                      {ev.vehicle_id ? `[${ev.vehicle_id}]` : `${Math.round((ev.confidence || 0.95) * 100)}%`}
                    </span>
                  </li>
                ))
            ) : (
              <li className="event-item">No security events logged yet</li>
            )}
          </ul>
        </div>

        {/* SYSTEM HEALTH */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>⚙️</span>
              <h3>SYSTEM TELEMETRY</h3>
            </div>
            <span className="status-pill">100% OPERATIONAL</span>
          </div>

          <div className="health-grid">
            <div className="health-item">
              <label>AI VISION ENGINE</label>
              <strong>YOLOv8 NANO (COCO)</strong>
            </div>
            <div className="health-item">
              <label>OCCUPANCY ENGINE</label>
              <strong>SPATIAL POLYGON IoU</strong>
            </div>
            <div className="health-item">
              <label>INFERENCE DEVICE</label>
              <strong>LOCAL CPU (OPTIMIZED)</strong>
            </div>
            <div className="health-item">
              <label>LATENCY</label>
              <strong style={{ color: 'var(--neon-cyan)' }}>REAL-TIME (&lt;45ms)</strong>
            </div>
          </div>
          <div className="section-list">
            {sections.map((section) => <div key={section.section_id}><label>Section {section.section_id}</label><span>{section.available}/{section.total} available</span></div>)}
          </div>
        </div>
      </section>

      {/* VIP TICKET MODAL */}
      <PassModal ticket={issuedTicket} onClose={() => setIssuedTicket(null)} />
    </div>
  )
}
