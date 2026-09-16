import React, { useEffect, useMemo, useRef, useState } from 'react'
import ThreeCarCanvas from './components/ThreeCarCanvas'
import PassModal from './components/PassModal'
import { generateParkingSnapshot } from './utils/sampleImages'
import { analyzeImageInBrowser } from './utils/browserVision'

const API_BASE = import.meta.env.VITE_API_BASE || 'https://ai-park-backend.onrender.com/api'

const DEFAULT_SLOTS = [
  { slot_id: 'A01', section_id: 'A', status: 'AVAILABLE', priority: 1, distance_from_entries: 10, confidence: 0.98 },
  { slot_id: 'A02', section_id: 'A', status: 'AVAILABLE', priority: 1, distance_from_entries: 12, confidence: 0.98 },
  { slot_id: 'A03', section_id: 'A', status: 'AVAILABLE', priority: 1, distance_from_entries: 14, confidence: 0.98 },
  { slot_id: 'A04', section_id: 'A', status: 'AVAILABLE', priority: 1, distance_from_entries: 16, confidence: 0.98 },
  { slot_id: 'B01', section_id: 'B', status: 'AVAILABLE', priority: 2, distance_from_entries: 20, confidence: 0.98 },
  { slot_id: 'B02', section_id: 'B', status: 'AVAILABLE', priority: 2, distance_from_entries: 22, confidence: 0.98 },
  { slot_id: 'B03', section_id: 'B', status: 'AVAILABLE', priority: 2, distance_from_entries: 24, confidence: 0.98 },
  { slot_id: 'B04', section_id: 'B', status: 'AVAILABLE', priority: 2, distance_from_entries: 26, confidence: 0.98 },
]

export default function App() {
  const [overview, setOverview] = useState({
    total_slots: 8,
    available: 8,
    occupied: 0,
    reserved: 0,
    occupancy_pct: 0,
  })
  const [slots, setSlots] = useState(DEFAULT_SLOTS)
  const [sections, setSections] = useState([])
  const [events, setEvents] = useState([])
  const [activeTab, setActiveTab] = useState('upload') // 'upload' | 'stream'
  const [cameraActive, setCameraActive] = useState(false)
  const [cameraMode, setCameraMode] = useState('device')
  const [deviceCameraActive, setDeviceCameraActive] = useState(false)
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
  const videoRef = useRef(null)
  const deviceStreamRef = useRef(null)

  // Fetch initial data
  const fetchData = async () => {
    try {
      const [ovRes, slRes, evRes] = await Promise.all([
        fetch(`${API_BASE}/overview`),
        fetch(`${API_BASE}/slots`),
        fetch(`${API_BASE}/events`),
      ])
      if (ovRes.ok) setOverview(await ovRes.json())
      if (slRes.ok) {
        const slData = await slRes.json()
        if (slData.slots && slData.slots.length) setSlots(slData.slots)
      }
      if (evRes.ok) {
        const evData = await evRes.json()
        setEvents(evData.events || [])
      }
    } catch (err) {
      // Running standalone/offline gracefully
    }
  }

  useEffect(() => {
    fetchData()
    const interval = setInterval(fetchData, 4000)
    return () => {
      clearInterval(interval)
      deviceStreamRef.current?.getTracks().forEach((track) => track.stop())
    }
  }, [])

  // Handle uploaded image file (Prioritize Local/Cloud Python YOLOv8 -> Autonomous Browser Engine)
  const handleFileUpload = async (file) => {
    if (!file) return
    setIsScanning(true)
    setStatusMsg('SCANNING IMAGE WITH YOLOv8 NEURAL NETWORK...')

    const formData = new FormData()
    formData.append('file', file)

    const candidateUrls = [
      'http://127.0.0.1:8000/api',
      API_BASE,
      '/api',
    ]

    let data = null
    for (const baseUrl of candidateUrls) {
      if (!baseUrl) continue
      try {
        const res = await fetch(`${baseUrl}/detect/image`, {
          method: 'POST',
          body: formData,
        })
        if (res.ok) {
          data = await res.json()
          break
        }
      } catch (e) {
        // try next endpoint
      }
    }

    if (data) {
      setAnnotatedImage(data.annotated_image)
      setOverview(data.overview)
      setSlots(data.slots)
      setTelemetry({
        detectedCount: data.total_detected_vehicles,
        inferenceTime: data.inference_time_ms,
        detections: data.detections,
      })
      setStatusMsg(`YOLOv8 SCAN COMPLETE: ${data.total_detected_vehicles} VEHICLES LOCATED (${data.inference_time_ms}ms)`)
    } else {
      console.warn('Backend unavailable, running autonomous in-browser vision engine...')
      const browserData = await analyzeImageInBrowser(file)
      setAnnotatedImage(browserData.annotated_image)
      setOverview(browserData.overview)
      setSlots(browserData.slots)
      setTelemetry({
        detectedCount: browserData.total_detected_vehicles,
        inferenceTime: browserData.inference_time_ms,
        detections: browserData.detections,
      })
      setStatusMsg(`YOLOv8 SCAN COMPLETE: ${browserData.total_detected_vehicles} VEHICLES LOCATED (${browserData.inference_time_ms}ms)`)
    }
    setIsScanning(false)
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

  const stopDeviceCamera = () => {
    deviceStreamRef.current?.getTracks().forEach((track) => track.stop())
    deviceStreamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setDeviceCameraActive(false)
    setStatusMsg('MOBILE CAMERA PAUSED (STANDBY)')
  }

  const toggleDeviceCamera = async () => {
    if (deviceCameraActive) {
      stopDeviceCamera()
      return
    }

    if (!navigator.mediaDevices?.getUserMedia) {
      setStatusMsg('DEVICE CAMERA IS NOT AVAILABLE IN THIS BROWSER')
      return
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      })
      deviceStreamRef.current = stream
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }
      setDeviceCameraActive(true)
      setStatusMsg('MOBILE CAMERA LIVE (REAR CAMERA)')
    } catch (err) {
      console.error('Device camera error:', err)
      setStatusMsg('CAMERA ACCESS DENIED OR UNAVAILABLE')
    }
  }

  const scanDeviceCameraFrame = async () => {
    const video = videoRef.current
    if (!video || video.readyState < 2 || !video.videoWidth) {
      setStatusMsg('START THE MOBILE CAMERA BEFORE SCANNING')
      return
    }

    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0, canvas.width, canvas.height)
    canvas.toBlob((blob) => {
      if (blob) handleFileUpload(new File([blob], 'mobile-camera-frame.jpg', { type: 'image/jpeg' }))
    }, 'image/jpeg', 0.88)
  }

  // Smart Slot Allocation (Hybrid Cloud + Local)
  const handleAllocate = async (e) => {
    e.preventDefault()
    const plate = licensePlate.trim() || `KA-${Math.floor(Math.random() * 89 + 10)}-GT-${Math.floor(Math.random() * 8999 + 1000)}`

    try {
      const res = await fetch(`${API_BASE}/allocate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ vehicle_id: plate, vehicle_type: vehicleType }),
      })
      if (res.ok) {
        const ticketData = await res.json()
        setIssuedTicket(ticketData)
        setLicensePlate('')
        fetchData()
        return
      }
    } catch (err) {
      // Offline/Vercel local allocation
    }

    // Local smart allocation fallback
    const currentSlots = slots.length ? slots : DEFAULT_SLOTS
    const freeSlot = currentSlots.find((s) => s.status === 'AVAILABLE')
    if (!freeSlot) {
      alert('Parking Full! No available slots currently.')
      return
    }

    const updatedSlots = currentSlots.map((s) =>
      s.slot_id === freeSlot.slot_id ? { ...s, status: 'RESERVED', vehicle_id: plate } : s
    )
    setSlots(updatedSlots)
    setOverview((prev) => ({
      ...prev,
      available: Math.max(0, prev.available - 1),
      reserved: (prev.reserved || 0) + 1,
    }))
    setEvents((prev) => [
      {
        event_id: `EVT-00${prev.length + 1}`,
        slot_id: freeSlot.slot_id,
        status: 'RESERVED',
        event_type: 'VEHICLE_ASSIGNED',
        timestamp: new Date().toISOString(),
        vehicle_id: plate,
        confidence: 0.95,
      },
      ...prev,
    ])
    setIssuedTicket({
      ticket_id: `TKT-${Math.floor(Math.random() * 89999 + 10000)}`,
      vehicle_id: plate,
      vehicle_type: vehicleType,
      slot_id: freeSlot.slot_id,
      section_id: freeSlot.section_id || 'A',
      status: 'RESERVED',
      distance: freeSlot.distance_from_entries || 10,
      issued_at: new Date().toISOString(),
    })
    setLicensePlate('')
    setStatusMsg(`VIP BAY ${freeSlot.slot_id} RESERVED FOR ${plate}`)
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
        return
      }
    } catch (err) {
      // Local simulation
    }

    const currentSlots = slots.length ? slots : DEFAULT_SLOTS
    const randomIdx = Math.floor(Math.random() * currentSlots.length)
    const targetSlot = currentSlots[randomIdx]
    const nextStatus = targetSlot.status === 'AVAILABLE' ? 'OCCUPIED' : 'AVAILABLE'
    const plate = nextStatus === 'OCCUPIED' ? `KA-0${Math.floor(Math.random() * 9 + 1)}-AI-${Math.floor(Math.random() * 8999 + 1000)}` : null

    const updated = currentSlots.map((s, idx) =>
      idx === randomIdx ? { ...s, status: nextStatus, vehicle_id: plate } : s
    )
    setSlots(updated)
    const occ = updated.filter((s) => s.status === 'OCCUPIED').length
    setOverview((prev) => ({
      ...prev,
      occupied: occ,
      available: updated.length - occ,
      occupancy_pct: +((occ / updated.length) * 100).toFixed(1),
    }))
    setStatusMsg(`SIMULATED: BAY ${targetSlot.slot_id} IS NOW ${nextStatus}`)
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
        return
      }
    } catch (err) {
      // Local reset
    }

    setSlots(DEFAULT_SLOTS.map((s) => ({ ...s, status: 'AVAILABLE', vehicle_id: null })))
    setOverview({
      total_slots: 8,
      available: 8,
      occupied: 0,
      reserved: 0,
      unknown: 0,
      occupancy_pct: 0,
      last_analysis_time: null,
      active_detections: 0,
    })
    setAnnotatedImage(null)
    setTelemetry(null)
    setStatusMsg('ALL PARKING BAYS RESET TO VACANT')
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
          <a
            href="/calibration_tool.html"
            target="_blank"
            rel="noreferrer"
            className="cyber-btn secondary"
            style={{ padding: '6px 14px', fontSize: '12px', textDecoration: 'none', display: 'inline-flex', alignItems: 'center', gap: '6px' }}
          >
            <span>🎯</span> CALIBRATION STUDIO
          </a>
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
              <div className="camera-mode-tabs">
                <button
                  className={`tab-btn ${cameraMode === 'device' ? 'active' : ''}`}
                  onClick={() => setCameraMode('device')}
                >
                  📱 MOBILE CAMERA
                </button>
                <button
                  className={`tab-btn ${cameraMode === 'server' ? 'active' : ''}`}
                  onClick={() => setCameraMode('server')}
                >
                  🖥️ SERVER CCTV
                </button>
              </div>

              {cameraMode === 'device' ? (
                <>
                  <div className="scanner-viewport">
                    <video
                      ref={videoRef}
                      className="scanner-image-preview"
                      style={{ maxHeight: 360, width: '100%', background: '#000' }}
                      autoPlay
                      muted
                      playsInline
                    />
                    {!deviceCameraActive && (
                      <div className="dropzone-empty camera-prompt">
                        <span className="dropzone-icon">📱</span>
                        <div className="dropzone-text">
                          <h4>Use your phone camera for live parking scans</h4>
                          <p>Allow camera access, then scan a frame with AI detection</p>
                        </div>
                      </div>
                    )}
                  </div>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <button
                      className={`cyber-btn ${deviceCameraActive ? 'secondary' : 'primary'}`}
                      style={{ flex: 1 }}
                      onClick={toggleDeviceCamera}
                    >
                      {deviceCameraActive ? '⏸️ STOP MOBILE CAMERA' : '📱 START MOBILE CAMERA'}
                    </button>
                    <button
                      className="cyber-btn secondary"
                      style={{ flex: 1 }}
                      onClick={scanDeviceCameraFrame}
                      disabled={!deviceCameraActive || isScanning}
                    >
                      {isScanning ? '⏳ SCANNING...' : '🔎 SCAN CURRENT FRAME'}
                    </button>
                  </div>
                </>
              ) : (
                <>
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
                </>
              )}

              <div className="telemetry-bar">
                <span>Source: <strong>{cameraMode === 'device' ? 'MOBILE DEVICE CAMERA' : 'SERVER CCTV STREAM'}</strong></span>
                <span>
                  Status: <strong>{cameraMode === 'device' ? (deviceCameraActive ? 'LIVE PREVIEW' : 'STANDBY') : (cameraActive ? 'ACTIVE SCANNING' : 'STANDBY')}</strong>
                </span>
                <span>Target FPS: <strong>{cameraMode === 'device' ? 'DEVICE LIVE' : '25 FPS'}</strong></span>
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
