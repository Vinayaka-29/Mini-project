import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import ThreeCarCanvas from './components/ThreeCarCanvas'
import PassModal from './components/PassModal'

// ── API configuration ────────────────────────────────────────────────────────
const API_BASE =
  import.meta.env.VITE_API_BASE ||
  (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1'
    ? 'http://127.0.0.1:8000/api'
    : 'https://ai-park-backend.onrender.com/api')

const WS_BASE = API_BASE.replace(/^https?/, (p) => (p === 'https' ? 'wss' : 'ws'))
  .replace('/api', '')

const COLD_START_TIMEOUT_MS = 60000  // Render free tier can take ~40-60s
const RETRY_INTERVAL_MS = 5000

// ── App ──────────────────────────────────────────────────────────────────────
export default function App() {
  // State
  const [overview, setOverview] = useState(null)
  const [slots, setSlots] = useState([])
  const [sections, setSections] = useState([])
  const [events, setEvents] = useState([])

  const [backendOnline, setBackendOnline] = useState(null)   // null=checking, true, false
  const [wakingUp, setWakingUp] = useState(false)
  const [wakeCountdown, setWakeCountdown] = useState(0)
  const [bgCalibrated, setBgCalibrated] = useState(false)
  const [detectionMode, setDetectionMode] = useState('background_subtraction')

  const [activeTab, setActiveTab] = useState('upload')
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
  const [statusMsg, setStatusMsg] = useState('CHECKING BACKEND…')
  const [isCarBoosted] = useState(true)

  const fileInputRef = useRef(null)
  const refFileInputRef = useRef(null)
  const videoRef = useRef(null)
  const deviceStreamRef = useRef(null)
  const wsRef = useRef(null)
  const retryTimerRef = useRef(null)
  const wakeTimerRef = useRef(null)
  const wakeStartRef = useRef(null)

  // ── Backend health check & cold-start handling ──────────────────────────
  const checkHealth = useCallback(async () => {
    try {
      const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(8000) })
      if (res.ok) {
        const data = await res.json()
        setBackendOnline(true)
        setWakingUp(false)
        setWakeCountdown(0)
        if (wakeTimerRef.current) clearInterval(wakeTimerRef.current)
        setBgCalibrated(data.bg_calibrated ?? false)
        setDetectionMode(data.detection_mode ?? 'background_subtraction')
        setStatusMsg(
          data.bg_calibrated
            ? 'SYSTEM ONLINE • BG-SUBTRACTION READY'
            : 'ONLINE • ⚠️ UPLOAD REFERENCE FRAME FIRST'
        )
        return true
      }
    } catch {
      // Fall through
    }
    return false
  }, [])

  const startWakeSequence = useCallback(() => {
    setWakingUp(true)
    setBackendOnline(false)
    setStatusMsg('⏳ WAKING BACKEND SERVER (~40s)…')
    wakeStartRef.current = Date.now()
    setWakeCountdown(Math.round(COLD_START_TIMEOUT_MS / 1000))

    // Countdown ticker
    wakeTimerRef.current = setInterval(() => {
      const elapsed = Date.now() - wakeStartRef.current
      const remaining = Math.max(0, Math.round((COLD_START_TIMEOUT_MS - elapsed) / 1000))
      setWakeCountdown(remaining)
    }, 1000)

    // Retry health check every 5s
    retryTimerRef.current = setInterval(async () => {
      const ok = await checkHealth()
      if (ok) {
        clearInterval(retryTimerRef.current)
        clearInterval(wakeTimerRef.current)
        fetchData()
      } else if (Date.now() - wakeStartRef.current > COLD_START_TIMEOUT_MS) {
        clearInterval(retryTimerRef.current)
        clearInterval(wakeTimerRef.current)
        setWakingUp(false)
        setStatusMsg('❌ BACKEND UNREACHABLE — CHECK SERVER')
      }
    }, RETRY_INTERVAL_MS)
  }, [checkHealth])

  // ── Data fetching ────────────────────────────────────────────────────────
  const fetchData = useCallback(async () => {
    try {
      const [ovRes, slRes, evRes, secRes] = await Promise.all([
        fetch(`${API_BASE}/overview`),
        fetch(`${API_BASE}/slots`),
        fetch(`${API_BASE}/events`),
        fetch(`${API_BASE}/sections`),
      ])
      if (ovRes.ok) setOverview(await ovRes.json())
      if (slRes.ok) {
        const d = await slRes.json()
        if (d.slots) setSlots(d.slots)
      }
      if (evRes.ok) {
        const d = await evRes.json()
        setEvents(d.events || [])
      }
      if (secRes.ok) {
        const d = await secRes.json()
        setSections(d.sections || [])
      }
    } catch {
      // WS handles live updates; polling is a fallback
    }
  }, [])

  // ── WebSocket subscription ───────────────────────────────────────────────
  const connectWS = useCallback(() => {
    if (!backendOnline) return
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(`${WS_BASE}/ws`)
      wsRef.current = ws

      ws.onopen = () => {
        // Keep-alive ping every 30s
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send('ping')
        }, 30000)
        ws._pingInterval = pingInterval
      }

      ws.onmessage = (e) => {
        try {
          const msg = JSON.parse(e.data)
          if (msg.type === 'STATE_UPDATE') {
            if (msg.overview) setOverview(msg.overview)
            if (msg.slots) setSlots(msg.slots)
            if (msg.events) setEvents(msg.events)
          }
        } catch {}
      }

      ws.onclose = () => {
        clearInterval(ws._pingInterval)
        // Reconnect after 3s if backend is still online
        setTimeout(() => {
          if (backendOnline) connectWS()
        }, 3000)
      }

      ws.onerror = () => ws.close()
    } catch {}
  }, [backendOnline])

  // ── Startup sequence ─────────────────────────────────────────────────────
  useEffect(() => {
    const init = async () => {
      const ok = await checkHealth()
      if (ok) {
        await fetchData()
      } else {
        startWakeSequence()
      }
    }
    init()

    return () => {
      clearInterval(retryTimerRef.current)
      clearInterval(wakeTimerRef.current)
      wsRef.current?.close()
      deviceStreamRef.current?.getTracks().forEach((t) => t.stop())
    }
  }, [])

  useEffect(() => {
    if (backendOnline) {
      connectWS()
      // Fallback polling every 10s (WS is primary)
      const pollInterval = setInterval(fetchData, 10000)
      return () => clearInterval(pollInterval)
    }
  }, [backendOnline, connectWS, fetchData])

  // ── Image upload / detection ─────────────────────────────────────────────
  const handleFileUpload = async (file) => {
    if (!file || !backendOnline) return
    setIsScanning(true)
    setStatusMsg('SCANNING IMAGE…')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/detect/image`, { method: 'POST', body: formData })
      if (res.ok) {
        const data = await res.json()
        setAnnotatedImage(data.annotated_image || null)
        if (data.overview) setOverview(data.overview)
        if (data.slots) setSlots(data.slots)
        setTelemetry({
          detectedCount: data.total_detected_vehicles,
          inferenceTime: data.inference_time_ms,
          detections: data.detections || [],
          mode: data.detection_mode,
        })
        setStatusMsg(
          `SCAN COMPLETE: ${data.total_detected_vehicles} OCCUPIED BAYS DETECTED (${data.inference_time_ms}ms)`
        )
      } else {
        const err = await res.json()
        setStatusMsg(`SCAN ERROR: ${err.detail || 'Unknown error'}`)
      }
    } catch (err) {
      setStatusMsg(`NETWORK ERROR: ${err.message}`)
    }

    setIsScanning(false)
  }

  // ── Reference frame upload ───────────────────────────────────────────────
  const handleReferenceUpload = async (file) => {
    if (!file || !backendOnline) return
    setStatusMsg('UPLOADING REFERENCE FRAME…')

    const formData = new FormData()
    formData.append('file', file)

    try {
      const res = await fetch(`${API_BASE}/reference-frame`, { method: 'POST', body: formData })
      if (res.ok) {
        const data = await res.json()
        setBgCalibrated(true)
        setStatusMsg('✅ REFERENCE FRAME SET — BG SUBTRACTION CALIBRATED')
      } else {
        const err = await res.json()
        setStatusMsg(`REFERENCE ERROR: ${err.detail}`)
      }
    } catch (err) {
      setStatusMsg(`NETWORK ERROR: ${err.message}`)
    }
  }

  // ── Sample photo presets (real photos from /public/) ─────────────────────
  const runSamplePhoto = async (filename) => {
    if (!backendOnline) return
    setIsScanning(true)
    setStatusMsg(`LOADING ${filename}…`)
    try {
      const res = await fetch(`/${filename}`)
      if (!res.ok) throw new Error(`Could not fetch /${filename}`)
      const blob = await res.blob()
      const file = new File([blob], filename, { type: blob.type || 'image/jpeg' })
      await handleFileUpload(file)
    } catch (err) {
      setStatusMsg(`Could not load sample: ${err.message}`)
      setIsScanning(false)
    }
  }

  // ── Camera controls ──────────────────────────────────────────────────────
  const toggleCamera = async () => {
    if (!backendOnline) return
    const next = !cameraActive
    try {
      const res = await fetch(`${API_BASE}/camera/toggle`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ active: next }),
      })
      if (res.ok) {
        setCameraActive(next)
        setStatusMsg(next ? 'LIVE CCTV STREAM ACTIVE' : 'CCTV STREAM PAUSED')
      }
    } catch {}
  }

  const stopDeviceCamera = () => {
    deviceStreamRef.current?.getTracks().forEach((t) => t.stop())
    deviceStreamRef.current = null
    if (videoRef.current) videoRef.current.srcObject = null
    setDeviceCameraActive(false)
    setStatusMsg('MOBILE CAMERA PAUSED')
  }

  const toggleDeviceCamera = async () => {
    if (deviceCameraActive) { stopDeviceCamera(); return }
    if (!navigator.mediaDevices?.getUserMedia) {
      setStatusMsg('DEVICE CAMERA NOT AVAILABLE IN THIS BROWSER')
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
      setStatusMsg('MOBILE CAMERA LIVE (REAR)')
    } catch (err) {
      setStatusMsg('CAMERA ACCESS DENIED')
    }
  }

  const scanDeviceCameraFrame = async () => {
    const video = videoRef.current
    if (!video || video.readyState < 2 || !video.videoWidth) {
      setStatusMsg('START MOBILE CAMERA BEFORE SCANNING')
      return
    }
    const canvas = document.createElement('canvas')
    canvas.width = video.videoWidth
    canvas.height = video.videoHeight
    canvas.getContext('2d').drawImage(video, 0, 0)
    canvas.toBlob((blob) => {
      if (blob) handleFileUpload(new File([blob], 'camera-frame.jpg', { type: 'image/jpeg' }))
    }, 'image/jpeg', 0.90)
  }

  // ── Slot allocation ──────────────────────────────────────────────────────
  const handleAllocate = async (e) => {
    e.preventDefault()
    if (!backendOnline) { setStatusMsg('BACKEND OFFLINE — CANNOT ALLOCATE'); return }

    const plate =
      licensePlate.trim() ||
      `KA-${Math.floor(Math.random() * 89 + 10)}-GT-${Math.floor(Math.random() * 8999 + 1000)}`

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
        setStatusMsg(`BAY ${ticketData.slot_id} RESERVED FOR ${plate}`)
      } else {
        const err = await res.json()
        setStatusMsg(`ALLOCATION FAILED: ${err.detail}`)
      }
    } catch (err) {
      setStatusMsg(`NETWORK ERROR: ${err.message}`)
    }
  }

  // ── Simulate / reset (backend only) ─────────────────────────────────────
  const simulateTraffic = async () => {
    if (!backendOnline) return
    try {
      const res = await fetch(`${API_BASE}/simulate/event`, { method: 'POST' })
      if (res.ok) { fetchData(); setStatusMsg('SIMULATED RANDOM VEHICLE MOVEMENT') }
    } catch {}
  }

  const resetBays = async () => {
    if (!backendOnline) return
    try {
      const res = await fetch(`${API_BASE}/simulate/reset`, { method: 'POST' })
      if (res.ok) { fetchData(); setAnnotatedImage(null); setTelemetry(null); setStatusMsg('ALL BAYS RESET TO VACANT') }
    } catch {}
  }

  // ── Derived values ───────────────────────────────────────────────────────
  const filteredSlots = useMemo(() => {
    if (sectionFilter === 'ALL') return slots
    return slots.filter((s) => (s.section_id || 'A') === sectionFilter)
  }, [slots, sectionFilter])

  const uniqueSections = useMemo(
    () => [...new Set(slots.map((s) => s.section_id || 'A'))].sort(),
    [slots]
  )

  const displayOverview = overview || { total_slots: 0, available: 0, occupied: 0, reserved: 0, occupancy_pct: 0 }

  // ── Offline banner ───────────────────────────────────────────────────────
  const renderOfflineBanner = () => {
    if (backendOnline === true) return null
    return (
      <div className="offline-banner" role="alert">
        {wakingUp ? (
          <>
            <span className="offline-icon">⏳</span>
            <div>
              <strong>WAKING BACKEND SERVER</strong>
              <p>Render free tier is spinning up… {wakeCountdown > 0 ? `~${wakeCountdown}s remaining` : 'almost there'}</p>
            </div>
          </>
        ) : backendOnline === false && !wakingUp ? (
          <>
            <span className="offline-icon">⚠️</span>
            <div>
              <strong>BACKEND UNREACHABLE</strong>
              <p>All detection features are disabled. Check the server or try again.</p>
            </div>
            <button className="cyber-btn secondary" onClick={() => { setBackendOnline(null); startWakeSequence() }}>
              RETRY
            </button>
          </>
        ) : (
          <>
            <span className="offline-icon">🔄</span>
            <strong>CONNECTING TO BACKEND…</strong>
          </>
        )}
      </div>
    )
  }

  // ── BG calibration banner ────────────────────────────────────────────────
  const renderCalibrationBanner = () => {
    if (!backendOnline || bgCalibrated || detectionMode !== 'background_subtraction') return null
    return (
      <div className="calibration-banner" role="status">
        <span>📸</span>
        <div>
          <strong>BACKGROUND SUBTRACTION NEEDS CALIBRATION</strong>
          <p>Upload an empty-lot reference photo (no cars) before scanning.</p>
        </div>
        <button className="cyber-btn secondary" onClick={() => refFileInputRef.current?.click()}>
          UPLOAD REFERENCE
        </button>
        <input
          type="file"
          ref={refFileInputRef}
          style={{ display: 'none' }}
          accept="image/*"
          onChange={(e) => { if (e.target.files?.[0]) handleReferenceUpload(e.target.files[0]) }}
        />
      </div>
    )
  }

  // ── Render ───────────────────────────────────────────────────────────────
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
          <div className={`status-pill ${backendOnline ? '' : 'offline'}`}>
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

      {/* OFFLINE BANNER */}
      {renderOfflineBanner()}

      {/* CALIBRATION BANNER */}
      {renderCalibrationBanner()}

      {/* HERO SECTION: 3D CYBER CAR + SMART ALLOCATION */}
      <section className="hero-section">
        <div className="three-car-panel">
          <ThreeCarCanvas isRunning={isCarBoosted} carColor="#00f0ff" />
        </div>

        <div className="action-hero-panel">
          <div>
            <div className="action-hero-title">
              <span>⚡</span>
              <h3>VIP SMART ALLOCATION</h3>
            </div>
            <p className="eyebrow" style={{ marginTop: 4 }}>
              Instant AI Nearest-Bay Reservation &amp; Digital Pass
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
                disabled={!backendOnline}
              />
            </div>

            <div className="cyber-input-group">
              <label>Vehicle Classification</label>
              <select
                className="cyber-input"
                value={vehicleType}
                onChange={(e) => setVehicleType(e.target.value)}
                disabled={!backendOnline}
              >
                <option value="car">Hypercar / Sedan</option>
                <option value="suv">Luxury SUV</option>
                <option value="ev">Electric Vehicle (EV)</option>
                <option value="bike">Superbike</option>
              </select>
            </div>

            <button type="submit" className="cyber-btn primary" disabled={!backendOnline}>
              RESERVE OPTIMAL BAY ➔
            </button>
          </form>

          <div className="quick-tools">
            <button className="cyber-btn secondary" onClick={simulateTraffic} disabled={!backendOnline} style={{ flex: 1 }}>
              🎲 SIMULATE TRAFFIC
            </button>
            <button className="cyber-btn secondary" onClick={resetBays} disabled={!backendOnline} style={{ flex: 1 }}>
              🔄 RESET ALL BAYS
            </button>
          </div>
        </div>
      </section>

      {/* STATS TELEMETRY GRID */}
      <section className="stats-grid">
        <div className="stat-card cyan">
          <span className="stat-label">Total Parking Bays</span>
          <strong className="stat-val">{displayOverview.total_slots}</strong>
          <span className="stat-sub">Camera-Monitored</span>
        </div>
        <div className="stat-card emerald">
          <span className="stat-label">Available Slots</span>
          <strong className="stat-val" style={{ color: 'var(--neon-emerald)' }}>{displayOverview.available}</strong>
          <span className="stat-sub">Ready for parking</span>
        </div>
        <div className="stat-card crimson">
          <span className="stat-label">Occupied Slots</span>
          <strong className="stat-val" style={{ color: 'var(--neon-crimson)' }}>{displayOverview.occupied}</strong>
          <span className="stat-sub">Vehicles detected</span>
        </div>
        <div className="stat-card amber">
          <span className="stat-label">Reserved Passes</span>
          <strong className="stat-val" style={{ color: 'var(--neon-amber)' }}>{displayOverview.reserved}</strong>
          <span className="stat-sub">VIP assigned</span>
        </div>
        <div className="stat-card cyan">
          <span className="stat-label">Occupancy Rate</span>
          <strong className="stat-val">{displayOverview.occupancy_pct}%</strong>
          <span className="stat-sub">Real-time capacity</span>
        </div>
      </section>

      {/* WORKSPACE: VISION SCANNER + PARKING GRID */}
      <section className="workspace-grid">
        {/* LEFT: VISION ENGINE */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>👁️</span>
              <h3>VISION INTELLIGENCE</h3>
            </div>
            <div className="tab-group">
              <button className={`tab-btn ${activeTab === 'upload' ? 'active' : ''}`} onClick={() => setActiveTab('upload')}>
                📸 IMAGE SCAN
              </button>
              <button className={`tab-btn ${activeTab === 'stream' ? 'active' : ''}`} onClick={() => setActiveTab('stream')}>
                📹 LIVE STREAM
              </button>
            </div>
          </div>

          {activeTab === 'upload' ? (
            <>
              <div className="scanner-viewport" onClick={() => backendOnline && fileInputRef.current?.click()}>
                <input
                  type="file"
                  ref={fileInputRef}
                  style={{ display: 'none' }}
                  accept="image/*"
                  capture="environment"
                  onChange={(e) => { if (e.target.files?.[0]) handleFileUpload(e.target.files[0]) }}
                />
                {isScanning && <div className="scanner-beam" />}
                {annotatedImage ? (
                  <img src={annotatedImage} alt="Detection Result" className="scanner-image-preview" />
                ) : (
                  <div className="dropzone-empty">
                    <span className="dropzone-icon">📷</span>
                    <div className="dropzone-text">
                      <h4>
                        {backendOnline
                          ? 'Drop Parking Photo or Tap to Snap'
                          : 'Backend Offline — Detection Disabled'}
                      </h4>
                      <p>
                        {backendOnline
                          ? 'Supports JPG, PNG, Mobile Camera, CCTV stills'
                          : 'Waiting for backend server…'}
                      </p>
                    </div>
                    {backendOnline && (
                      <button className="cyber-btn secondary" onClick={(e) => { e.stopPropagation(); fileInputRef.current?.click() }}>
                        SELECT IMAGE
                      </button>
                    )}
                  </div>
                )}
              </div>

              {/* Real sample photo presets */}
              <div className="preset-strip">
                <span>⚡ Test with real photos:</span>
                <button className="preset-chip" onClick={() => runSamplePhoto('sample_cardboard_test.jpg')} disabled={!backendOnline || isScanning}>
                  📦 Cardboard Lot
                </button>
                <button className="preset-chip" onClick={() => runSamplePhoto('sample_cctv_test.jpg')} disabled={!backendOnline || isScanning}>
                  📹 CCTV Frame
                </button>
              </div>

              {telemetry && (
                <div className="telemetry-bar">
                  <span>Occupied Bays: <strong>{telemetry.detectedCount}</strong></span>
                  <span>Time: <strong>{telemetry.inferenceTime}ms</strong></span>
                  <span>Mode: <strong>{telemetry.mode === 'background_subtraction' ? 'BG-SUB' : 'YOLOv8'}</strong></span>
                </div>
              )}
            </>
          ) : (
            <>
              <div className="camera-mode-tabs">
                <button className={`tab-btn ${cameraMode === 'device' ? 'active' : ''}`} onClick={() => setCameraMode('device')}>
                  📱 MOBILE CAMERA
                </button>
                <button className={`tab-btn ${cameraMode === 'server' ? 'active' : ''}`} onClick={() => setCameraMode('server')}>
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
                      autoPlay muted playsInline
                    />
                    {!deviceCameraActive && (
                      <div className="dropzone-empty camera-prompt">
                        <span className="dropzone-icon">📱</span>
                        <div className="dropzone-text">
                          <h4>Use your phone camera for live parking scans</h4>
                          <p>Allow camera access, then snap a frame for AI detection</p>
                        </div>
                      </div>
                    )}
                  </div>
                  <div style={{ display: 'flex', gap: 10 }}>
                    <button className={`cyber-btn ${deviceCameraActive ? 'secondary' : 'primary'}`} style={{ flex: 1 }} onClick={toggleDeviceCamera}>
                      {deviceCameraActive ? '⏸️ STOP CAMERA' : '📱 START MOBILE CAMERA'}
                    </button>
                    <button className="cyber-btn secondary" style={{ flex: 1 }} onClick={scanDeviceCameraFrame} disabled={!deviceCameraActive || isScanning || !backendOnline}>
                      {isScanning ? '⏳ SCANNING…' : '🔎 SCAN FRAME'}
                    </button>
                  </div>
                </>
              ) : (
                <>
                  <div className="scanner-viewport">
                    {backendOnline ? (
                      <img
                        src={`${API_BASE}/camera/feed`}
                        alt="Live CCTV"
                        className="scanner-image-preview"
                        style={{ maxHeight: 360, width: '100%', background: '#000' }}
                      />
                    ) : (
                      <div className="dropzone-empty">
                        <span className="dropzone-icon">📡</span>
                        <div className="dropzone-text"><h4>Backend Offline</h4><p>Server CCTV stream unavailable</p></div>
                      </div>
                    )}
                  </div>
                  <button className={`cyber-btn ${cameraActive ? 'secondary' : 'primary'}`} onClick={toggleCamera} disabled={!backendOnline}>
                    {cameraActive ? '⏸️ PAUSE LIVE DETECTION' : '▶️ START LIVE SCANNING'}
                  </button>
                </>
              )}

              <div className="telemetry-bar">
                <span>Source: <strong>{cameraMode === 'device' ? 'MOBILE DEVICE' : 'SERVER CCTV'}</strong></span>
                <span>Status: <strong>{cameraMode === 'device' ? (deviceCameraActive ? 'LIVE' : 'STANDBY') : (cameraActive ? 'SCANNING' : 'STANDBY')}</strong></span>
                <span>Mode: <strong>{detectionMode === 'background_subtraction' ? 'BG-SUB' : 'YOLOv8'}</strong></span>
              </div>
            </>
          )}
        </div>

        {/* RIGHT: PARKING GRID */}
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title">
              <span>🅿️</span>
              <h3>FACILITY PARKING BAYS</h3>
            </div>
            <div className="tab-group">
              <button className={`tab-btn ${sectionFilter === 'ALL' ? 'active' : ''}`} onClick={() => setSectionFilter('ALL')}>ALL</button>
              {uniqueSections.map((sec) => (
                <button key={sec} className={`tab-btn ${sectionFilter === sec ? 'active' : ''}`} onClick={() => setSectionFilter(sec)}>
                  SEC {sec}
                </button>
              ))}
            </div>
          </div>

          {slots.length === 0 && backendOnline === false ? (
            <div className="dropzone-empty" style={{ minHeight: 200 }}>
              <span className="dropzone-icon">📡</span>
              <div className="dropzone-text">
                <h4>No bay data</h4>
                <p>{wakingUp ? 'Waiting for backend…' : 'Backend offline'}</p>
              </div>
            </div>
          ) : (
            <div className="parking-grid">
              {filteredSlots.map((slot) => {
                const statusClass = slot.status.toLowerCase()
                return (
                  <div
                    key={slot.slot_id}
                    className={`slot-card ${statusClass}`}
                    onClick={() => { if (slot.status === 'AVAILABLE') setLicensePlate(`VIP-${slot.slot_id}`) }}
                    title={slot.status === 'AVAILABLE' ? 'Click to pre-fill reservation' : slot.status}
                  >
                    <div className="slot-header">
                      <span className="slot-id-badge">{slot.slot_id}</span>
                      <span className="slot-badge">{slot.status}</span>
                    </div>
                    <div className="slot-visual-center">
                      {slot.status === 'OCCUPIED' ? (
                        <span role="img" aria-label="Car" style={{ filter: 'drop-shadow(0 0 10px rgba(255,0,85,0.6))' }}>🏎️</span>
                      ) : slot.status === 'RESERVED' ? (
                        <span role="img" aria-label="Reserved">🎫</span>
                      ) : (
                        <span role="img" aria-label="Available" style={{ filter: 'drop-shadow(0 0 10px rgba(0,255,157,0.5))' }}>🅿️</span>
                      )}
                    </div>
                    <div className="slot-meta">
                      <span>{slot.vehicle_id || (slot.status === 'AVAILABLE' ? 'TAP TO RESERVE' : 'IN USE')}</span>
                      <span>{slot.confidence ? `${Math.round(slot.confidence * 100)}% CONF` : ''}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>
      </section>

      {/* BOTTOM: EVENTS + SYSTEM HEALTH */}
      <section className="bottom-grid">
        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>📡</span><h3>REAL-TIME AUDIT LOG</h3></div>
            <span className="hud-badge">{events.length} LOGGED</span>
          </div>
          <ul className="event-list">
            {events.length ? (
              [...events].reverse().map((ev, idx) => (
                <li key={ev.event_id || idx} className="event-item">
                  <span className="event-time">{ev.timestamp ? ev.timestamp.slice(11, 19) : 'LIVE'}</span>
                  <span className="event-slot">{ev.slot_id}</span>
                  <span className={`event-status ${ev.status?.toLowerCase()}`}>{ev.event_type || ev.status}</span>
                  <span style={{ color: 'var(--text-muted)' }}>
                    {ev.vehicle_id ? `[${ev.vehicle_id}]` : `${Math.round((ev.confidence || 0.95) * 100)}%`}
                  </span>
                </li>
              ))
            ) : (
              <li className="event-item">No events logged yet</li>
            )}
          </ul>
        </div>

        <div className="panel">
          <div className="panel-header">
            <div className="panel-title"><span>⚙️</span><h3>SYSTEM TELEMETRY</h3></div>
            <span className={`status-pill ${backendOnline ? '' : 'offline'}`}>
              {backendOnline ? '● ONLINE' : '● OFFLINE'}
            </span>
          </div>
          <div className="health-grid">
            <div className="health-item">
              <label>DETECTION MODE</label>
              <strong>{detectionMode === 'background_subtraction' ? 'BG SUBTRACTION' : 'YOLOv8'}</strong>
            </div>
            <div className="health-item">
              <label>BG CALIBRATED</label>
              <strong style={{ color: bgCalibrated ? 'var(--neon-emerald)' : 'var(--neon-crimson)' }}>
                {bgCalibrated ? '✅ YES' : '❌ NEEDS SETUP'}
              </strong>
            </div>
            <div className="health-item">
              <label>OCCUPANCY ENGINE</label>
              <strong>POLYGON IoU + BG-SUB</strong>
            </div>
            <div className="health-item">
              <label>LIVE UPDATES</label>
              <strong style={{ color: wsRef.current?.readyState === 1 ? 'var(--neon-emerald)' : 'var(--text-muted)' }}>
                {wsRef.current?.readyState === 1 ? 'WEBSOCKET' : 'POLLING'}
              </strong>
            </div>
          </div>
          <div className="section-list">
            {sections.map((section) => (
              <div key={section.section_id}>
                <label>Section {section.section_id}</label>
                <span>{section.available}/{section.total} available</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* VIP TICKET MODAL */}
      <PassModal ticket={issuedTicket} onClose={() => setIssuedTicket(null)} />
    </div>
  )
}
