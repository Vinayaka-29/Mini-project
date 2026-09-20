import React, { useEffect, useRef, useState } from 'react'
import * as THREE from 'three'

export default function ThreeCarCanvas({ isRunning = true, carColor = '#00f0ff' }) {
  const mountRef = useRef(null)
  const [headlightsOn, setHeadlightsOn] = useState(true)
  const [nitroActive, setNitroActive] = useState(false)
  const carGroupRef = useRef(null)
  const wheelsRef = useRef([])
  const particlesRef = useRef(null)
  const lightsRef = useRef([])

  useEffect(() => {
    const currentMount = mountRef.current
    if (!currentMount) return

    const width = currentMount.clientWidth || 600
    const height = currentMount.clientHeight || 280

    // Scene
    const scene = new THREE.Scene()

    // Camera
    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 1000)
    camera.position.set(5.5, 3.2, 6.0)
    camera.lookAt(0, 0.4, 0)

    // Renderer
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
    renderer.setSize(width, height)
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    renderer.shadowMap.enabled = true
    renderer.shadowMap.type = THREE.PCFSoftShadowMap
    currentMount.appendChild(renderer.domElement)

    // Lighting
    const ambientLight = new THREE.AmbientLight(0xffffff, 1.2)
    scene.add(ambientLight)

    const mainLight = new THREE.DirectionalLight(0x00f0ff, 2.5)
    mainLight.position.set(5, 8, 5)
    mainLight.castShadow = true
    scene.add(mainLight)

    const rimLight = new THREE.DirectionalLight(0xff0077, 2.0)
    rimLight.position.set(-5, 4, -5)
    scene.add(rimLight)

    // Cyber Grid Floor
    const gridHelper = new THREE.GridHelper(24, 24, 0x00f0ff, 0x112233)
    gridHelper.position.y = -0.01
    scene.add(gridHelper)

    // Holographic Circular Ring on Floor
    const ringGeo = new THREE.RingGeometry(2.2, 2.3, 48)
    const ringMat = new THREE.MeshBasicMaterial({ color: 0x00ff9d, side: THREE.DoubleSide, transparent: true, opacity: 0.6 })
    const floorRing = new THREE.Mesh(ringGeo, ringMat)
    floorRing.rotation.x = -Math.PI / 2
    scene.add(floorRing)

    // Car Body Construction (Cyber Luxury Hypercar)
    const carGroup = new THREE.Group()
    carGroupRef.current = carGroup
    scene.add(carGroup)

    // Materials
    const bodyMaterial = new THREE.MeshStandardMaterial({
      color: new THREE.Color(carColor),
      metalness: 0.85,
      roughness: 0.18,
    })

    const carbonMaterial = new THREE.MeshStandardMaterial({
      color: 0x111317,
      metalness: 0.9,
      roughness: 0.3,
    })

    const glassMaterial = new THREE.MeshPhysicalMaterial({
      color: 0x0a1018,
      metalness: 0.1,
      roughness: 0.05,
      transmission: 0.8,
      transparent: true,
      opacity: 0.85,
    })

    const glowCyanMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff })
    const glowRedMat = new THREE.MeshBasicMaterial({ color: 0xff0055 })
    const neonAmberMat = new THREE.MeshBasicMaterial({ color: 0xffb700 })

    // Lower Chassis
    const lowerBodyGeo = new THREE.BoxGeometry(3.6, 0.45, 1.7)
    const lowerBody = new THREE.Mesh(lowerBodyGeo, bodyMaterial)
    lowerBody.position.y = 0.45
    lowerBody.castShadow = true
    carGroup.add(lowerBody)

    // Cabin / Roof
    const cabinGeo = new THREE.BoxGeometry(1.8, 0.45, 1.35)
    const cabin = new THREE.Mesh(cabinGeo, glassMaterial)
    cabin.position.set(-0.2, 0.82, 0)
    cabin.castShadow = true
    carGroup.add(cabin)

    // Front Nose Wedge
    const noseGeo = new THREE.CylinderGeometry(0.75, 0.85, 0.7, 4)
    const nose = new THREE.Mesh(noseGeo, bodyMaterial)
    nose.rotation.y = Math.PI / 4
    nose.rotation.z = Math.PI / 2
    nose.position.set(1.6, 0.42, 0)
    carGroup.add(nose)

    // Rear Spoiler
    const spoilerGeo = new THREE.BoxGeometry(0.2, 0.06, 1.8)
    const spoiler = new THREE.Mesh(spoilerGeo, carbonMaterial)
    spoiler.position.set(-1.7, 0.95, 0)
    carGroup.add(spoiler)

    const spoilerStand1 = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.35, 0.08), carbonMaterial)
    spoilerStand1.position.set(-1.65, 0.75, 0.5)
    carGroup.add(spoilerStand1)

    const spoilerStand2 = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.35, 0.08), carbonMaterial)
    spoilerStand2.position.set(-1.65, 0.75, -0.5)
    carGroup.add(spoilerStand2)

    // Underglow Neon Bar
    const underglowGeo = new THREE.BoxGeometry(3.2, 0.04, 1.4)
    const underglowMat = new THREE.MeshBasicMaterial({ color: 0x00f0ff, transparent: true, opacity: 0.85 })
    const underglow = new THREE.Mesh(underglowGeo, underglowMat)
    underglow.position.y = 0.16
    carGroup.add(underglow)

    // Headlights (Twin Laser Optics)
    const hl1 = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.08, 0.35), glowCyanMat)
    hl1.position.set(1.8, 0.48, 0.55)
    carGroup.add(hl1)

    const hl2 = new THREE.Mesh(new THREE.BoxGeometry(0.1, 0.08, 0.35), glowCyanMat)
    hl2.position.set(1.8, 0.48, -0.55)
    carGroup.add(hl2)

    // Headlight Spotlights
    const spot1 = new THREE.SpotLight(0x00f0ff, 4, 15, Math.PI / 6, 0.4)
    spot1.position.set(1.8, 0.5, 0.55)
    spot1.target.position.set(6, 0, 0.55)
    scene.add(spot1.target)
    carGroup.add(spot1)

    const spot2 = new THREE.SpotLight(0x00f0ff, 4, 15, Math.PI / 6, 0.4)
    spot2.position.set(1.8, 0.5, -0.55)
    spot2.target.position.set(6, 0, -0.55)
    scene.add(spot2.target)
    carGroup.add(spot2)

    lightsRef.current = [spot1, spot2]

    // Taillights Strip
    const tl = new THREE.Mesh(new THREE.BoxGeometry(0.08, 0.08, 1.55), glowRedMat)
    tl.position.set(-1.8, 0.52, 0)
    carGroup.add(tl)

    // 4 Wheels
    const wheelGeo = new THREE.CylinderGeometry(0.34, 0.34, 0.28, 24)
    const rimGeo = new THREE.CylinderGeometry(0.24, 0.24, 0.3, 12)
    const rimMat = new THREE.MeshStandardMaterial({ color: 0x00ff9d, metalness: 0.95, roughness: 0.1 })
    const tireMat = new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.85 })

    const wheelPositions = [
      [1.1, 0.34, 0.88],
      [1.1, 0.34, -0.88],
      [-1.1, 0.34, 0.88],
      [-1.1, 0.34, -0.88],
    ]

    const wheelMeshes = []
    wheelPositions.forEach(([x, y, z]) => {
      const wheelGroup = new THREE.Group()
      wheelGroup.position.set(x, y, z)

      const tire = new THREE.Mesh(wheelGeo, tireMat)
      tire.rotation.x = Math.PI / 2
      tire.castShadow = true
      wheelGroup.add(tire)

      const rim = new THREE.Mesh(rimGeo, rimMat)
      rim.rotation.x = Math.PI / 2
      wheelGroup.add(rim)

      carGroup.add(wheelGroup)
      wheelMeshes.push(wheelGroup)
    })
    wheelsRef.current = wheelMeshes

    // Speed Trail Particles
    const particleCount = 80
    const particleGeo = new THREE.BufferGeometry()
    const particlePos = new Float32Array(particleCount * 3)
    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePos[i] = (Math.random() - 0.5) * 12
      particlePos[i + 1] = Math.random() * 2.5
      particlePos[i + 2] = (Math.random() - 0.5) * 8
    }
    particleGeo.setAttribute('position', new THREE.BufferAttribute(particlePos, 3))
    const particleMat = new THREE.PointsMaterial({
      color: 0x00f0ff,
      size: 0.08,
      transparent: true,
      opacity: 0.75,
    })
    const particleSystem = new THREE.Points(particleGeo, particleMat)
    scene.add(particleSystem)
    particlesRef.current = particleSystem

    // Mouse Orbit Interaction
    let mouseX = 0
    let targetRotation = 0
    const handleMouseMove = (e) => {
      const rect = currentMount.getBoundingClientRect()
      mouseX = ((e.clientX - rect.left) / rect.width) * 2 - 1
      targetRotation = mouseX * 0.8
    }
    window.addEventListener('mousemove', handleMouseMove)

    // Animation Loop
    let reqId
    let clock = new THREE.Clock()

    const animate = () => {
      reqId = requestAnimationFrame(animate)
      const delta = clock.getDelta()
      const time = clock.getElapsedTime()

      // Float & subtle suspension bounce
      if (carGroupRef.current) {
        carGroupRef.current.position.y = Math.sin(time * 3) * 0.03
        // Smooth rotation follow
        carGroupRef.current.rotation.y += (targetRotation + 0.35 - carGroupRef.current.rotation.y) * 0.05
      }

      // Rotate wheels if running
      if (isRunning) {
        wheelsRef.current.forEach((w) => {
          w.rotation.z -= delta * 8
        })
      }

      // Pulse floor ring
      floorRing.rotation.z += 0.01
      ringMat.opacity = 0.4 + Math.sin(time * 4) * 0.25

      // Move particle trail
      if (particlesRef.current) {
        const positions = particlesRef.current.geometry.attributes.position.array
        for (let i = 0; i < particleCount * 3; i += 3) {
          positions[i] -= delta * (nitroActive ? 16 : 6)
          if (positions[i] < -6) {
            positions[i] = 6
          }
        }
        particlesRef.current.geometry.attributes.position.needsUpdate = true
      }

      renderer.render(scene, camera)
    }

    animate()

    // Resize Handler
    const handleResize = () => {
      if (!currentMount) return
      const w = currentMount.clientWidth
      const h = currentMount.clientHeight
      camera.aspect = w / h
      camera.updateProjectionMatrix()
      renderer.setSize(w, h)
    }
    window.addEventListener('resize', handleResize)

    return () => {
      cancelAnimationFrame(reqId)
      window.removeEventListener('mousemove', handleMouseMove)
      window.removeEventListener('resize', handleResize)
      if (currentMount && renderer.domElement) {
        currentMount.removeChild(renderer.domElement)
      }
      renderer.dispose()
    }
  }, [isRunning, carColor, nitroActive])

  // Headlight toggle handler
  const toggleHeadlights = () => {
    setHeadlightsOn(!headlightsOn)
    lightsRef.current.forEach((l) => {
      l.intensity = headlightsOn ? 0 : 4
    })
  }

  return (
    <div className="three-car-container">
      <div ref={mountRef} className="three-viewport" />
      <div className="three-hud-overlay">
        <div className="hud-badge cyber-pulse">
          <span className="hud-dot" />
          <span>AI-POWERED 3D CHASSIS TELEMETRY</span>
        </div>
        <div className="three-controls">
          <button
            className={`cyber-btn-mini ${headlightsOn ? 'active' : ''}`}
            onClick={toggleHeadlights}
            title="Toggle Laser Headlights"
          >
            💡 {headlightsOn ? 'LIGHTS ON' : 'LIGHTS OFF'}
          </button>
          <button
            className={`cyber-btn-mini ${nitroActive ? 'nitro-active' : ''}`}
            onClick={() => setNitroActive(!nitroActive)}
            title="Activate Speed Burst"
          >
            ⚡ {nitroActive ? 'NITRO MAX' : 'BOOST'}
          </button>
        </div>
      </div>
    </div>
  )
}
