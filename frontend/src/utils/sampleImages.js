/**
 * Generates synthetic realistic parking lot snapshot blobs for instant 1-click testing.
 */
export function generateParkingSnapshot(scenario = 'half') {
  return new Promise((resolve) => {
    const canvas = document.createElement('canvas')
    canvas.width = 640
    canvas.height = 360
    const ctx = canvas.getContext('2d')

    // Asphalt ground
    const bgGrad = ctx.createLinearGradient(0, 0, 0, 360)
    bgGrad.addColorStop(0, '#1a1f2c')
    bgGrad.addColorStop(1, '#111622')
    ctx.fillStyle = bgGrad
    ctx.fillRect(0, 0, 640, 360)

    // Parking lot tarmac
    ctx.fillStyle = '#222938'
    ctx.fillRect(50, 40, 540, 240)

    // Yellow and white parking demarcation markings
    ctx.strokeStyle = '#e2b714'
    ctx.lineWidth = 3
    ctx.beginPath()
    ctx.setLineDash([12, 12])
    ctx.moveTo(60, 150)
    ctx.lineTo(580, 150)
    ctx.stroke()
    ctx.setLineDash([])

    // Draw slot dividers
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.6)'
    ctx.lineWidth = 2
    for (let x = 100; x <= 500; x += 90) {
      // Top row slots
      ctx.strokeRect(x, 60, 80, 60)
      // Bottom row slots
      ctx.strokeRect(x, 180, 80, 60)

      // Slot numbers
      ctx.fillStyle = 'rgba(255,255,255,0.4)'
      ctx.font = 'bold 12px monospace'
      ctx.fillText(`P-${Math.floor(x / 90)}`, x + 8, 80)
      ctx.fillText(`P-${Math.floor(x / 90) + 4}`, x + 8, 200)
    }

    // Determine occupied slot positions based on scenario
    let occupiedCoords = []
    if (scenario === 'half') {
      occupiedCoords = [
        { x: 110, y: 70, color: '#e63946', label: 'Car 01' },
        { x: 290, y: 70, color: '#457b9d', label: 'Car 02' },
        { x: 200, y: 190, color: '#f4a261', label: 'Car 03' },
      ]
    } else if (scenario === 'crowded') {
      occupiedCoords = [
        { x: 110, y: 70, color: '#e63946', label: 'Car 01' },
        { x: 200, y: 70, color: '#2a9d8f', label: 'Car 02' },
        { x: 290, y: 70, color: '#457b9d', label: 'Car 03' },
        { x: 380, y: 70, color: '#e76f51', label: 'Car 04' },
        { x: 110, y: 190, color: '#9b5de5', label: 'Car 05' },
        { x: 290, y: 190, color: '#00bbf9', label: 'Car 06' },
      ]
    } // empty scenario has no cars

    // Draw realistic miniature cars in the slots
    occupiedCoords.forEach((car) => {
      // Car shadow
      ctx.fillStyle = 'rgba(0,0,0,0.5)'
      ctx.beginPath()
      ctx.ellipse(car.x + 30, car.y + 22, 34, 18, 0, 0, Math.PI * 2)
      ctx.fill()

      // Car body
      ctx.fillStyle = car.color
      ctx.roundRect ? ctx.roundRect(car.x, car.y, 60, 36, 6) : ctx.fillRect(car.x, car.y, 60, 36)
      ctx.fill()

      // Windshields & Roof
      ctx.fillStyle = '#0f172a'
      ctx.fillRect(car.x + 15, car.y + 6, 28, 24)
      ctx.fillStyle = car.color
      ctx.fillRect(car.x + 20, car.y + 8, 18, 20)

      // Headlights
      ctx.fillStyle = '#fef08a'
      ctx.fillRect(car.x + 54, car.y + 4, 4, 6)
      ctx.fillRect(car.x + 54, car.y + 26, 4, 6)

      // Taillights
      ctx.fillStyle = '#ef4444'
      ctx.fillRect(car.x + 2, car.y + 4, 4, 6)
      ctx.fillRect(car.x + 2, car.y + 26, 4, 6)
    })

    // Timestamp & Camera Watermark
    ctx.fillStyle = '#00ff9d'
    ctx.font = 'bold 12px monospace'
    ctx.fillText(`CAM_01 CCTV FEED • [${new Date().toLocaleTimeString()}]`, 60, 310)

    canvas.toBlob((blob) => {
      resolve(blob)
    }, 'image/jpeg', 0.92)
  })
}
