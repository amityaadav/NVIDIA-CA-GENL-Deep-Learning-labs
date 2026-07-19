import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from "react";

const SIZE = 280; // display + backing size; multiple of 28 keeps downscaling clean

/**
 * A square canvas the user draws a single digit on with mouse or trackpad.
 * Dark stroke on white so it reads like pen on paper; preprocessing inverts it.
 * Shows a hint inside the box while it's blank. Exposes clear() and the canvas.
 */
const DrawCanvas = forwardRef(function DrawCanvas({ onStrokeEnd }, ref) {
  const canvasRef = useRef(null);
  const drawing = useRef(false);
  const last = useRef({ x: 0, y: 0 });
  const [blank, setBlank] = useState(true); // toggles the "draw here" hint

  useImperativeHandle(ref, () => ({
    canvas: () => canvasRef.current,
    clear: () => { paintBlank(canvasRef.current); setBlank(true); },
  }));

  useEffect(() => {
    paintBlank(canvasRef.current);
  }, []);

  const posFromEvent = (e) => {
    const rect = canvasRef.current.getBoundingClientRect();
    const p = e.touches ? e.touches[0] : e;
    return {
      x: ((p.clientX - rect.left) / rect.width) * SIZE,
      y: ((p.clientY - rect.top) / rect.height) * SIZE,
    };
  };

  const start = (e) => {
    e.preventDefault();
    drawing.current = true;
    last.current = posFromEvent(e);
    if (blank) setBlank(false); // hide the hint once drawing begins
  };

  const move = (e) => {
    if (!drawing.current) return;
    e.preventDefault();
    const ctx = canvasRef.current.getContext("2d");
    const p = posFromEvent(e);
    ctx.strokeStyle = "#111";
    ctx.lineWidth = 20;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.beginPath();
    ctx.moveTo(last.current.x, last.current.y);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
    last.current = p;
  };

  const end = () => {
    if (!drawing.current) return;
    drawing.current = false;
    onStrokeEnd?.();
  };

  return (
    <div className="draw-wrap">
      <canvas
        ref={canvasRef}
        width={SIZE}
        height={SIZE}
        className="draw-canvas"
        onMouseDown={start}
        onMouseMove={move}
        onMouseUp={end}
        onMouseLeave={end}
        onTouchStart={start}
        onTouchMove={move}
        onTouchEnd={end}
        aria-label="Draw a digit from 0 to 9"
      />
      {blank && <div className="draw-placeholder">Draw a digit, then run inference</div>}
    </div>
  );
});

function paintBlank(canvas) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

export default DrawCanvas;
