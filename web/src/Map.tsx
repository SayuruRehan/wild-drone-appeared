import { useEffect, useRef } from "react";
import type { Frame, Replay } from "./types";
import { colours } from "./types";

export type Layers = {
  freshness: boolean;
  priority: boolean;
  sensors: boolean;
  trails: boolean;
};
export function MissionMap({
  replay,
  frame,
  layers,
  drone,
  onCell,
  compact = false,
}: {
  replay: Replay;
  frame: Frame;
  layers: Layers;
  drone: number | null;
  onCell: (index: number) => void;
  compact?: boolean;
}) {
  const ref = useRef<HTMLCanvasElement>(null);
  const last = useRef(frame.positions);
  useEffect(() => {
    const canvas = ref.current!;
    const ctx = canvas.getContext("2d")!;
    let animation = 0;
    const previous = last.current;
    last.current = frame.positions;
    const reduced = window.matchMedia(
      "(prefers-reduced-motion: reduce)",
    ).matches;
    const begin = performance.now();
    const draw = (now: number) => {
      const size = canvas.clientWidth;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = size * dpr;
      canvas.height = size * dpr;
      ctx.scale(dpr, dpr);
      const n = replay.config.grid_size,
        unit = size / n;
      ctx.fillStyle = "#14261e";
      ctx.fillRect(0, 0, size, size);
      for (let i = 0; i < n * n; i++) {
        const x = (i % n) * unit,
          y = Math.floor(i / n) * unit;
        const fuel = frame.fuel[i] / replay.config.max_fuel,
          terrain = replay.terrain[i];
        if (layers.freshness) {
          const f =
            1 - Math.min(frame.ages[i] / replay.config.inspection_deadline, 1);
          ctx.fillStyle = `rgb(${Math.round(21 + 45 * f)},${Math.round(40 + 128 * f)},${Math.round(34 + 99 * f)})`;
        } else {
          ctx.fillStyle =
            frame.fuel[i] === 0
              ? "#26302a"
              : `rgb(${Math.round(26 + 22 * terrain)},${Math.round(44 + 29 * fuel + 13 * terrain)},${Math.round(31 + 12 * fuel)})`;
        }
        ctx.fillRect(x, y, unit + 0.4, unit + 0.4);
        if (layers.priority && replay.priority[i] > 1) {
          ctx.fillStyle = "rgba(212,175,100,.16)";
          ctx.fillRect(x, y, unit, unit);
        }
        if (!layers.freshness && frame.burning[i]) {
          ctx.fillStyle = frame.fuel[i] > 30 ? "#f38144" : "#ad512e";
          ctx.fillRect(
            x + 0.7,
            y + 0.7,
            Math.max(1, unit - 1.4),
            Math.max(1, unit - 1.4),
          );
          ctx.fillStyle = "#ffd094";
          ctx.fillRect(x + unit * 0.3, y + unit * 0.3, unit * 0.4, unit * 0.4);
        }
      }
      ctx.strokeStyle = "rgba(197,218,197,.055)";
      ctx.lineWidth = 0.5;
      for (let i = 0; i <= n; i++) {
        ctx.beginPath();
        ctx.moveTo(i * unit, 0);
        ctx.lineTo(i * unit, size);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(0, i * unit);
        ctx.lineTo(size, i * unit);
        ctx.stroke();
      }
      if (layers.trails)
        for (let d = 0; d < 3; d++) {
          ctx.strokeStyle = colours[d];
          ctx.globalAlpha = drone === null || drone === d ? 0.64 : 0.12;
          ctx.lineWidth = compact ? 1.5 : 2;
          ctx.beginPath();
          replay.frames
            .slice(Math.max(0, frame.step - 70), frame.step + 1)
            .forEach((f, k) => {
              const [r, c] = f.positions[d];
              if (k === 0) ctx.moveTo((c + 0.5) * unit, (r + 0.5) * unit);
              else ctx.lineTo((c + 0.5) * unit, (r + 0.5) * unit);
            });
          ctx.stroke();
          ctx.globalAlpha = 1;
        }
      const [br, bc] = replay.base;
      ctx.fillStyle = "#dae4cf";
      ctx.strokeStyle = "#111c17";
      ctx.lineWidth = 2;
      ctx.fillRect((bc + 0.5) * unit - 7, (br + 0.5) * unit - 7, 14, 14);
      ctx.strokeRect((bc + 0.5) * unit - 7, (br + 0.5) * unit - 7, 14, 14);
      ctx.font = "600 9px system-ui";
      ctx.fillStyle = "#edf2e4";
      ctx.fillText("BASE", (bc + 0.5) * unit + 12, (br + 0.5) * unit + 3);
      const t = reduced ? 1 : Math.min(1, (now - begin) / 160);
      for (let d = 0; d < 3; d++) {
        const [r, c] = frame.positions[d];
        const old = previous[d] || [r, c];
        const atBase = r === br && c === bc;
        const x =
            (old[1] + (c - old[1]) * t + 0.5) * unit +
            (atBase ? (d - 1) * 13 : 0),
          y = (old[0] + (r - old[0]) * t + 0.5) * unit + (atBase ? -14 : 0);
        ctx.globalAlpha = drone === null || drone === d ? 1 : 0.4;
        if (layers.sensors && !frame.lost[d]) {
          const radius = replay.config.sensor_radius;
          ctx.fillStyle = colours[d] + "16";
          ctx.strokeStyle = colours[d] + "60";
          ctx.lineWidth = 1;
          ctx.setLineDash([3, 3]);
          const left = Math.max(0, c - radius) * unit,
            top = Math.max(0, r - radius) * unit;
          const right = Math.min(n, c + radius + 1) * unit,
            bottom = Math.min(n, r + radius + 1) * unit;
          ctx.fillRect(left, top, right - left, bottom - top);
          ctx.strokeRect(left, top, right - left, bottom - top);
          ctx.setLineDash([]);
        }
        ctx.save();
        ctx.translate(x, y);
        ctx.strokeStyle = frame.lost[d] ? "#85918b" : colours[d];
        ctx.fillStyle = "#12231c";
        ctx.lineWidth = 1.7;
        if (drone === d) {
          ctx.beginPath();
          ctx.arc(0, 0, 14, 0, Math.PI * 2);
          ctx.stroke();
        }
        ctx.beginPath();
        ctx.moveTo(-5, -5);
        ctx.lineTo(5, 5);
        ctx.moveTo(-5, 5);
        ctx.lineTo(5, -5);
        ctx.stroke();
        for (const [dx, dy] of [
          [-5, -5],
          [5, 5],
          [-5, 5],
          [5, -5],
        ]) {
          ctx.beginPath();
          ctx.arc(dx, dy, 3, 0, Math.PI * 2);
          ctx.fill();
          ctx.stroke();
        }
        ctx.fillStyle = frame.lost[d] ? "#85918b" : colours[d];
        ctx.fillRect(-2, -3, 4, 6);
        ctx.restore();
        if (!atBase) {
          ctx.font = "600 10px system-ui";
          ctx.fillStyle = colours[d];
          ctx.fillText(`D${d + 1}`, x + 10, y - 8);
        }
        ctx.globalAlpha = 1;
      }
      if (t < 1) animation = requestAnimationFrame(draw);
    };
    animation = requestAnimationFrame(draw);
    const observer = new ResizeObserver(() => {
      cancelAnimationFrame(animation);
      animation = requestAnimationFrame(draw);
    });
    observer.observe(canvas);
    return () => {
      cancelAnimationFrame(animation);
      observer.disconnect();
    };
  }, [replay, frame, layers, drone, compact]);
  return (
    <canvas
      ref={ref}
      className="mission-canvas"
      role="img"
      aria-label={`Actual ${replay.config.grid_size} by ${replay.config.grid_size} mission map at step ${frame.step}. ${frame.lost.filter(Boolean).length} drones lost. Use the row and column controls to inspect a cell.`}
      onClick={(event) => {
        const box = event.currentTarget.getBoundingClientRect(),
          n = replay.config.grid_size;
        const col = Math.min(
            n - 1,
            Math.floor(((event.clientX - box.left) / box.width) * n),
          ),
          row = Math.min(
            n - 1,
            Math.floor(((event.clientY - box.top) / box.height) * n),
          );
        onCell(row * n + col);
      }}
    />
  );
}
