import { useState } from "react";
import { colours } from "./types";

export function LineChart({
  series,
  label,
  percent = false,
}: {
  series: { name: string; values: number[]; colour?: string }[];
  label: string;
  percent?: boolean;
}) {
  const [hover, setHover] = useState<number | null>(null);
  const all = series.flatMap((s) => s.values).filter(Number.isFinite);
  if (!all.length) return null;
  const min = Math.min(...all),
    max = Math.max(...all),
    span = max - min || 1,
    count = Math.max(...series.map((s) => s.values.length));
  const x = (i: number) => 48 + (i / Math.max(1, count - 1)) * 622,
    y = (v: number) => 174 - ((v - min) / span) * 142;
  const format = (v: number) =>
    percent ? `${(v * 100).toFixed(1)}%` : v.toFixed(1);
  return (
    <div className="line-chart">
      <div className="chart-legend">
        {series.map((s, i) => (
          <span key={s.name}>
            <i style={{ background: s.colour || colours[i % 3] }} />
            {s.name}
            {hover !== null && s.values[hover] !== undefined ? (
              <strong>{format(s.values[hover])}</strong>
            ) : null}
          </span>
        ))}
      </div>
      <svg
        viewBox="0 0 700 205"
        role="img"
        aria-label={label}
        onMouseMove={(e) => {
          const b = e.currentTarget.getBoundingClientRect();
          setHover(
            Math.min(
              count - 1,
              Math.max(
                0,
                Math.round(
                  ((((e.clientX - b.left) / b.width) * 700 - 48) / 622) *
                    (count - 1),
                ),
              ),
            ),
          );
        }}
        onMouseLeave={() => setHover(null)}
      >
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line
              x1="48"
              x2="670"
              y1={32 + 142 * f}
              y2={32 + 142 * f}
              stroke="#324039"
              strokeDasharray="3 5"
            />
            <text
              x="38"
              y={36 + 142 * f}
              textAnchor="end"
              fill="#89978c"
              fontSize="10"
            >
              {format(max - f * span)}
            </text>
          </g>
        ))}
        {series.map((s, i) => (
          <path
            key={s.name}
            d={s.values
              .map(
                (v, j) =>
                  `${j ? "L" : "M"}${x(j).toFixed(2)},${y(v).toFixed(2)}`,
              )
              .join(" ")}
            fill="none"
            stroke={s.colour || colours[i % 3]}
            strokeWidth="1.8"
            opacity=".9"
          />
        ))}
        {hover !== null && (
          <g>
            <line
              x1={x(hover)}
              x2={x(hover)}
              y1="24"
              y2="174"
              stroke="#cad9ca"
              opacity=".4"
            />
            {series.map((s, i) =>
              s.values[hover] === undefined ? null : (
                <circle
                  key={s.name}
                  cx={x(hover)}
                  cy={y(s.values[hover])}
                  r="3.5"
                  fill={s.colour || colours[i % 3]}
                />
              ),
            )}
            <text
              x={x(hover)}
              y="195"
              fill="#e8ecdf"
              textAnchor="middle"
              fontSize="10"
            >
              {hover}
            </text>
          </g>
        )}
        <text x="48" y="195" fill="#89978c" fontSize="10">
          0
        </text>
        <text x="670" y="195" textAnchor="end" fill="#89978c" fontSize="10">
          {count - 1}
        </text>
      </svg>
    </div>
  );
}
