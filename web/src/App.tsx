import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowDown,
  ArrowRight,
  BatteryCharging,
  BookOpen,
  Check,
  CheckCheck,
  ChevronRight,
  CircleHelp,
  Compass,
  Download,
  ExternalLink,
  Flame,
  FlaskConical,
  Layers3,
  Leaf,
  Maximize2,
  Menu,
  Mountain,
  Pause,
  Play,
  Radar,
  RotateCcw,
  ShieldCheck,
  SkipForward,
  SlidersHorizontal,
  Target,
  Wind,
  X,
} from "lucide-react";
import { MissionMap, type Layers } from "./Map";
import { LineChart } from "./Charts";
import { frameAt, loadReplay, nearestMission } from "./data";
import {
  colours,
  commandNames,
  percent,
  policyLabel,
  policyNames,
} from "./types";
import type { Capabilities, Evidence, Frame, Replay, Row } from "./types";

type Page = "overview" | "mission" | "research" | "results" | "lab";
const navigation = [
  { id: "overview", icon: Compass, label: "Overview" },
  { id: "mission", icon: Radar, label: "Mission explorer" },
  { id: "research", icon: BookOpen, label: "The research" },
  { id: "results", icon: Activity, label: "Results & evidence" },
  { id: "lab", icon: FlaskConical, label: "Simulation lab" },
] as const;
const repository = "https://github.com/SayuruRehan/wild-drone-appeared";
const sourceLink = (path: string) =>
  `${repository}/blob/1eb5d20f5f9c5e63ffea717e8846ba4273df8d06/${path}`;
const windNames = ["Calm", "North", "South", "West", "East"];
const numeric = (row: Row, key: string) => Number(row[key]);

function useReplay(evidence: Evidence, selection: string, world: number) {
  const [replay, setReplay] = useState<Replay | null>(null),
    [error, setError] = useState("");
  const [policy, seedText] = selection.split(":");
  const mission = nearestMission(
    evidence.missions,
    policy,
    seedText === "none" ? null : Number(seedText),
    world,
  );
  useEffect(() => {
    let active = true;
    setReplay(null);
    setError("");
    loadReplay(mission)
      .then((r) => {
        if (active) setReplay(r);
      })
      .catch((e) => {
        if (active) setError(e.message);
      });
    return () => {
      active = false;
    };
  }, [mission]);
  return { replay, error };
}

function ControllerSelect({
  value,
  onChange,
  label,
}: {
  value: string;
  onChange: (value: string) => void;
  label: string;
}) {
  return (
    <label className="control-label">
      {label}
      <select
        aria-label={label}
        value={value}
        onChange={(e) => onChange(e.target.value)}
      >
        {["greedy", "sweep", "masked_random", "random"].map((p) => (
          <option key={p} value={`${p}:none`}>
            {policyLabel(p)}
          </option>
        ))}
        {["dqn", "reinforce", "a2c"].flatMap((p) =>
          [0, 1, 2].map((s) => (
            <option key={`${p}${s}`} value={`${p}:${s}`}>
              {policyLabel(p, s)} · masked
            </option>
          )),
        )}
      </select>
    </label>
  );
}

function Badge({
  children,
  amber = false,
}: {
  children: React.ReactNode;
  amber?: boolean;
}) {
  return (
    <span className={`badge ${amber ? "amber" : ""}`}>
      <span />
      {children}
    </span>
  );
}
function Intro({
  eyebrow,
  title,
  text,
  children,
}: {
  eyebrow: string;
  title: string;
  text: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="page-intro">
      <div>
        <div className="eyebrow">{eyebrow}</div>
        <h1>{title}</h1>
        <p>{text}</p>
      </div>
      {children}
    </div>
  );
}

function TimelineControls({
  step,
  setStep,
  playing,
  setPlaying,
  speed,
  setSpeed,
  horizon,
}: {
  step: number;
  setStep: (step: number) => void;
  playing: boolean;
  setPlaying: (playing: boolean) => void;
  speed: number;
  setSpeed: (speed: number) => void;
  horizon: number;
}) {
  return (
    <div className="playback">
      <div className="playback-top">
        <div className="playback-buttons">
          <button
            className="play-button"
            aria-label={playing ? "Pause mission" : "Play mission"}
            onClick={() => {
              if (step >= horizon) setStep(0);
              setPlaying(!playing);
            }}
          >
            {playing ? (
              <Pause size={16} fill="currentColor" />
            ) : (
              <Play size={16} fill="currentColor" />
            )}
          </button>
          <button
            className="icon-button"
            aria-label="Restart mission"
            onClick={() => {
              setStep(0);
              setPlaying(false);
            }}
          >
            <RotateCcw size={16} />
          </button>
          <button
            className="icon-button"
            aria-label="Advance one step"
            onClick={() => {
              setStep(Math.min(horizon, step + 1));
              setPlaying(false);
            }}
          >
            <SkipForward size={16} />
          </button>
          <span className="timecode">
            STEP <strong>{String(step).padStart(3, "0")}</strong> / {horizon}
          </span>
        </div>
        <label className="speed-control">
          Playback
          <select
            aria-label="Playback speed"
            value={speed}
            onChange={(e) => setSpeed(Number(e.target.value))}
          >
            {[1, 2, 4, 8].map((s) => (
              <option key={s} value={s}>
                {s}×
              </option>
            ))}
          </select>
        </label>
      </div>
      <input
        className="timeline-slider"
        type="range"
        aria-label="Mission timeline"
        min="0"
        max={horizon}
        value={step}
        onChange={(e) => {
          setStep(Number(e.target.value));
          setPlaying(false);
        }}
      />
      <div className="timeline-labels">
        <span>Launch</span>
        <span>One shared charger. Three independent batteries.</span>
        <span>Mission end</span>
      </div>
    </div>
  );
}

function DronePanel({
  replay,
  frame,
  selected,
  select,
}: {
  replay: Replay;
  frame: Frame;
  selected: number | null;
  select: (drone: number | null) => void;
}) {
  return (
    <section className="telemetry">
      <div className="section-caption">
        <span>FLEET TELEMETRY</span>
        <span>{3 - frame.lost.filter(Boolean).length}/3 online</span>
      </div>
      {frame.positions.map(([r, c], d) => {
        const atBase = r === replay.base[0] && c === replay.base[1],
          distance =
            Math.abs(r - replay.base[0]) + Math.abs(c - replay.base[1]),
          margin = frame.batteries[d] - distance * replay.config.move_cost;
        const status = frame.lost[d]
          ? "Lost"
          : frame.charger_owner === d
            ? "Charging"
            : atBase
              ? frame.batteries[d] < replay.config.battery_capacity
                ? "Waiting for charger"
                : "At base"
              : margin <= 3 * replay.config.move_cost
                ? "Critical return margin"
                : "In flight";
        return (
          <button
            className={`drone-card ${selected === d ? "selected" : ""}`}
            key={d}
            onClick={() => select(selected === d ? null : d)}
            style={{ "--drone-colour": colours[d] } as React.CSSProperties}
            aria-pressed={selected === d}
          >
            <div className="drone-heading">
              <span className="drone-symbol">
                <Radar size={18} />
              </span>
              <strong>Drone 0{d + 1}</strong>
              <span className={`drone-status ${frame.lost[d] ? "lost" : ""}`}>
                {status}
              </span>
            </div>
            <div className="battery-line">
              <span>
                <BatteryCharging size={13} /> Battery
              </span>
              <strong>
                {frame.batteries[d].toFixed(0)}{" "}
                <small>/ {replay.config.battery_capacity}</small>
              </strong>
            </div>
            <div className="battery-track">
              <span
                style={{
                  width: `${(frame.batteries[d] / replay.config.battery_capacity) * 100}%`,
                }}
              />
            </div>
            <div className="drone-detail">
              <span>
                Position{" "}
                <b>
                  {r}, {c}
                </b>
              </span>
              <span>
                Command <b>{commandNames[frame.commands[d]]}</b>
              </span>
            </div>
            <div className="drone-detail">
              <span>
                Return margin <b>{margin.toFixed(1)} units</b>
              </span>
              <span>
                Distance <b>{distance} cells</b>
              </span>
            </div>
          </button>
        );
      })}
      <p className="micro-copy">
        Select a drone to follow its path. Return margin is battery minus the
        direct flight cost to base; it is not a safety guarantee.
      </p>
    </section>
  );
}

function CellInspector({
  replay,
  frame,
  cell,
  setCell,
}: {
  replay: Replay;
  frame: Frame;
  cell: number;
  setCell: (cell: number) => void;
}) {
  const n = replay.config.grid_size,
    row = Math.floor(cell / n),
    col = cell % n;
  return (
    <div className="cell-inspector">
      <div className="section-caption">
        <span>CELL INSPECTOR</span>
        <Target size={13} />
      </div>
      <div className="cell-coordinates">
        <label>
          Row
          <input
            aria-label="Inspect cell row"
            type="number"
            min="0"
            max={n - 1}
            value={row}
            onChange={(e) =>
              setCell(
                Math.min(
                  n - 1,
                  Math.max(0, Math.trunc(Number(e.target.value))),
                ) *
                  n +
                  col,
              )
            }
          />
        </label>
        <label>
          Column
          <input
            aria-label="Inspect cell column"
            type="number"
            min="0"
            max={n - 1}
            value={col}
            onChange={(e) =>
              setCell(
                row * n +
                  Math.min(
                    n - 1,
                    Math.max(0, Math.trunc(Number(e.target.value))),
                  ),
              )
            }
          />
        </label>
        <span className={frame.burning[cell] ? "fire-text" : ""}>
          {frame.burning[cell]
            ? "Burning"
            : frame.fuel[cell] === 0
              ? "Fuel depleted"
              : "Not burning"}
        </span>
      </div>
      <div className="cell-stats">
        <span>
          Fuel <b>{frame.fuel[cell]}</b>
        </span>
        <span>
          Priority <b>{replay.priority[cell]}×</b>
        </span>
        <span>
          Inspection age <b>{frame.ages[cell]} steps</b>
        </span>
      </div>
    </div>
  );
}

function MapPanel({
  replay,
  step,
  layers,
  drone,
  onCell,
  compact,
}: {
  replay: Replay;
  step: number;
  layers: Layers;
  drone: number | null;
  onCell: (cell: number) => void;
  compact?: boolean;
}) {
  const frame = frameAt(replay, step),
    holder = useRef<HTMLDivElement>(null);
  return (
    <div className="map-panel" ref={holder}>
      <div className="map-caption">
        <div>
          <span className="small-dot" />
          {layers.freshness ? "INSPECTION FRESHNESS" : "FIRE & TERRAIN"}
          <span className="caption-separator">/</span>
          {policyLabel(replay.policy, replay.training_seed)}
        </div>
        <button
          className="icon-button"
          aria-label="Expand map"
          onClick={() => {
            void holder.current?.requestFullscreen();
          }}
        >
          <Maximize2 size={14} />
        </button>
      </div>
      <div className="map-wrap">
        <MissionMap
          replay={replay}
          frame={frame}
          layers={layers}
          drone={drone}
          onCell={onCell}
          compact={compact}
        />
        <div className="map-compass">
          N<ArrowDown size={17} style={{ transform: "rotate(180deg)" }} />
        </div>
        <div className="map-scale">
          {replay.config.grid_size} × {replay.config.grid_size}
          <span>Illustrative grid</span>
        </div>
        <div className="wind-overlay">
          <Wind size={14} /> Wind · {windNames[frame.wind]}
        </div>
      </div>
      <div className="map-legend">
        <span>
          <i className={layers.freshness ? "legend-fresh" : "legend-fire"} />
          {layers.freshness ? "Fresh inspection" : "Active fire"}
        </span>
        <span>
          <i className="legend-priority" />
          High priority
        </span>
        <span>
          <i className="legend-base" />
          Shared base
        </span>
        <span>
          <i className="legend-sensor" />
          Sensor footprint
        </span>
      </div>
      {step > frame.step && (
        <div className="terminal-note">
          Fleet lost at step {frame.step}. The rest of the mission is unserved.
        </div>
      )}
    </div>
  );
}

function DecisionPanel({ frame }: { frame: Frame }) {
  const decision = frame.decision;
  const top = decision
    ? decision.allowed
        .map((a) => ({ action: a, score: decision.scores[a] }))
        .sort((a, b) => b.score - a.score || a.action - b.action)
        .slice(0, 4)
    : [];
  return (
    <div className="decision-panel">
      <div className="section-caption">
        <span>WHAT HAPPENED THIS STEP</span>
        <CircleHelp size={14} />
      </div>
      <p>
        {frame.events.length
          ? frame.events[0]
          : frame.step === 0
            ? "The fleet starts at base with full batteries. Every cell begins due for inspection."
            : "Surviving drones inspect their sensor footprints. Inspection age resets there; the fire then evolves."}
      </p>
      {frame.events.slice(1).map((e, i) => (
        <p className="micro-copy" key={i}>
          {e}
        </p>
      ))}
      {decision && (
        <details>
          <summary>
            {decision.allowed.length} allowed actions · inspect the decision
          </summary>
          <p className="micro-copy">
            Recorded{" "}
            {decision.kind === "q_values" ? "DQN Q estimates" : "actor logits"}{" "}
            for the state before this transition. The selected action had the
            highest allowed score; first index breaks ties. Scores do not
            explain the network’s internal reasoning.
          </p>
          <div className="score-list">
            {top.map((t) => (
              <div key={t.action}>
                <span>
                  Action {t.action}
                  {t.action === frame.action ? " · selected" : ""}
                </span>
                <b>{t.score.toFixed(4)}</b>
              </div>
            ))}
          </div>
        </details>
      )}
      <details>
        <summary>
          Reward breakdown <b>{frame.reward.toFixed(4)}</b>
        </summary>
        <div className="score-list">
          {Object.entries(frame.reward_components).map(([key, value]) => (
            <div key={key}>
              <span>{key.replaceAll("_", " ")}</span>
              <b>{value.toFixed(4)}</b>
            </div>
          ))}
        </div>
        <p className="micro-copy">
          Inspection metrics and reward use the hazard priorities before fire
          evolution; this map shows the resulting state.
        </p>
      </details>
    </div>
  );
}

function MissionExplorer({
  evidence,
  overview,
  navigate,
}: {
  evidence: Evidence;
  overview: boolean;
  navigate: (page: Page) => void;
}) {
  const [left, setLeft] = useState("greedy:none"),
    [right, setRight] = useState("dqn:0"),
    [world, setWorld] = useState(22000),
    [compare, setCompare] = useState(!overview);
  const [focus, setFocus] = useState<"left" | "right">("left");
  const [step, setStep] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(2),
    [drone, setDrone] = useState<number | null>(null),
    [cell, setCell] = useState(820);
  const [layers, setLayers] = useState<Layers>({
    freshness: false,
    priority: true,
    sensors: true,
    trails: true,
  });
  const a = useReplay(evidence, left, world),
    b = useReplay(evidence, right, world),
    ready = !!a.replay && (!compare || !!b.replay);
  useEffect(() => {
    setPlaying(false);
    setStep(0);
  }, [left, right, world]);
  useEffect(() => {
    if (!playing || !ready) return;
    const timer = window.setInterval(
      () =>
        setStep((s) => {
          if (s >= 300) {
            setPlaying(false);
            return 300;
          }
          return s + 1;
        }),
      250 / speed,
    );
    return () => clearInterval(timer);
  }, [playing, speed, ready]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).matches("input,select,textarea,button"))
        return;
      if (e.code === "Space") {
        e.preventDefault();
        setPlaying((p) => !p);
      }
      if (e.code === "ArrowRight") {
        setPlaying(false);
        setStep((s) => Math.min(300, s + 1));
      }
      if (e.code === "ArrowLeft") {
        setPlaying(false);
        setStep((s) => Math.max(0, s - 1));
      }
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, []);
  const primary = compare && focus === "right" ? b.replay : a.replay,
    frame = primary && frameAt(primary, step);
  return (
    <>
      <Intro
        eyebrow={
          overview
            ? "A WILDFIRE INSPECTION STUDY"
            : "WATCH THE CONTROLLERS AT WORK"
        }
        title={
          overview
            ? "A changing frontier.\nThree drones. One mission."
            : "Inside the mission."
        }
        text={
          overview
            ? "Keeping an eye on a wildfire is a coordination problem. Explore how a small, battery-limited fleet inspects a changing landscape—and what reinforcement learning actually achieved."
            : "Follow every move, recharge, and inspection. Compare controllers on the same world, then pause to understand a single decision."
        }
      >
        <Badge>Recorded study · verified replay</Badge>
      </Intro>
      {overview && (
        <div className="overview-strip">
          <div>
            <Leaf size={19} />
            <span>
              <strong>Inspect, then revisit.</strong> We know where the fire is.
              The task is to keep inspections fresh.
            </span>
          </div>
          <button className="text-button" onClick={() => navigate("research")}>
            How we got here <ArrowRight size={16} />
          </button>
        </div>
      )}
      <div className="mission-toolbar">
        <div className="selection-controls">
          <ControllerSelect
            value={left}
            onChange={setLeft}
            label="Controller"
          />
          {compare && (
            <ControllerSelect
              value={right}
              onChange={setRight}
              label="Compare with"
            />
          )}
          <label className="control-label">
            Scenario
            <select
              aria-label="Scenario"
              value={world}
              onChange={(e) => setWorld(Number(e.target.value))}
            >
              {evidence.worlds.map((w) => (
                <option key={w} value={w}>
                  World {w}
                </option>
              ))}
            </select>
          </label>
        </div>
        <button
          className={`outline-button ${compare ? "active" : ""}`}
          onClick={() => setCompare(!compare)}
        >
          <Layers3 size={15} />
          {compare ? "Single view" : "Compare controllers"}
        </button>
      </div>
      <div className="layer-controls">
        <div className="segmented">
          <button
            className={!layers.freshness ? "active" : ""}
            onClick={() => setLayers({ ...layers, freshness: false })}
          >
            <Flame size={13} />
            Fire & fuel
          </button>
          <button
            className={layers.freshness ? "active" : ""}
            onClick={() => setLayers({ ...layers, freshness: true })}
          >
            <Radar size={13} />
            Freshness
          </button>
        </div>
        <div className="layer-toggles">
          {(["priority", "sensors", "trails"] as const).map((k) => (
            <label key={k}>
              <input
                type="checkbox"
                checked={layers[k]}
                onChange={(e) =>
                  setLayers({ ...layers, [k]: e.target.checked })
                }
              />
              {k === "sensors"
                ? "Sensor footprints"
                : k === "priority"
                  ? "Priority zones"
                  : "Flight trails"}
            </label>
          ))}
        </div>
      </div>
      {a.error || (b.error && compare) ? (
        <div className="error-message" role="alert">
          {a.error || b.error}
        </div>
      ) : null}
      <div className={`mission-layout ${compare ? "comparing" : ""}`}>
        <div className="mission-stage">
          {ready && primary && frame ? (
            <>
              <div className={compare ? "comparison-maps" : ""}>
                <MapPanel
                  replay={a.replay!}
                  step={step}
                  layers={layers}
                  drone={!compare || focus === "left" ? drone : null}
                  onCell={(c) => {
                    setCell(c);
                    setFocus("left");
                  }}
                  compact={compare}
                />
                {compare && b.replay && (
                  <MapPanel
                    replay={b.replay}
                    step={step}
                    layers={layers}
                    drone={focus === "right" ? drone : null}
                    onCell={(c) => {
                      setCell(c);
                      setFocus("right");
                    }}
                    compact
                  />
                )}
              </div>
              {compare && (
                <div className="inspector-choice segmented">
                  <button
                    className={focus === "left" ? "active" : ""}
                    onClick={() => setFocus("left")}
                  >
                    Inspect{" "}
                    {policyLabel(a.replay!.policy, a.replay!.training_seed)}
                  </button>
                  <button
                    className={focus === "right" ? "active" : ""}
                    onClick={() => setFocus("right")}
                  >
                    Inspect{" "}
                    {policyLabel(b.replay!.policy, b.replay!.training_seed)}
                  </button>
                </div>
              )}
              <TimelineControls
                step={step}
                setStep={setStep}
                playing={playing}
                setPlaying={setPlaying}
                speed={speed}
                setSpeed={setSpeed}
                horizon={300}
              />
              <div className="mission-stats">
                <div>
                  <span>Current freshness</span>
                  <strong>
                    {percent(
                      step > frame.step ? 0 : frame.metrics.weighted_freshness,
                    )}
                  </strong>
                  <small>Priority-weighted inspection recency</small>
                </div>
                <div>
                  <span>Mission return so far</span>
                  <strong>{frame.cumulative_return.toFixed(2)}</strong>
                  <small>Includes energy and overdue costs</small>
                </div>
                <div>
                  <span>Safe returns</span>
                  <strong>
                    {frame.safe_returns.toString().padStart(2, "0")}
                  </strong>
                  <small>Return events, not a success rate</small>
                </div>
                <div>
                  <span>Drone losses</span>
                  <strong>
                    {frame.metrics.total_losses}
                    <em> / 3</em>
                  </strong>
                  <small>
                    {frame.metrics.total_losses
                      ? "Permanent losses"
                      : "Fleet intact at this step"}
                  </small>
                </div>
              </div>
              {compare && b.replay && (
                <section className="panel">
                  <div className="section-caption">
                    <span>INSPECTION QUALITY OVER THIS MISSION</span>
                    <span>Higher is fresher</span>
                  </div>
                  <LineChart
                    percent
                    label="Comparison of per-step inspection freshness"
                    series={[a.replay!, b.replay].map((r, i) => ({
                      name: policyLabel(r.policy, r.training_seed),
                      colour: colours[i],
                      values: Array.from({ length: 301 }, (_, s) =>
                        s < r.frames.length
                          ? r.frames[s].metrics.weighted_freshness
                          : 0,
                      ),
                    }))}
                  />
                  <p className="micro-copy">
                    These are individual recorded missions. The Results view
                    shows means across all test worlds and fitted seeds.
                  </p>
                </section>
              )}
            </>
          ) : (
            <div className="map-loading">
              <Radar size={36} />
              <strong>Loading a verified mission…</strong>
              <span>Checking the replay’s SHA-256 before playback.</span>
            </div>
          )}
        </div>
        {primary && frame && (
          <aside className="mission-inspector">
            <div className="inspector-label">
              Inspecting {policyLabel(primary.policy, primary.training_seed)}
            </div>
            <DronePanel
              replay={primary}
              frame={frame}
              selected={drone}
              select={setDrone}
            />
            <CellInspector
              replay={primary}
              frame={frame}
              cell={cell}
              setCell={setCell}
            />
            <DecisionPanel frame={frame} />
            <div className="evidence-stamp">
              <ShieldCheck size={19} />
              <div>
                <strong>Every step is traceable.</strong>
                <span>
                  State hashes, rewards, masks and summaries verified against
                  the frozen archive.
                </span>
              </div>
            </div>
          </aside>
        )}
      </div>
      {overview && (
        <div className="overview-bottom">
          <div>
            <span className="eyebrow">THE CENTRAL FINDING</span>
            <h2>
              Safe flight isn’t the same
              <br />
              as useful surveillance.
            </h2>
            <p>
              The feasibility rules kept the fleet alive. The learned
              controllers still inspected much less effectively than Greedy.
            </p>
            <button className="text-button" onClick={() => navigate("results")}>
              Explore the evidence <ArrowRight size={16} />
            </button>
          </div>
          <div className="finding-numbers">
            <div>
              <strong>
                {percent(
                  numeric(
                    evidence.summary.find((r) => r.policy === "greedy")!,
                    "weighted_freshness",
                  ),
                )}
              </strong>
              <span>Greedy mean freshness</span>
            </div>
            <div>
              <strong>
                {percent(
                  numeric(
                    evidence.summary.find((r) => r.policy === "dqn")!,
                    "weighted_freshness",
                  ),
                )}
              </strong>
              <span>DQN mean freshness</span>
            </div>
            <p>
              40×40 frozen study · full-horizon means. Freshness is a recency
              score, not the percentage of land covered.
            </p>
          </div>
        </div>
      )}
      <p className="source-note">
        <CheckCheck size={13} />
        39 curated replays · three worlds · nine fitted learners and four
        baselines. Aggregate findings use all 260 missions.{" "}
        <a
          href={sourceLink("docs/experiments/rl-main-2026-10-11/README.md")}
          target="_blank"
          rel="noreferrer"
        >
          View source evidence <ExternalLink size={11} />
        </a>
      </p>
    </>
  );
}

function Research({ navigate }: { navigate: (page: Page) => void }) {
  const stages = [
    {
      title: "Define the mission.",
      label: "01 / THE PROBLEM",
      text: "One central controller sees the complete fire, fuel, terrain, inspection age, wind, drone and charger state. Three drones must repeatedly inspect important cells, return on time, and share one charger.",
      detail:
        "This is a fully observable finite-horizon Markov decision process. The drones inspect known hazards; they do not discover a hidden fire map or extinguish the fire. Grid cells and energy units are illustrative, not calibrated real-world measurements.",
      path: "docs/design.md",
      icon: Mountain,
    },
    {
      title: "Set a meaningful reference.",
      label: "02 / THE BASELINES",
      text: "Random provides a sanity check. Sweep patrols partitions of the grid. Greedy chooses immediate inspection gain. A bounded planner samples possible futures.",
      detail:
        "Greedy, Sweep and Planner use battery-aware proposal rules. The planner has simulator-model access and independently sampled hypothetical futures; it is approximate, not an optimal policy. Its computation cost is part of the comparison.",
      path: "src/wild_drone/baselines.py",
      icon: Compass,
    },
    {
      title: "Try three ways to learn.",
      label: "03 / THE LEARNERS",
      text: "DQN estimates joint-action values. REINFORCE learns from complete episode returns. A2C combines an actor with an n-step value critic. Each controls the entire fleet.",
      detail:
        "They share a full-grid convolutional representation and 125 joint actions: five commands for each of three drones. REINFORCE uses a learned Monte Carlo baseline; A2C is synchronous and single-environment. Training budgets do not establish convergence.",
      path: "src/wild_drone/learning.py",
      icon: Activity,
    },
    {
      title: "Investigate the failures.",
      label: "04 / THE DIAGNOSTICS",
      text: "Early unmasked learners struggled with returning and recharging. Stability, exploration, imitation, representation and action-head experiments tested possible causes without hiding negative results.",
      detail:
        "Supervised probes test capability and generalization, not successful reinforcement learning. Broader centered-inspection training improved 20×20 development missions, but the 40×40 supervised pilot failed its full-mission gate. These are separate studies at preserved revisions.",
      path: "docs/experiments/scale-pilot-2026-10-11/README.md",
      icon: CircleHelp,
    },
    {
      title: "Separate safety from learning.",
      label: "05 / THE FEASIBILITY MASK",
      text: "The masked learners choose among battery- and collision-feasible actions. An emergency-return rule handles conflicting return proposals. Masked random receives the same mask.",
      detail:
        "A fair control is essential: if masked random also survives, survival cannot be attributed entirely to learned coordination. The rules are part of the controller, and no universal real-flight safety guarantee is claimed.",
      path: "src/wild_drone/masked_learning.py",
      icon: ShieldCheck,
    },
    {
      title: "Freeze the study. Follow the evidence.",
      label: "06 / THE MAIN COMPARISON",
      text: "Nine independent fits. Three algorithms, three seeds, 300 episodes each. Final checkpoints were evaluated on 20 new worlds, alongside four baselines: 260 missions.",
      detail:
        "No validation selection, supervised warm starts, reward shaping or extra outcome-driven fits. An independent audit replayed 75,911 mission actions. None of the nine fitted policies passed either complete performance gate. A completed negative study is still informative.",
      path: "docs/scale-rl-protocol.md",
      icon: CheckCheck,
    },
  ];
  return (
    <>
      <Intro
        eyebrow="FROM QUESTION TO EVIDENCE"
        title="How we got here."
        text="A reproducible simulator, three learning methods, and a series of experiments that changed our understanding—not just our scores."
      />
      <div className="research-intro">
        <div>
          <h2>
            Coordinate the inspection.
            <br />
            Respect the battery.
          </h2>
          <p>
            Moving earns no credit on its own. What matters is whether important
            places remain recently inspected, while the drones have enough
            energy to get home.
          </p>
        </div>
        <div className="mdp-flow">
          <span>Observe complete state</span>
          <ArrowRight size={16} />
          <span>Choose joint action</span>
          <ArrowRight size={16} />
          <span>Inspect & recharge</span>
          <ArrowRight size={16} />
          <span>Fire evolves</span>
          <RotateCcw size={16} />
        </div>
      </div>
      <div className="research-timeline">
        {stages.map(({ title, label, text, detail, path, icon: Icon }, i) => (
          <article key={title}>
            <div className="timeline-marker">
              <Icon size={21} />
            </div>
            <div>
              <span className="eyebrow">{label}</span>
              <h2>{title}</h2>
              <p>{text}</p>
              <details>
                <summary>
                  Go deeper <ChevronRight size={14} />
                </summary>
                <p>{detail}</p>
                <a
                  className="text-button"
                  href={sourceLink(path)}
                  target="_blank"
                  rel="noreferrer"
                >
                  Read the implementation / protocol <ExternalLink size={13} />
                </a>
              </details>
            </div>
            <span className="stage-number">0{i + 1}</span>
          </article>
        ))}
      </div>
      <div className="research-conclusion">
        <span className="eyebrow">WHAT THE STUDY ESTABLISHES</span>
        <h2>
          The fleet survived.
          <br />
          The learning advantage didn’t emerge.
        </h2>
        <p>
          The next question is how to improve inspection decisions—not how to
          make the results look better.
        </p>
        <button className="primary-button" onClick={() => navigate("results")}>
          Read the results <ArrowRight size={16} />
        </button>
      </div>
    </>
  );
}

function Results({ evidence }: { evidence: Evidence }) {
  const [metric, setMetric] = useState("weighted_freshness"),
    [algorithm, setAlgorithm] = useState("dqn"),
    [condition, setCondition] = useState("default"),
    [follow, setFollow] = useState("planner");
  const labels: Record<string, string> = {
    weighted_freshness: "Inspection freshness",
    return: "Mission return",
    overdue_fraction: "Overdue inspections",
    total_losses: "Drone losses",
  };
  const format = (value: number) =>
    metric === "weighted_freshness" || metric === "overdue_fraction"
      ? percent(value)
      : value.toFixed(2);
  const ordered = useMemo(
    () =>
      [...evidence.summary].sort(
        (a, b) =>
          numeric(b, "weighted_freshness") - numeric(a, "weighted_freshness"),
      ),
    [evidence],
  );
  const maxValue =
    Math.max(
      ...ordered.map(
        (r) => Math.abs(numeric(r, metric)) + numeric(r, metric + "_sd"),
      ),
    ) || 1;
  const followRows =
    follow === "planner"
      ? evidence.followups.planner
      : follow === "sensitivity"
        ? evidence.followups.sensitivity.filter(
            (r) => r.condition === condition,
          )
        : evidence.followups.rates;
  return (
    <>
      <Intro
        eyebrow="RESULTS, WITH THEIR LIMITS"
        title="What actually happened."
        text="The study is complete. A reliable learned inspection advantage has not been demonstrated. Explore the full evidence, including the follow-up experiments."
      >
        <Badge>Frozen study · 11 Oct 2026</Badge>
      </Intro>
      <div className="study-strip">
        <div>
          <strong>09</strong>
          <span>Independent fits</span>
        </div>
        <div>
          <strong>300</strong>
          <span>Episodes per fit</span>
        </div>
        <div>
          <strong>260</strong>
          <span>Test missions</span>
        </div>
        <div>
          <strong>75,911</strong>
          <span>Audited mission actions</span>
        </div>
      </div>
      <section className="results-main panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">THE 40×40 MAIN COMPARISON</span>
            <h2>
              Safety came from the rules.
              <br />
              Inspection still needs better decisions.
            </h2>
          </div>
          <label className="control-label">
            Metric
            <select
              aria-label="Result metric"
              value={metric}
              onChange={(e) => setMetric(e.target.value)}
            >
              {Object.entries(labels).map(([value, name]) => (
                <option key={value} value={value}>
                  {name}
                </option>
              ))}
            </select>
          </label>
        </div>
        <div className="bars">
          {ordered.map((row) => (
            <div className="bar-row" key={String(row.policy)}>
              <span>{policyNames[String(row.policy)]}</span>
              <div
                className={`bar-track ${metric === "return" ? "negative-bars" : ""}`}
              >
                <div
                  className={`bar-fill ${row.policy === "greedy" ? "greedy" : ""}`}
                  style={{
                    width: `${(Math.abs(numeric(row, metric)) / maxValue) * 100}%`,
                  }}
                />
                <div
                  className="bar-uncertainty"
                  style={{
                    left: `${metric === "return" ? 100 - (Math.abs(numeric(row, metric)) / maxValue) * 100 : (Math.abs(numeric(row, metric)) / maxValue) * 100}%`,
                    width: `${(numeric(row, metric + "_sd") / maxValue) * 200}%`,
                  }}
                />
              </div>
              <strong>
                {format(numeric(row, metric))}
                <small> ± {format(numeric(row, metric + "_sd"))}</small>
              </strong>
            </div>
          ))}
        </div>
        <p className="chart-explanation">
          {metric === "return" && (
            <>
              Zero is at the right; negative returns extend left. Higher, closer
              to zero, is better.{" "}
            </>
          )}
          Bars show full-mission means; whiskers show sample standard deviation.
          RL variation is across <strong>three fitted-seed means</strong>, each
          averaged over 20 worlds. Baseline variation is across{" "}
          <strong>20 worlds</strong>. These are different uncertainty units, not
          confidence intervals.
        </p>
        <div className="result-callout">
          <ShieldCheck size={23} />
          <div>
            <strong>
              Zero losses is a safety result. It is not evidence of learned
              coordination.
            </strong>
            <p>
              Masked random also preserved all three drones. Every fitted
              learner failed both the complete learned-contribution gate and the
              Greedy-relative gate.
            </p>
          </div>
        </div>
      </section>
      <section className="panel learning-panel">
        <div className="panel-heading">
          <div>
            <span className="eyebrow">THE TRAINING RECORD</span>
            <h2>Learning curves, without a victory lap.</h2>
          </div>
          <label className="control-label">
            Algorithm
            <select
              aria-label="Training algorithm"
              value={algorithm}
              onChange={(e) => setAlgorithm(e.target.value)}
            >
              {["dqn", "reinforce", "a2c"].map((a) => (
                <option value={a} key={a}>
                  {policyNames[a]}
                </option>
              ))}
            </select>
          </label>
        </div>
        <LineChart
          label={`${algorithm} training return for all three seeds`}
          series={evidence.training
            .filter((t) => t.policy === algorithm)
            .map((t) => ({
              name: `Seed ${t.seed}`,
              values: t.rows.map((r) => r.return),
            }))}
        />
        <p className="micro-copy">
          Raw per-episode training return, episodes 1–300 (chart index 0 is
          episode 1). Hover to inspect. Improvements in training return alone do
          not establish held-out competence or convergence.
        </p>
      </section>
      <section className="followup-section">
        <Intro
          eyebrow="FOLLOW-UP REFERENCE EXPERIMENTS"
          title="What else did we check?"
          text="These follow-ups are separate from the frozen main comparison. Their budgets, worlds, and limitations stay visible."
        />
        <div className="segmented follow-tabs">
          {[
            ["planner", "Planning"],
            ["sensitivity", "Environment sensitivity"],
            ["rates", "Learning rates"],
          ].map(([key, title]) => (
            <button
              key={key}
              className={follow === key ? "active" : ""}
              onClick={() => setFollow(key)}
            >
              {title}
            </button>
          ))}
        </div>
        <div className="panel">
          <Badge amber>Follow-up experiment</Badge>
          <h2>
            {follow === "planner"
              ? "Does looking ahead help?"
              : follow === "sensitivity"
                ? "What changes when the environment changes?"
                : "Did the learning rate explain the gap?"}
          </h2>
          <p className="body-copy">
            {follow === "planner"
              ? "At the default settings, the bounded planner matched Greedy’s freshness but had a slightly worse mean return. The same-machine serial benchmark found substantially higher decision cost. Model access did not deliver a clear default-setting quality advantage."
              : follow === "sensitivity"
                ? "Fixed policies were evaluated under changed battery, recharge and spread settings, with no retraining. Greedy and Planner remained strongest in inspection quality; four Greedy fleet failures outside the default revealed a return deadlock."
                : "Nine fits tested three rates per algorithm, using one training seed and 20 validation worlds. None closed the gap to Greedy. Rate ordering is suggestive, not reliable evidence; these fits do not reproduce the archived Linux training trajectories."}
          </p>
          {follow === "sensitivity" && (
            <label className="control-label condition-control">
              Condition
              <select
                aria-label="Sensitivity condition"
                value={condition}
                onChange={(e) => setCondition(e.target.value)}
              >
                {[
                  ...new Set(
                    evidence.followups.sensitivity.map((r) =>
                      String(r.condition),
                    ),
                  ),
                ].map((c) => (
                  <option key={c} value={c}>
                    {c.replaceAll("_", " ")}
                  </option>
                ))}
              </select>
            </label>
          )}
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Controller</th>
                  {follow === "rates" && <th>Learning rate</th>}
                  <th>Freshness ↑</th>
                  <th>Return ↑</th>
                  <th>Losses ↓</th>
                  <th>Aggregation</th>
                </tr>
              </thead>
              <tbody>
                {followRows.map((r, i) => (
                  <tr key={i}>
                    <td>{policyNames[String(r.policy)] || r.policy}</td>
                    {follow === "rates" && (
                      <td>{r.learning_rate || "Baseline"}</td>
                    )}
                    <td>{percent(numeric(r, "weighted_freshness_mean"))}</td>
                    <td>{numeric(r, "return_mean").toFixed(2)}</td>
                    <td>{numeric(r, "total_losses_mean").toFixed(2)}</td>
                    <td>
                      {follow === "rates"
                        ? "20 worlds · one fit per rate"
                        : r.unit === "seed_means"
                          ? "3 fitted-seed means"
                          : "20 worlds"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <a
            className="text-button"
            href={sourceLink("docs/experiments/rl-followups-2026-10-11/README.md")}
            target="_blank"
            rel="noreferrer"
          >
            Full findings and limitations <ExternalLink size={13} />
          </a>
        </div>
      </section>
      <section className="provenance panel">
        <span className="eyebrow">EVIDENCE YOU CAN FOLLOW</span>
        <h2>Recorded, checked, preserved.</h2>
        <p>
          Replay exports verify {evidence.verification.steps.toLocaleString()}{" "}
          selected transitions. The complete independent study audit covers
          75,911 actions. Main-study chart means and sample SDs are recomputed
          from the 260 mission rows.
        </p>
        <div className="provenance-grid">
          <div>
            <span>Frozen source revision</span>
            <code>{evidence.source_revision.slice(0, 12)}</code>
          </div>
          <div>
            <span>Environment</span>
            <strong>40×40 · horizon 300 · battery 160</strong>
          </div>
          <div>
            <span>Selection rule</span>
            <strong>Final checkpoints · no validation selection</strong>
          </div>
          <div>
            <span>Observation</span>
            <strong>Full state · 125 joint actions</strong>
          </div>
        </div>
        <div className="provenance-links">
          <a href={`${import.meta.env.BASE_URL}data/evidence.json`} download>
            <Download size={14} />
            Download evidence index
          </a>
          <a
            href={sourceLink("Wildfire_Drone_Fleet.ipynb")}
            target="_blank"
            rel="noreferrer"
          >
            <BookOpen size={14} />
            Technical notebook
          </a>
          <a
            href={sourceLink("docs/experiments/rl-main-2026-10-11/README.md")}
            target="_blank"
            rel="noreferrer"
          >
            <ExternalLink size={14} />
            Complete archive
          </a>
        </div>
        <details>
          <summary>Definitions and limits</summary>
          <p>
            Freshness is weighted inspection recency, not area coverage. Early
            fleet loss pads the remaining mission with zero freshness and full
            overdue status. Survival does not mean every drone finished at base.
            Historical 20×20 and current 40×40 studies differ in settings and
            revisions; they do not isolate grid scaling. No 60×60 results or
            significance claims are made.
          </p>
        </details>
      </section>
    </>
  );
}

function Lab({ capabilities }: { capabilities: Capabilities | null }) {
  const [controller, setController] = useState("greedy"),
    [grid, setGrid] = useState(40),
    [world, setWorld] = useState(24000),
    [battery, setBattery] = useState(160),
    [recharge, setRecharge] = useState(8),
    [spread, setSpread] = useState(0.04),
    [job, setJob] = useState<string | null>(null),
    [status, setStatus] = useState(""),
    [error, setError] = useState(""),
    [replay, setReplay] = useState<Replay | null>(null),
    [step, setStep] = useState(0),
    [playing, setPlaying] = useState(false),
    [speed, setSpeed] = useState(2),
    [cell, setCell] = useState(820),
    [drone, setDrone] = useState<number | null>(null),
    [freshness, setFreshness] = useState(false);
  const last = useRef(-1),
    replayRef = useRef<Replay | null>(null);
  const available = capabilities?.controllers.find((c) => c.id === controller),
    isRL = available?.kind === "rl";
  useEffect(() => {
    if (!job) return;
    let active = true,
      timer = 0;
    const poll = async () => {
      try {
        const response = await fetch(
          `/api/missions/${job}?after_step=${last.current}`,
        );
        if (!response.ok) throw new Error("Mission status unavailable.");
        const result = await response.json();
        if (!active) return;
        setStatus(result.status);
        if (result.packet) {
          const previous = replayRef.current?.frames || [];
          const next = {
            ...result.packet,
            id: job,
            frames: [...previous, ...result.frames],
            summary: result.summary,
          } as Replay;
          replayRef.current = next;
          setReplay(next);
          if (result.frames.length)
            last.current = result.frames[result.frames.length - 1].step;
        }
        if (result.status === "failed") setError(result.error);
        if (result.status === "running" || result.status === "queued")
          timer = window.setTimeout(poll, 300);
      } catch (e) {
        if (active) {
          setError((e as Error).message);
          setStatus("failed");
        }
      }
    };
    void poll();
    return () => {
      active = false;
      clearTimeout(timer);
      void fetch(`/api/missions/${job}`, { method: "DELETE" });
    };
  }, [job]);
  useEffect(() => {
    if (!playing || !replay) return;
    const timer = window.setInterval(
      () =>
        setStep((s) => {
          if (s >= replay.frames.length - 1) {
            if (status === "completed" || status === "cancelled")
              setPlaying(false);
            return s;
          }
          return s + 1;
        }),
      250 / speed,
    );
    return () => clearInterval(timer);
  }, [playing, speed, replay, status]);
  const run = async () => {
    setError("");
    setStatus("queued");
    setReplay(null);
    replayRef.current = null;
    last.current = -1;
    setStep(0);
    setPlaying(false);
    try {
      const response = await fetch("/api/missions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          policy: controller,
          world,
          grid,
          battery,
          recharge,
          spread,
        }),
      });
      const result = await response.json();
      if (!response.ok)
        throw new Error(result.detail || "Could not start mission.");
      setJob(result.id);
    } catch (e) {
      setStatus("failed");
      setError((e as Error).message);
    }
  };
  const frame = replay?.frames.length ? frameAt(replay, step) : null;
  return (
    <>
      <Intro
        eyebrow="TRY A NEW MISSION"
        title={"Change a condition.\nWatch what follows."}
        text="A bounded sandbox using the actual Python simulator. These runs are explorations, kept separate from the published study."
      >
        <Badge amber>
          {capabilities
            ? "Local simulator connected"
            : "Local simulator required"}
        </Badge>
      </Intro>
      {!capabilities && (
        <div className="local-instructions panel">
          <FlaskConical size={26} />
          <div>
            <h2>The simulation lab runs on your machine.</h2>
            <p>
              This hosted showcase contains verified replays and results. To run
              new missions, launch the local Python service from your checkout
              and open its address.
            </p>
            <pre>
              python -m pip install -e '.[ui,rl]'{"\n"}cd web && npm ci && npm
              run build{"\n"}cd ..{"\n"}wild-drone-ui
            </pre>
            <p className="micro-copy">
              Open http://127.0.0.1:8000. Optional: add --checkpoint-root with
              the nine frozen fit directories. The hosted site does not attempt
              to contact your local machine.
            </p>
          </div>
        </div>
      )}
      <div className="lab-layout">
        <section className="lab-controls panel">
          <div className="section-caption">
            <span>MISSION SETTINGS</span>
            <SlidersHorizontal size={16} />
          </div>
          <label className="control-label">
            Controller
            <select
              aria-label="Sandbox controller"
              value={controller}
              disabled={
                !capabilities || status === "running" || status === "queued"
              }
              onChange={(e) => {
                const id = e.target.value;
                setController(id);
                if (
                  capabilities?.controllers.find((c) => c.id === id)?.kind ===
                  "rl"
                ) {
                  setGrid(40);
                  setBattery(160);
                  setRecharge(8);
                  setSpread(0.04);
                }
              }}
            >
              {(
                capabilities?.controllers ||
                ["greedy", "sweep", "random", "masked_random", "planner"].map(
                  (id) => ({ id, available: true, kind: "baseline" }),
                )
              ).map((c) => (
                <option key={c.id} value={c.id} disabled={!c.available}>
                  {c.id.includes("-")
                    ? policyLabel(
                        c.id.split("-")[0],
                        Number(c.id.split("-")[1]),
                      )
                    : policyLabel(c.id)}
                  {!c.available ? " · checkpoint missing" : ""}
                </option>
              ))}
            </select>
          </label>
          <label className="control-label">
            Grid preset
            <select
              aria-label="Grid preset"
              value={grid}
              disabled={isRL}
              onChange={(e) => {
                const g = Number(e.target.value);
                setGrid(g);
                setBattery(g === 20 ? 80 : 160);
                setRecharge(g === 20 ? 5 : 8);
                setCell(g === 20 ? 210 : 820);
              }}
            >
              <option value="20">20×20 · 160 steps</option>
              <option value="40">40×40 · 300 steps</option>
            </select>
          </label>
          <label className="control-label">
            Scenario seed
            <input
              aria-label="Scenario seed"
              type="number"
              min="0"
              max={2147483647}
              value={world}
              onChange={(e) => setWorld(Number(e.target.value))}
            />
          </label>
          {[
            {
              name: "Battery capacity",
              value: battery,
              min: 40,
              max: 240,
              step: 10,
              set: setBattery,
            },
            {
              name: "Recharge rate",
              value: recharge,
              min: 1,
              max: 16,
              step: 1,
              set: setRecharge,
            },
            {
              name: "Fire spread probability",
              value: spread,
              min: 0,
              max: 0.12,
              step: 0.01,
              set: setSpread,
            },
          ].map((control) => (
            <label className="lab-range" key={control.name}>
              <span>
                {control.name}
                <strong>{control.value}</strong>
              </span>
              <input
                type="range"
                min={control.min}
                max={control.max}
                step={control.step}
                value={control.value}
                disabled={isRL}
                onChange={(e) => control.set(Number(e.target.value))}
              />
            </label>
          ))}
          {isRL && (
            <p className="micro-copy">
              Frozen learners use their original 40×40 configuration. Explore
              changed environments with baselines or the archived sensitivity
              results.
            </p>
          )}
          <button
            className="primary-button"
            disabled={
              !capabilities ||
              !available?.available ||
              status === "running" ||
              status === "queued"
            }
            onClick={() => {
              void run();
            }}
          >
            <Play size={15} />
            Run mission
          </button>
          {(status === "running" || status === "queued") && (
            <button
              className="outline-button"
              onClick={() => {
                if (job)
                  void fetch(`/api/missions/${job}`, { method: "DELETE" });
              }}
            >
              Cancel mission
            </button>
          )}
          <p className="micro-copy">
            No training is launched. Planner uses its existing bounded defaults
            and may take longer.
          </p>
          {status && (
            <Badge amber>
              {status}{" "}
              {replay ? `· ${replay.frames.length - 1} steps generated` : ""}
            </Badge>
          )}
          {error && (
            <p className="error-message" role="alert">
              {error}
            </p>
          )}
        </section>
        <div>
          {replay && frame ? (
            <>
              <Badge amber>New sandbox run · not published evidence</Badge>
              <div className="layer-controls">
                <div className="segmented">
                  <button
                    className={!freshness ? "active" : ""}
                    onClick={() => setFreshness(false)}
                  >
                    Fire & fuel
                  </button>
                  <button
                    className={freshness ? "active" : ""}
                    onClick={() => setFreshness(true)}
                  >
                    Freshness
                  </button>
                </div>
              </div>
              <MapPanel
                replay={replay}
                step={step}
                layers={{
                  freshness,
                  priority: true,
                  sensors: true,
                  trails: true,
                }}
                drone={drone}
                onCell={setCell}
              />
              <TimelineControls
                step={step}
                setStep={setStep}
                playing={playing}
                setPlaying={setPlaying}
                speed={speed}
                setSpeed={setSpeed}
                horizon={replay.frames.length - 1}
              />
              <div className="lab-telemetry">
                <DronePanel
                  replay={replay}
                  frame={frame}
                  selected={drone}
                  select={setDrone}
                />
                <div>
                  <CellInspector
                    replay={replay}
                    frame={frame}
                    cell={Math.min(cell, replay.config.grid_size ** 2 - 1)}
                    setCell={setCell}
                  />
                  <DecisionPanel frame={frame} />
                  {replay.summary && (
                    <div className="panel">
                      <h3>Completed mission</h3>
                      <p>
                        Full-horizon freshness:{" "}
                        {percent(replay.summary.weighted_freshness)} · Return:{" "}
                        {replay.summary.return.toFixed(2)} · Losses:{" "}
                        {replay.summary.total_losses}/3
                      </p>
                      <button
                        className="text-button"
                        onClick={() => {
                          const a = document.createElement("a");
                          a.href = URL.createObjectURL(
                            new Blob([JSON.stringify(replay)], {
                              type: "application/json",
                            }),
                          );
                          a.download = `sandbox-${replay.world}-${replay.policy}.json`;
                          a.click();
                          URL.revokeObjectURL(a.href);
                        }}
                      >
                        Download this run <Download size={14} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </>
          ) : (
            <div className="lab-empty">
              <FlaskConical size={38} />
              <h2>Your next mission starts here.</h2>
              <p>
                Choose a controller and a world. Inspect the live-generated
                states as they arrive, or play the completed mission at your own
                pace.
              </p>
              <span>
                Same simulator. New conditions. Clearly separate evidence.
              </span>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

export default function App() {
  const initialPage = window.location.hash.slice(1) as Page;
  const [page, setPage] = useState<Page>(
      navigation.some((n) => n.id === initialPage) ? initialPage : "overview",
    ),
    [menu, setMenu] = useState(false),
    [evidence, setEvidence] = useState<Evidence | null>(null),
    [error, setError] = useState(""),
    [capabilities, setCapabilities] = useState<Capabilities | null>(null);
  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}data/evidence.json`)
      .then((r) => {
        if (!r.ok) throw new Error("Evidence could not be loaded.");
        return r.json();
      })
      .then(setEvidence)
      .catch((e) => setError(e.message));
    fetch("/api/capabilities")
      .then((r) => (r.ok ? r.json() : null))
      .then((c) => {
        if (c?.mode === "local") setCapabilities(c);
      })
      .catch(() => {});
    const update = () => {
      const next = window.location.hash.slice(1);
      if (navigation.some((n) => n.id === next)) setPage(next as Page);
    };
    window.addEventListener("hashchange", update);
    return () => window.removeEventListener("hashchange", update);
  }, []);
  const navigate = (next: Page) => {
    setPage(next);
    window.location.hash = next;
    setMenu(false);
    window.scrollTo({
      top: 0,
      behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches
        ? "auto"
        : "smooth",
    });
  };
  return (
    <div className="app-shell">
      <a href="#main" className="skip-link">
        Skip to content
      </a>
      <aside className={`sidebar ${menu ? "open" : ""}`}>
        <button
          className="brand"
          onClick={() => navigate("overview")}
          aria-label="Wild Drone home"
        >
          <div className="brand-mark">
            <Mountain size={28} />
            <span />
          </div>
          <div>
            WILD DRONE<span>APPEARED</span>
          </div>
        </button>
        <div className="workspace-label">RESEARCH / MISSION CONTROL</div>
        <nav aria-label="Main navigation">
          {navigation.map(({ id, icon: Icon, label }) => (
            <button
              key={id}
              className={`${page === id ? "active" : ""} ${id === "lab" ? "lab-nav" : ""}`}
              onClick={() => navigate(id)}
            >
              <Icon size={18} />
              <span>{label}</span>
              {page === id && <ChevronRight size={13} />}
              {id === "lab" && (
                <i className={capabilities ? "connected-dot" : "offline-dot"} />
              )}
            </button>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-study">
            <span className="small-dot" />
            <span>40×40 study complete</span>
          </div>
          <p>
            Three learning methods.
            <br />
            One honest comparison.
          </p>
          <a
            href={sourceLink("Wildfire_Drone_Fleet.ipynb")}
            target="_blank"
            rel="noreferrer"
          >
            <BookOpen size={15} />
            Technical notebook
            <ExternalLink size={12} />
          </a>
          <div className="sidebar-foot">
            GROUP 07<span>REINFORCEMENT LEARNING</span>
          </div>
        </div>
      </aside>
      {menu && (
        <button
          className="menu-backdrop"
          aria-label="Close menu"
          onClick={() => setMenu(false)}
        />
      )}
      <div className="main-shell">
        <header className="topbar">
          <div>
            <button
              className="mobile-menu icon-button"
              aria-label={menu ? "Close menu" : "Open menu"}
              onClick={() => setMenu(!menu)}
            >
              {menu ? <X size={20} /> : <Menu size={20} />}
            </button>
            <span className="topbar-project">Wildfire inspection</span>
            <ChevronRight size={12} />
            <span>{navigation.find((n) => n.id === page)?.label}</span>
          </div>
          <div>
            <span className="topbar-status">
              <i />
              {capabilities ? "Local + replay" : "Evidence showcase"}
            </span>
            <span className="topbar-grid">40 × 40</span>
          </div>
        </header>
        <main id="main" className={`content page-${page}`}>
          {error ? (
            <div className="error-message" role="alert">
              {error}
            </div>
          ) : !evidence ? (
            <div className="initial-loading">
              <Mountain size={44} />
              <h1>Preparing mission control.</h1>
              <p>Loading the experiment record.</p>
            </div>
          ) : page === "overview" || page === "mission" ? (
            <MissionExplorer
              key={page}
              evidence={evidence}
              overview={page === "overview"}
              navigate={navigate}
            />
          ) : page === "research" ? (
            <Research navigate={navigate} />
          ) : page === "results" ? (
            <Results evidence={evidence} />
          ) : (
            <Lab capabilities={capabilities} />
          )}
        </main>
        <footer className="footer">
          <span>
            <Mountain size={13} />
            WILD DRONE APPEARED
          </span>
          <span>
            Illustrative simulator · fully observable MDP · no real-flight
            safety claims
          </span>
          <span>Research, made visible.</span>
        </footer>
      </div>
    </div>
  );
}
