export type Config = {
  grid_size: number;
  horizon: number;
  battery_capacity: number;
  move_cost: number;
  recharge_rate: number;
  sensor_radius: number;
  inspection_deadline: number;
  max_fuel: number;
  spread_probability: number;
  [key: string]: number;
};
export type Metrics = {
  weighted_freshness: number;
  overdue_fraction: number;
  total_losses: number;
};
export type Frame = {
  step: number;
  burning: number[];
  fuel: number[];
  ages: number[];
  positions: number[][];
  batteries: number[];
  lost: boolean[];
  wind: number;
  charger_owner: number;
  commands: number[];
  action: number | null;
  metrics: Metrics;
  reward: number;
  reward_components: Record<string, number>;
  cumulative_return: number;
  energy_used: number;
  safe_returns: number;
  charge_completions: number;
  collision_attempts: number;
  events: string[];
  terminal: boolean;
  decision: { scores: number[]; allowed: number[]; kind: string } | null;
};
export type Summary = {
  return: number;
  weighted_freshness: number;
  overdue_fraction: number;
  total_losses: number;
  steps: number;
  [key: string]: number;
};
export type Replay = {
  id: string;
  kind: string;
  policy: string;
  world: number;
  training_seed: number | null;
  config: Config;
  base: number[];
  terrain: number[];
  priority: number[];
  frames: Frame[];
  summary: Summary | null;
  source_revision?: string;
  verified?: boolean;
};
export type Mission = {
  id: string;
  policy: string;
  seed: number | null;
  world: number;
  file: string;
  bytes: number;
  sha256: string;
  summary: Summary;
};
export type Row = Record<string, number | string>;
export type Evidence = {
  study: string;
  source_revision: string;
  config: Config;
  worlds: number[];
  missions: Mission[];
  summary: Row[];
  training: { policy: string; seed: number; rows: Record<string, number>[] }[];
  followups: { planner: Row[]; sensitivity: Row[]; rates: Row[] };
  verification: { missions: number; steps: number };
  sources: Record<string, { sha256: string; bytes: number }>;
};
export type Capabilities = {
  mode: string;
  controllers: {
    id: string;
    available: boolean;
    kind: string;
    reason?: string;
  }[];
};
export const colours = ["#8cdbc5", "#e6bd78", "#b0b9f9"];
export const policyNames: Record<string, string> = {
  greedy: "Greedy",
  sweep: "Sweep",
  random: "Random",
  masked_random: "Masked random",
  planner: "Planner",
  dqn: "DQN",
  reinforce: "REINFORCE",
  a2c: "A2C",
};
export const policyLabel = (policy: string, seed?: number | null) =>
  `${policyNames[policy] || policy}${seed != null ? ` · seed ${seed}` : ""}`;
export const percent = (value: number) => `${(value * 100).toFixed(1)}%`;
export const commandNames = ["Stay", "North", "South", "West", "East"];
