import { gunzipSync, strFromU8 } from "fflate";
import type { Mission, Replay } from "./types";

const cache = new Map<string, Replay>();
export async function loadReplay(mission: Mission): Promise<Replay> {
  if (cache.has(mission.id)) return cache.get(mission.id)!;
  const response = await fetch(`/data/${mission.file}`);
  if (!response.ok)
    throw new Error("This replay could not be loaded. Please try again.");
  const raw = new Uint8Array(await response.arrayBuffer());
  const digest = Array.from(
    new Uint8Array(await crypto.subtle.digest("SHA-256", raw)),
    (b) => b.toString(16).padStart(2, "0"),
  ).join("");
  if (digest !== mission.sha256)
    throw new Error(
      "Replay integrity check failed. Reload the latest showcase.",
    );
  const replay: Replay = JSON.parse(strFromU8(gunzipSync(raw)));
  if (cache.size >= 6) cache.delete(cache.keys().next().value!);
  cache.set(mission.id, replay);
  return replay;
}
export const frameAt = (replay: Replay, step: number) =>
  replay.frames[Math.min(Math.max(0, step), replay.frames.length - 1)];
export const nearestMission = (
  missions: Mission[],
  policy: string,
  seed: number | null,
  world: number,
) =>
  missions.find(
    (m) => m.policy === policy && m.seed === seed && m.world === world,
  )!;
