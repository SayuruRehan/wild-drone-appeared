import { describe, it } from "node:test";
import assert from "node:assert/strict";
import { frameAt, nearestMission } from "./data.ts";
import type { Frame, Mission, Replay } from "./types";

describe("shared playback timeline", () => {
  const frames = [0, 1, 2].map(
    (step) => ({ step, terminal: step === 2 }) as Frame,
  );
  const replay = { frames } as Replay;
  it("supports initial frame, seeking and terminal hold after early failure", () => {
    assert.equal(frameAt(replay, 0).step, 0);
    assert.equal(frameAt(replay, 1).step, 1);
    assert.equal(frameAt(replay, 300).step, 2);
    assert.equal(frameAt(replay, -1).step, 0);
  });
  it("selects controller, fitted seed and shared world independently", () => {
    const missions = [
      { policy: "dqn", seed: 0, world: 22000, id: "a" },
      { policy: "dqn", seed: 1, world: 22000, id: "b" },
      { policy: "greedy", seed: null, world: 22009, id: "c" },
    ] as Mission[];
    assert.equal(nearestMission(missions, "dqn", 1, 22000).id, "b");
    assert.equal(nearestMission(missions, "greedy", null, 22009).id, "c");
  });
});
