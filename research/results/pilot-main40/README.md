# Round-1 pilot on 40×40 (Colab T4) — joint, VDN, DCG (model_version 1), structured (+)

Environment `main40`, trained on 3 drones, 300k steps (structured / structured+ stopped at
200k / 175k), one seed, final checkpoint, evaluated on development worlds 30000–30019 at
N ∈ {3, 4, 5, 6, 8} (`table.md`). Code: the branch as cloned when the run started (before
the safety guard and snapshots; DCG is the unnormalised pilot version).

## Reading

1. **With 300k steps, learners match or beat Greedy on 40×40.** VDN (trained on 3 drones)
   is level with or above Greedy at every fleet size up to 8 (+0.005 to +0.017; −0.006 at
   N = 8). DCG beats Greedy at N = 3–4 (+0.026 / +0.029, 16/20 and 15/20 worlds) but
   transfers worse to 6–8 drones. On 20×20 with 100k steps, VDN fell 0.09 behind Greedy at
   N = 8, so budget matters a lot.
2. **The joint-action DQN fails again** (0.084, masked random 0.095), now at 300k steps on
   40×40 as well as on 20×20 at three learning rates.
3. **Assign remains the strongest controller at 3–6 drones**, so there is headroom left.
4. **Drone losses occurred** for structured+ at every N (9–64 over 20 worlds) and for VDN
   (N = 4, 8) and DCG (N = 5, 8), although the feasibility mask should make losses
   impossible and no baseline or fuzz test ever lost a drone. These evaluations ran on CPU,
   so non-finite GPU outputs are ruled out. Under investigation with
   `scripts/diagnose_losses.py --raw` on these exact checkpoints. **No safety claim may be
   made until this is explained and fixed.**
