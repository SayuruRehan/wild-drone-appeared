# Fixed-corpus supervised fitting-budget diagnostic

The coverage screen improved held-out decision agreement with broader teacher data,
but 44 of 45 final flights lost the fleet. Test whether more fitting of that corpus
helps before changing representation or using supervised weights to initialize RL.
This protocol fixes the budget before examining outcomes.

Collect one complete real Greedy corpus on training scenarios 0–99 and validation
teacher states on 5000–5004. Greedy can fail on training scenarios; retain its real
pre-action labels, with no restored battery or invented post-termination states.
All three fit seeds 0/1/2 receive the same fixed corpus from their first update.
Use the existing FleetNetwork, observation transform, 125 outputs, development physics,
unshielded argmax, Adam rate 0.001, batch size 64 sampled uniformly with replacement, and
gradient norm clip 0.5. Keep optimizer/sampler state throughout 20 rounds of 1,000 updates.
No weighting, new collection, architecture, reward, shield, or RL update belongs here.

Evaluate label metrics at initialization and each 1,000-update boundary. Save all
rounds with exact tensor/metadata audits and byte/model/data/source hashes. Compare
5,000/10,000/15,000/20,000-update autonomous policies on the same five validation
scenarios. These checkpoints share an optimization trajectory within each seed;
treat changes as paired budget comparisons, not extra independent fitted models.
The 5,000-update control receives the full corpus upfront, unlike the previous
progressive teacher arm, so previous runs are context and not extra replicates.

Retain the original joint-agreement-selected export separately: maximum validation
joint agreement among rounds 1–20, earliest ties, initialization ineligible. Selection
uses the same five scenarios and is biased. Its metric is not autonomous return.
Report selected policies separately from the budget checkpoints, with Greedy as one
shared five-scenario baseline: 80 autonomous missions. Do not retrospectively select
a budget and call its reused-validation performance a final generalization result.
Final test scenarios 10000+ remain unused.

Primary outcomes are mission return, full-fleet losses, freshness, safe returns,
completed charging and actually executed critical return moves. Critical teacher-state
agreement and charging holds are complementary diagnostics, not success criteria alone.
A useful outcome requires sustained inspection and return/recharge coordination;
higher training accuracy, fewer collisions, or survival alone is insufficient.

Use fresh output folders; no resume interface. Preserve earlier records and explicitly
supervised checkpoint identities. Average the five scenarios within each fit and
calculate mean/sample SD over the three seeds. Pair budget differences within seed.
Archive raw numerical/provenance records and collection audits; independently replay
flights and verify identities before CI and merging via sr→master.
