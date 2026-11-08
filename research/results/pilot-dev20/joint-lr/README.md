# Joint dueling DDQN learning-rate check (dev20, N = 3, seed 0, 150k steps)

Development freshness on worlds 30000–30009 during training (masked random ≈ 0.15,
factored learners ≈ 0.32 at 100k, Greedy 0.33):

| Steps | LR 1e-4 | LR 2.5e-4 (pilot, 2 seeds) | LR 5e-4 |
|---:|---:|---:|---:|
| 25k | 0.136 | 0.164 / 0.156 | 0.172 |
| 50k | 0.154 | 0.170 / 0.171 | 0.120 |
| 100k | 0.146 | 0.165 / 0.147 | 0.166 |
| 150k | 0.151 | – | 0.180 |

At three learning rates and up to 150k steps, the joint learner stays at masked-random level
while its TD loss grows. That makes "the joint-action head fails to learn at matched budget
and matched representation" a robust finding for H1. Whether it learns with a much longer
budget is the open question the 300k GPU run answers.
