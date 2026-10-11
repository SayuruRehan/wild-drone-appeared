# 40×40 prospective supervised inspection pilot

Frozen source: 7c9562aa6064f46d96cc9a163de3deb2ecaa3e19

- Ranking criterion all seeds: **True**
- Full-mission criterion all seeds: **False**
- Complete episodes: 140; steps: 42000
- Independent audit: {'passed': True, 'ranking_rows': 8943, 'mission_episodes': 140, 'mission_steps': 42000, 'all_action_rewards_checked': 214125, 'checkpoint_source_hashes_verified': True, 'simulator_replay': True, 'full_score_vector_reconstruction': True, 'score_reconstruction_atol': 1e-06, 'score_reconstruction_rtol': 0, 'ranking_max_absolute_score_drift': 5.960464477539062e-07, 'mission_max_absolute_score_drift': 0.0, 'reconstructed_feasible_argmax_exact': True}

This is supervised inspection scoring, **not trained reinforcement learning**. A failed predeclared criterion is a negative pilot, not an invitation to retune. Training data and checkpoint binaries are retained in the run artifact, while the exact binary hashes are tracked here. Confirmation and final test worlds were not used.
