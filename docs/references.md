# References and their role

1. Farama Foundation. [Create a Custom Environment](https://gymnasium.farama.org/introduction/create_custom_env/)
   and [environment API](https://gymnasium.farama.org/api/env/). Basis for reset/step/seeding
   and Gymnasium checks; no existing PettingZoo wildfire environment is claimed.
2. Mnih et al. (2015). [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236).
   Nature 518, 529–533, doi:10.1038/nature14236. DQN algorithm foundation.
3. Williams (1992). [Simple statistical gradient-following algorithms for connectionist reinforcement learning](https://doi.org/10.1007/BF00992696).
   Machine Learning 8, 229–256. REINFORCE foundation.
4. Mnih et al. (2016). [Asynchronous Methods for Deep Reinforcement Learning](https://proceedings.mlr.press/v48/mniha16.html).
   ICML/PMLR 48. n-step advantage actor–critic background. Our implementation is synchronous
   single-environment A2C, not the paper's asynchronous A3C architecture.
5. Viseras, Meissner and Marchal Gomez (2021). Wildfire front monitoring with multiple UAVs
   using deep Q-learning. IEEE Access. [doi:10.1109/ACCESS.2021.3055651](https://doi.org/10.1109/ACCESS.2021.3055651).
   Wildfire-fleet context from the approved proposal; not evidence that our synthetic dynamics
   are realistic or our fully observable setting will outperform planning.
6. Khoshdel, Luo and Afghah (2024). [PyroTrack](https://arxiv.org/abs/2403.11095).
   Belief-based deep reinforcement learning path planning for aerial wildfire monitoring in
   partially observable environments. Related work; this project's complete-state controller
   deliberately has a different observation assumption.
7. Julian and Kochenderfer (2019). [Distributed wildfire surveillance with autonomous aircraft using deep reinforcement learning](https://arxiv.org/abs/1810.04244).
   Journal of Guidance, Control, and Dynamics 42(8), 1768–1778. Related fleet/planning work,
   not a validation of this project's outcome or novelty.
8. van Hasselt, Guez and Silver (2016). [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461).
   AAAI. Basis for the optional Double DQN target: online action selection and target-network
   evaluation. Its reported overestimation reduction does not establish stability here.

The fire model is a deliberately simple stochastic cellular automaton. These references do
not calibrate its probabilities, physical cell size, wind or battery rates. Literature review
and measured/calibrated settings are still needed for practical deployment or publication.
