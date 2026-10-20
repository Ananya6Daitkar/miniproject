# Exactly what was implemented

**Project:** NERV Maintenance Lab
**Team:** Ananya Daitkar (10846), Lavanya Borse (10839)
**Research basis:** Ge et al. (2026), [DOI 10.1016/j.asoc.2026.115689](https://doi.org/10.1016/j.asoc.2026.115689)

## 1. What this project is

An executable deep-reinforcement-learning study of railway maintenance, with a trained policy, simulator, baselines, test experiments and browser interface. It is an independent implementation in the NERV maintenance domain. The destination repository was empty when implementation began. No original NERV source was copied.

The publicly available paper description identifies a maintenance MDP, discrete SAC and stochastic Gamma deterioration. This project implements those concepts. Full equations, exact hyperparameters and the original inspection dataset were unavailable; **all specific simulator parameters below are our assumptions, not claimed paper values**. The paper-linked repository describes a PPO baseline, so it was not treated as a SAC reproduction.

## 2. Where each component lives

| File | Implemented functionality |
|---|---|
| `railml/environment.py` | Seeded deterioration, observation, action feasibility, machine movement, maintenance, reward and episode traces |
| `railml/sac.py` | Convolutional actor, twin Q critics, target critics, replay memory, discrete SAC updates and checkpoint loading |
| `railml/train.py` | Simulator interaction, gradient updates, validation selection, CSV logs and checkpoint manifests |
| `railml/baselines.py` | CP-SAT, threshold rule, no-maintenance control and common rollout runner |
| `railml/evaluate.py` | Held-out tests, stress tests, raw CSV, statistical summary and demo-checkpoint selection |
| `railml/server.py` | Simulation, training-results, research and CSV-export API |
| `web/` | Interactive corridor, replay, schedule, probability inspector, graphs and comparisons |
| `tests/` | Environment, learning-update, checkpoint, optimizer and API tests |
| `artifacts/` | Actual trained weights, training logs, manifests and measured results |

## 3. Synthetic railway environment

- Ten units, each 5 km: a synthetic 50 km corridor.
- One maintenance-machine group; sixty sequential maintenance windows per episode.
- Initial condition independently sampled uniformly between 1.5 and 5.8. Lower is better.
- Deterioration rates sampled uniformly between 0.07 and 0.20 per window.
- Independent increments: `Gamma(shape=2, scale=rate/2)`. Their expected value is the segment's rate.
- At each window, the available duration is one of 75, 90 or 120 minutes.
- Each segment is unavailable with probability 0.15 in the standard setting.
- Tamping takes 45 minutes and changes condition to `0.35 × previous condition`, before deterioration.
- Travel speed is 0.6 km/min; distance is the difference between segment indices times 5 km.
- A condition above 6 is called overdue. This is a simulation threshold, not a railway standard.

Initial conditions, all deterioration increments, all windows and all unavailability events are sampled before policy execution. Policies cannot inspect future samples. Every method gets the same exogenous scenario for a given seed.

### State

Five channels over the ten units: normalized condition, normalized deterioration rate, one-hot machine position, feasible-service mask and relative travel distance. Remaining episode fraction and current window duration add two scalar features. Total: **52 observation features**.

### Actions

Eleven discrete actions: `0 = wait`; `1..10 = tamp that segment`. A service is feasible only when the track unit is available and `travel_time + 45 <= window_duration`. Wait is always feasible. Invalid actions are rejected before state mutation and have zero actor probability.

There is at most one tamping action per window. The machine stays at the serviced segment. There is no separate relocation-only action, multi-machine coordination, train timetable, crossing conflict or emergency dispatch model. An unavailable unit cannot be serviced; transit over intermediate unavailable units is not modeled. These are declared simplifications.

### Cost and reward

For a service action: `cost = 10 + 0.20 × travel_km`. Wait costs zero. Costs are synthetic units.

After action and deterioration:

```text
penalty = cost / 10
        + 0.50 × mean(track_condition)
        + 2.0 × sum(max(track_condition - 6, 0))
reward  = -penalty
```

This balances work cost, general condition and severity above the threshold. Return is the sum of rewards; a less negative value is better. Overdue exposure is also separately counted as the number of segment-windows above 6. Reward coefficients are project choices.

## 4. The actual ML implementation

Algorithm: **Discrete Soft Actor-Critic**, an off-policy deep reinforcement-learning method. No labeled schedules are used.

- Actor: maps the observation to eleven action logits.
- Two independently initialized critics estimate action values.
- Each network uses three 1D convolution layers, 16 channels each, kernel size 3, ReLU, followed by a 64-unit dense layer.
- The actor's infeasible logits are masked before softmax.
- Replay memory stores up to 50,000 transitions; minibatches have 128 samples.
- First 1,000 steps use random feasible actions. Subsequent training actions sample the actor distribution.
- Adam learning rate: 0.0003; discount: 0.97; entropy coefficient: fixed 0.10; target interpolation: 0.01.
- Gradients are clipped to norm 10. CPU execution uses one PyTorch thread.
- Evaluation chooses the highest-probability feasible action deterministically.

### SAC update equations

```text
V(next) = sum_a pi(a|next) × [min(Q1_target, Q2_target)(next,a) - alpha log pi(a|next)]
y       = reward + gamma × (1 - done) × V(next)
L_Q     = MSE(Q1(state,action), y) + MSE(Q2(state,action), y)
L_actor = mean_state sum_a pi(a|state) × [alpha log pi(a|state) - min(Q1,Q2)(state,a)]
Q_target <- (1 - tau) × Q_target + tau × Q
```

Only feasible actions carry probability mass. Terminal transitions do not bootstrap. The complete finite-horizon episode is treated as terminal. The target networks do not receive gradients.

**Checkpoint meaning:** `best.pt` contains actor parameters and metadata. `demo.pt` is the checkpoint selected on validation scores among completed runs. It is not a hand-written policy. Checkpoint hashes appear in manifests/report.

## 5. Baselines

1. **CP-SAT:** at each window, score feasible service/wait candidates using the current condition and known mean deterioration. Choose exactly one action minimizing expected one-window penalty. This is a deliberately myopic optimization baseline, not the original NERV scheduler or globally optimal multi-window planning.
2. **Threshold rule:** maintain the worst feasible segment when its condition is at least 4.5; otherwise wait.
3. **No maintenance:** always wait, a control illustrating deterioration without intervention.

All use the same state information, constraints, costs and scenario noise. CP-SAT is technically an integer optimizer over one-window candidates; for this small action space, direct enumeration would yield the same answer. Tests verify that equivalence.

## 6. Training and evaluation protocol

- Training seed 0 scenarios begin at 0; seed 1 at 20,000; seed 2 at 40,000.
- Twenty thousand interaction steps per requested run, approximately 333 completed episodes.
- Validation scenarios: 100000–100004; evaluation every 2,000 training steps.
- Best within-run checkpoint and final demo model are selected using validation return, never test scores.
- Test scenarios: 200000–200019, held out from training and validation.
- Standard suite: default environment.
- Stress suite: deterioration rates multiplied by 1.5 and unavailability probability increased to 0.30.
- Metrics: return, maintenance cost, mean condition, overdue segment-windows, travel, maintenance count and invalid actions.

Read `artifacts/evaluation.json` for the actual completed runs included. `evaluation.csv` contains every policy/run/scenario result. SAC mean and standard deviation pool runs × scenarios; they are descriptive statistics, not independent-sample confidence intervals. Per-training-seed results are also reported. Paired comparisons average SAC runs for each scenario before subtracting baseline return.

Zero invalid actions demonstrates enforcement of the simulator's limited constraints. It does not prove real-world safety or good maintenance outcomes.

## 7. Dashboard and API

The browser runs all four methods on the same requested scenario. It visualizes actual returned traces, not prewritten outcomes. Users can replay windows, inspect costs/durations/actions and export the selected schedule. Separate tabs expose actual training CSV logs, validation results, held-out results and methodology. The model is loaded from disk; if it is missing, the server returns an explicit error rather than silently substituting a rule.

## 8. What is not claimed

- No exact replication of the paper's numerical results, reward function or original field case study.
- No real-world deployment, cost calibration or railway safety certification.
- No guarantee that SAC beats CP-SAT or a simple rule, and no fabricated superiority claims.
- No automatic retraining button: training is deliberately an explicit CLI operation.
- No complete port of original NERV's train arbitration, block scheduling or multiple-machine features.

## 9. Academic contribution to explain

“We implemented an end-to-end discrete-SAC maintenance study, enforced simulator constraints, trained actual neural networks, evaluated held-out scenarios against non-learning baselines, and exposed the decisions and evidence in a reproducible dashboard. The simulation is a documented adaptation of the paper's public method.”

The strongest evidence is the code, saved weights, training logs, evaluation CSV and successful tests—not a claim of full marks.
