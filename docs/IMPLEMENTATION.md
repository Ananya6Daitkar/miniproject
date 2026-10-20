# How the project was implemented

**Project:** NERV Maintenance Lab
**Team:** Ananya Daitkar (10846) and Lavanya Borse (10839)

## 1. Project idea in simple words

This project teaches a computer how to plan railway-track maintenance.

The computer controls one maintenance machine on a **simulated 50 km railway
corridor**. The corridor is divided into ten sections of 5 km each. During an
episode, the computer repeatedly decides whether to:

1. wait, or
2. send the machine to tamp one track section.

The computer is rewarded for keeping the track in good condition. It is
penalised when it travels too far, spends too much money, or allows the track
to become badly damaged.

This is an independent educational implementation inspired by the public
description of a railway-maintenance research paper. It is **not** an exact
copy of that paper and it does not use real railway inspection data.

## 2. What is included

| Location | Plain-language purpose |
|---|---|
| `railml/environment.py` | Creates the virtual railway and applies maintenance actions. |
| `railml/sac.py` | Contains the neural network and the learning method. |
| `railml/train.py` | Trains the neural network and saves the best model. |
| `railml/baselines.py` | Runs three comparison methods. |
| `railml/evaluate.py` | Tests all methods on new scenarios and creates reports. |
| `railml/server.py` | Provides the web dashboard and API. |
| `web/` | Contains the browser interface. |
| `tests/` | Checks that the environment, learning code, API and saved model work. |
| `artifacts/` | Contains saved model weights, logs and measured results. |

## 3. The simulated railway

The project does not connect to a real railway. It creates a small virtual
railway so that the learning process can be tested safely.

- The corridor is 50 km long.
- It has ten track sections, each 5 km long.
- One maintenance machine is available.
- Each episode contains 60 maintenance opportunities, called windows.
- A window gives the machine 75, 90 or 120 minutes.
- Tamping one section takes 45 minutes.
- Some sections may be unavailable. In the normal setting, this happens 15%
  of the time.
- A track condition above 6 is marked as overdue. This is only a project
  threshold; it is not an official railway safety limit.

The starting conditions, damage, available time and unavailable sections are
generated before a scenario starts. The computer cannot look into the future.
All methods receive the same scenario when they are compared.

## 4. What the computer sees and what it can do

At every maintenance window, the computer receives information such as:

- the current condition of every section,
- how quickly each section is getting worse,
- the machine's current position,
- which sections can be serviced now,
- the travel distance to each section,
- how much time remains in the episode, and
- how long the current window is.

There are eleven possible actions:

- action `0`: wait;
- actions `1` to `10`: tamp one of the ten sections.

The program removes impossible actions before the neural network chooses. For
example, it will not allow the machine to service an unavailable section or to
start work when there is not enough time.

## 5. How the neural network learns

The project uses **Discrete Soft Actor-Critic**, abbreviated as **SAC**.

In simple terms, SAC works like this:

1. The computer looks at the current railway condition.
2. It chooses an action, sometimes trying a less obvious action to explore.
3. The simulator applies the action and returns a reward.
4. The computer stores the experience.
5. Its neural network is updated so that better future decisions become more
   likely.
6. The process repeats many times.

The neural network has two main jobs:

- The **actor** suggests which action to take.
- The **critics** estimate how useful each action is.

The project uses two critics so that the computer is less likely to become too
confident about a poor action. A saved file such as `best.pt` contains the
trained actor network and its metadata. It is a real trained model, not a
hand-written list of maintenance rules.

## 6. How rewards are calculated

The simulator gives a negative penalty after each action. A good decision has
a smaller penalty, so its reward is closer to zero.

The penalty includes:

- the cost of travelling and doing maintenance,
- the average condition of the track, and
- an extra penalty for every section whose condition is above 6.

The **return** is the sum of all rewards in one episode. Because penalties are
subtracted, a return of `-100` is better than `-500`.

The costs and condition values are synthetic project values. They are useful
for comparing methods inside this simulator, but they are not real railway
prices or measurements.

## 7. Comparison methods

The trained SAC model is compared with three simpler methods:

1. **CP-SAT:** chooses the action with the lowest expected immediate penalty.
   It looks only one maintenance window ahead, so it is not a complete global
   railway planner.
2. **Threshold rule:** services the worst feasible section when its condition
   reaches 4.5; otherwise it waits.
3. **No maintenance:** always waits. This shows what happens when maintenance
   is never performed.

All four methods receive the same information and the same scenario noise.
This makes the comparison fairer.

## 8. Training and testing process

Three independent SAC training runs were made using seeds 0, 1 and 2.

- Each run used 20,000 interaction steps, not 20,000 full episodes.
- Each run completed about 333 episodes.
- Validation scenarios were used to select the best checkpoint during
  training.
- The final test scenarios were kept separate from training and validation.
- There were 20 held-out test scenarios in the normal setting and 20 in the
  stress setting.
- The stress setting makes deterioration faster and makes more track sections
  unavailable.

The checkpoint from training seed 1 was selected for the demo because it had
the best validation result among the completed runs.

The project does not report a normal classification accuracy percentage. This
is a decision-making problem, so it reports return, cost, track condition,
overdue exposure, travel and rule violations instead.

## 9. Dashboard and API

The browser dashboard can:

- run the SAC model,
- run the comparison methods,
- replay the maintenance decisions,
- show the selected action and its probability,
- show cost and reward for each step,
- display training and evaluation results, and
- export a schedule as CSV.

The server loads the saved model from disk. If the model is missing, it shows
an error instead of silently replacing the model with a rule.

## 10. What the results mean

The measured results are in [`artifacts/RESULTS.md`](../artifacts/RESULTS.md).
The short version is:

- SAC learned to make valid maintenance decisions.
- It recorded zero simulator constraint violations.
- It performed much better than doing no maintenance.
- In these experiments, the simple threshold rule and CP-SAT baseline had a
  better average return than SAC.
- Therefore, the project demonstrates a working, reproducible learning
  system, but it does not claim that SAC is the best method.

Zero rule violations means that the model followed the simulator's rules. It
does **not** mean the model is 100% accurate or safe for real railway use.

## 11. Important limitations

- The railway and all measurements are synthetic.
- The simulator is much smaller than a real railway network.
- There is only one machine.
- Train schedules, emergency work and multiple-machine coordination are not
  modeled.
- The reward values are project choices, not calibrated railway costs.
- The project is for education and research demonstration. It is not railway
  safety certification or a real-world deployment.

## 12. References

- Research paper: [DOI 10.1016/j.asoc.2026.115689](https://doi.org/10.1016/j.asoc.2026.115689)
- SAC reference: [Discrete SAC](https://arxiv.org/abs/1910.07207)
- Detailed machine-readable results: [`artifacts/evaluation.json`](../artifacts/evaluation.json)
