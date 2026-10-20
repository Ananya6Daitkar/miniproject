# NERV Maintenance Lab: complete project explanation

This document explains the whole project from beginning to end in simple
language. It is intended for someone who has not studied machine learning or
railway scheduling before.

## Quick dictionary

| Project word | Easy meaning |
|---|---|
| Simulator | A computer-made railway used for practice. |
| Neural network | A computer program that changes its internal numbers while learning. |
| SAC | The name of the learning method used by the project. |
| Action | The choice made by the computer, such as wait or repair section 4. |
| Reward | Feedback after an action. Better decisions get better feedback. |
| Return | All rewards added together for one complete test. |
| Episode | One complete run from the first maintenance window to the last. |
| Baseline | A simpler method used for comparison. |
| Checkpoint | A saved copy of the trained model. |
| Validation | A separate check used to choose the best saved model. |
| Held-out test | A new scenario kept away from training so the final test is fair. |
| Seed | A number used to make random experiments repeatable. |

## 1. The problem this project tries to solve

Railway tracks slowly become worse over time. A maintenance machine is used
to repair the track, but the machine cannot repair every section at once.

The planner must decide:

- which section should be repaired,
- when the repair should happen,
- whether travelling to that section is worth the time and cost, and
- whether it is better to wait for a later opportunity.

The project uses a computer program to learn these decisions.

## 2. What is built in this project?

The project is a small virtual railway-maintenance laboratory. It contains:

1. a railway simulator;
2. a neural network that learns maintenance decisions;
3. three simpler comparison methods;
4. a testing and evaluation system; and
5. a web dashboard for viewing the decisions.

The project does not operate a real railway. It is an experiment that can be
run safely on a computer.

## 3. Step one: create the virtual railway

The simulator creates a 50 km railway corridor. It divides the corridor into
ten sections, with each section representing 5 km.

Each section has a condition value. A lower value means better condition. As
time passes, the condition becomes worse because of simulated deterioration.

The simulator also creates:

- a maintenance machine,
- a machine position,
- a limited amount of time in each maintenance window,
- travel distances,
- sections that are temporarily unavailable, and
- random deterioration events.

The project uses a fixed random seed when needed so that experiments can be
repeated.

## 4. Step two: give the computer a choice

At the start of each maintenance window, the computer sees the current state
of the railway. The state contains the information needed to make a decision:

- current condition of all ten sections;
- deterioration speed of each section;
- current machine location;
- which actions are possible;
- distance to the sections;
- remaining time in the episode; and
- duration of the current maintenance window.

The computer can choose one of eleven actions:

```text
0       wait
1       tamp section 1
2       tamp section 2
...
10      tamp section 10
```

The simulator checks whether an action is possible. For example, it rejects a
repair if the section is unavailable or if the machine cannot travel there
and finish the work in the available time.

## 5. Step three: calculate a reward

After the computer chooses an action, the simulator calculates a reward.

The computer is encouraged to:

- keep the track condition low;
- avoid allowing sections to become overdue; and
- avoid unnecessary travel and maintenance cost.

The computer is discouraged by penalties for:

- travelling and working;
- poor average track condition; and
- very poor condition above the overdue threshold.

The reward is usually negative because it represents a penalty. A reward of
`-5` is better than `-20`. Adding all rewards from one complete episode gives
the **return**.

Return is the main overall performance measure in this project. It is not a
percentage accuracy score.

## 6. Step four: train the SAC model

The learning model is called **Discrete Soft Actor-Critic**, or **SAC**.

SAC learns by repeated practice:

1. It observes the railway state.
2. It chooses a possible action.
3. The simulator changes the railway state.
4. The simulator gives a reward.
5. The model remembers what happened.
6. The model changes its internal weights to prefer better decisions in the
   future.

This process is repeated thousands of times. The saved neural-network weights
are the model's learned behaviour.

The actor part of the model suggests actions. The critic parts estimate how
useful those actions are. During final evaluation, the model chooses the
highest-probability valid action instead of choosing randomly.

## 7. Step five: train more than once

Machine learning can produce slightly different results depending on the
random starting point. To check this, the project trains three runs:

- training seed 0;
- training seed 1; and
- training seed 2.

Each run uses 20,000 interaction steps and completes about 333 episodes.

The project keeps a validation set separate from the training scenarios. The
best model from each run is selected using validation performance. The model
from seed 1 was selected for the demonstration.

The final test scenarios are not used for training or checkpoint selection.
This helps provide a fairer estimate of how the trained model behaves on new
scenarios.

## 8. Step six: compare with simpler methods

The trained SAC model is not tested alone. It is compared with three other
methods:

### CP-SAT

This method chooses the action with the best immediate estimated result. It is
an optimizer, not a neural network. It mainly looks one window ahead.

### Threshold rule

This method checks the worst available section. If its condition reaches a
chosen level, the method repairs it. Otherwise, it waits.

### No maintenance

This method always waits. It shows what happens when the railway is never
repaired.

All methods receive the same simulated scenario. This allows their results to
be compared fairly.

## 9. Step seven: test normal and difficult conditions

The project uses two test suites:

### Standard test

This uses the normal deterioration rate and a 15% probability that a section
is unavailable.

### Stress test

This makes the problem more difficult by:

- increasing deterioration by 50%; and
- increasing unavailable sections to 30%.

Each suite contains 20 held-out scenarios.

## 10. What results were obtained?

The average SAC results were:

| Situation | Return | Cost | Average condition | Overdue windows |
|---|---:|---:|---:|---:|
| Standard | -448.59 | 733.72 | 3.12 | 65.70 |
| Stress | -738.08 | 738.20 | 3.81 | 99.17 |

The results show that SAC learned to take valid maintenance actions and was
much better than the no-maintenance method. However, in this experiment, the
threshold rule and CP-SAT method achieved better average returns than SAC.

That is an important result: the project should not claim that the neural
network is the best method. It should claim that a complete, trainable and
reproducible SAC system was built and tested honestly.

## 11. Is there an accuracy score?

No traditional accuracy score is used here.

Accuracy is common when a model predicts a known answer, such as:

- spam or not spam;
- healthy or damaged; or
- cat or dog.

This project does not have one correct action for every situation. Several
maintenance actions may be reasonable, and the quality of an action depends on
its cost and its effect on future track condition.

Therefore, the project uses:

- **return:** overall result; closer to zero is better;
- **cost:** simulated maintenance and travel cost; lower is better;
- **mean condition:** average track condition; lower is better;
- **overdue windows:** how often track became too poor; lower is better; and
- **constraint violations:** whether the program broke simulator rules; zero
  is desired.

Zero constraint violations does not mean 100% accuracy. It means the model
followed the rules that were implemented in this simulator.

## 12. Step eight: show everything in the dashboard

The web dashboard loads the saved model and lets a user:

- run SAC and the comparison methods;
- view the current simulated track condition;
- replay decisions one window at a time;
- inspect action probabilities;
- view costs and rewards;
- inspect training and test results; and
- export the chosen schedule as a CSV file.

The dashboard does not use pre-written fake results. It asks the server to run
the methods and displays the returned simulation traces.

## 13. Important limitations

This project is a learning and demonstration system, not a real railway
control system.

- The data is synthetic.
- The railway is much smaller than a real railway network.
- Only one machine is modeled.
- Real train traffic and emergency work are not modeled.
- The costs are made-up simulator units.
- The condition threshold is a project setting, not a safety standard.
- The model has not been approved for real railway operations.

## 14. One-minute explanation

> This project uses a simulated 50 km railway with ten track sections and one
> maintenance machine. A reinforcement-learning model called SAC learns whether
> to wait or repair a section. It learns by trying actions, receiving rewards,
> and updating its neural network. The trained model is tested on new normal and
> difficult scenarios and compared with an optimizer, a threshold rule and a
> no-maintenance method. The project does not use a normal accuracy percentage;
> it uses return, cost, track condition, overdue exposure and rule violations.
> The model learned valid decisions, but the simpler comparison methods had a
> better average return in the included experiment. The system is an educational
> simulator and is not a real railway safety system.

## 15. Where to read next

- [`docs/IMPLEMENTATION.md`](IMPLEMENTATION.md): implementation details in
  simple language.
- [`artifacts/RESULTS.md`](../artifacts/RESULTS.md): measured results.
- [`docs/DEMO.md`](DEMO.md): demonstration and viva steps.
- [`docs/VERIFICATION.md`](VERIFICATION.md): tests and verification evidence.
