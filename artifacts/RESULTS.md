# Results in simple language

These numbers were produced by running:

```bash
python -m railml.evaluate
```

They are results from the project's **simulated railway**. They are not
results from a real railway and they are not the published results of the
research paper.

## Main results

| Test setting | Method | Overall return | Cost | Average condition | Overdue windows |
|---|---|---:|---:|---:|---:|
| Normal | SAC model | -448.59 ± 417.99 | 733.72 ± 27.47 | 3.12 ± 0.89 | 65.70 ± 68.33 |
| Normal | CP-SAT optimizer | -227.16 ± 18.98 | 206.90 ± 25.79 | 4.47 ± 0.08 | 89.60 ± 11.50 |
| Normal | Threshold rule | **-131.69 ± 4.50** | 341.75 ± 35.37 | 3.25 ± 0.07 | **0.50 ± 1.10** |
| Normal | No maintenance | -2747.18 ± 528.42 | 0.00 ± 0.00 | 7.58 ± 0.56 | 397.60 ± 43.76 |
| Difficult | SAC model | -738.08 ± 624.17 | 738.20 ± 24.62 | 3.81 ± 0.95 | 99.17 ± 74.41 |
| Difficult | CP-SAT optimizer | -288.72 ± 32.61 | 313.10 ± 37.18 | 4.61 ± 0.07 | 107.85 ± 9.23 |
| Difficult | Threshold rule | **-156.11 ± 14.04** | 462.90 ± 33.78 | 3.50 ± 0.11 | **4.85 ± 5.38** |
| Difficult | No maintenance | -4995.52 ± 773.56 | 0.00 ± 0.00 | 9.58 ± 0.71 | 466.85 ± 31.07 |

The `±` value shows how much the result varied between scenarios. It is a
descriptive spread, not a guarantee about future performance.

## How to read the table

### Return

Return is the combined result of the whole episode. It includes maintenance
cost and track-condition penalties. Because the project uses negative rewards,
a number closer to zero is better.

For example, `-100` is better than `-500`.

### Cost

This is the simulated cost of travel and maintenance. It is not a real money
amount.

### Average condition

Lower means the track was in better condition in this simulator.

### Overdue windows

This counts how often a track section went above the project's overdue
condition threshold. Lower is better.

## What can honestly be concluded?

- The SAC model was trained and produced a real neural-network policy.
- The model followed the simulator's rules. It recorded zero constraint
  violations in the evaluation summary.
- SAC was much better than doing no maintenance.
- The threshold rule and CP-SAT method had better average return than SAC in
  the included experiment.
- The threshold rule also produced the fewest overdue windows.
- Therefore, this project demonstrates a complete and reproducible learning
  pipeline. It does **not** prove that SAC is the best maintenance method.

## Is this an accuracy percentage?

No. This project is a decision-making problem, not a classification problem.
There is no single correct answer for every maintenance window, so a normal
accuracy percentage is not appropriate.

The most important performance number is the return, together with cost,
condition and overdue exposure. Zero constraint violations means “the model
obeyed the simulator rules”; it does not mean “the model is 100% accurate.”

## Test setup

- The SAC model was trained in three independent runs.
- Each run used 20,000 interaction steps.
- The final evaluation used 20 new scenarios for the normal setting and 20 new
  scenarios for the difficult setting.
- The demo checkpoint was selected from training seed 1 using validation
  results, not test results.
- All methods received the same scenario noise for a fair comparison.

For every individual scenario, see
[`evaluation.csv`](evaluation.csv). For the full machine-readable summary, see
[`evaluation.json`](evaluation.json).
