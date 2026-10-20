# NERV Maintenance Lab

**A working railway-maintenance ML mini project with discrete Soft Actor-Critic, a simulator, an interactive dashboard and measured baseline comparisons.**

Team: **Ananya Daitkar — 10846 · Lavanya Borse — 10839**

Research reference: Hanzhang Ge and colleagues, *Optimization of heavy railway maintenance machinery tamping scheduling based on deep reinforcement learning*, Applied Soft Computing (2026), article 115689. [Paper / DOI](https://doi.org/10.1016/j.asoc.2026.115689).

This is an **independent, scaled-down implementation inspired by the public description**, not an exact reproduction of the paper. It uses synthetic data and documented project assumptions. It is newly written software, not a copied or complete port of the original NERV repository.

## Run the finished demo

Use **Python 3.11**. A trained checkpoint and evaluation results are included; no training is needed to demonstrate the project.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements-lock.txt
python -m uvicorn railml.server:app --host 127.0.0.1 --port 8000
```

Windows: use `py -3.11 -m venv .venv`, then `.venv\Scripts\activate`.

Open **http://127.0.0.1:8000**. API documentation: **http://127.0.0.1:8000/docs**.

The application runs offline after dependency installation. It does not need an API key, cloud account or paid AI service. CPU inference is sufficient.

### Docker

```bash
docker compose up --build
```

Open the same address. Docker packaging is provided; see `docs/VERIFICATION.md` for what was actually tested.

## What you can demonstrate

- Run SAC, CP-SAT, a threshold rule and a no-maintenance control on identical scenarios.
- Replay every decision on a 50 km synthetic corridor with ten track units.
- Inspect feasible actions, actor probabilities, work duration, cost and reward.
- Change deterioration and unavailable-track probability; run a new seed.
- View training curves, validation checkpoints and held-out standard/stress results.
- Export the selected policy's schedule as CSV.

The SAC controller is a **trained neural network**, not a disguised rule. CP-SAT and the threshold controller are explicitly separate baselines.

## Documentation for the submission

- **[Exactly what was implemented](docs/IMPLEMENTATION.md)** — code map, environment, equations, SAC details, assumptions, limitations.
- **[Measured results](artifacts/RESULTS.md)** — actual test results, including weaknesses.
- **[Demo and viva guide](docs/DEMO.md)** — what to show and how to explain it.
- **[Verification](docs/VERIFICATION.md)** — tests and execution evidence.

## Reproduce training and evaluation

```bash
python -m railml.train --steps 20000 --seeds 0 1 2 --out output/reproduction
python -m railml.evaluate --artifacts output/reproduction
```

Training takes substantially longer than inference. Every run contains 20,000 simulator steps, not 20,000 episodes. Best checkpoints are selected using validation return only. Evaluation uses twenty held-out seeds in both standard and stress settings. See the report for how many completed runs are included in the shipped experiment.

The saved checkpoints contain actor weights for inference, not optimizer/replay state for resuming training. Retraining starts from scratch. To serve new results, copy the completed reproduction artifacts into `artifacts/` and restart the server. Preserve the original results if comparing experiments.

## Test

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Results must be interpreted honestly

SAC is not guaranteed to beat a well-designed rule or optimizer. Report return together with condition, overdue exposure and cost. Doing nothing costs zero but allows the track to deteriorate. A working learning pipeline and a reproducible, honest comparison are the academic contribution; there is no promise of full marks.

## Sources and attribution

- [Maintenance paper](https://doi.org/10.1016/j.asoc.2026.115689).
- [Paper-linked repository](https://github.com/chrozing/Heavy-Railway-Maintenance-Machinery-Tamping-Scheduling-based-on-Deep-Reinforcement-Learning): its README describes PPO, while the publication describes discrete SAC. Its source is not copied here.
- [Discrete SAC algorithm: Christodoulou, 2019](https://arxiv.org/abs/1910.07207).
- [Original NERV-railsync](https://github.com/PranayS909/NERV-railsync): project inspiration; its code was not copied. Original train-arbitration and multi-machine block-planning features are not included here.

Educational simulation only. Synthetic cost units and condition scores are not calibrated railway measurements. The simulator does not certify real railway safety.
