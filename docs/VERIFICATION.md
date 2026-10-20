# Verification record

Verified locally on macOS / Python 3.11 during implementation:

- `python -m pytest -q`: **12 passed**. One third-party Starlette/AnyIO deprecation warning.
- Three independent SAC runs completed, **20,000 interaction steps each**. Checkpoint manifests and raw training CSV files are included.
- `python -m railml.evaluate`: completed standard and stress evaluation, twenty held-out scenarios per suite for each model/baseline.
- Model hash, checkpoint round trip, terminal Bellman behavior, actor/critic/target updates, invalid-action rejection, scenario pairing, optimizer correctness, API validation and CSV response tested.
- Local Uvicorn server started successfully. Browser dashboard loaded, displayed computed policy comparisons, and training/results navigation was checked.
- Docker/Compose and a GitHub Actions workflow are supplied. The Docker image subsequently built and started successfully on Render free compute. The remote CI outcome was not verified.

See `artifacts/RESULTS.md` for measured performance. The threshold rule is a strong baseline; model quality varies across training seeds. Passing tests establishes implementation behavior, not superior predictive or control performance.
