"""Local demo API. Training is an explicit CLI operation, never a hidden fallback."""
import csv
from functools import lru_cache
import io
import json
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import torch

from .baselines import run_episode
from .environment import Config
from .sac import SAC

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
torch.set_num_threads(1)
app = FastAPI(title="NERV Maintenance Lab", version="1.0.0")
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


class Scenario(BaseModel):
    seed: int = Field(default=200000, ge=100000, le=999999)
    horizon: int = Field(default=60, ge=10, le=120)
    deterioration: float = Field(default=1., ge=.5, le=2.)
    closures: float = Field(default=.15, ge=0., le=.6)


@lru_cache(maxsize=1)
def model():
    path = ARTIFACTS / "demo.pt"
    if not path.exists():
        raise HTTPException(503, "No trained model. Run training and evaluation first.")
    return SAC.load(path)


@app.get("/")
def index():
    return FileResponse(ROOT / "web/index.html")


@app.get("/api/health")
def health():
    return {"status": "ok", "trained_model": (ARTIFACTS / "demo.pt").exists()}


@app.get("/api/research")
def research():
    checkpoint = model()
    return {
        "title": "Optimization of heavy railway maintenance machinery tamping scheduling "
                 "based on deep reinforcement learning",
        "doi": "https://doi.org/10.1016/j.asoc.2026.115689",
        "status": "Independent, scaled-down implementation based on public paper descriptions",
        "metadata": checkpoint.metadata,
    }


@app.post("/api/simulate")
def simulate(scenario: Scenario):
    config = Config(horizon=scenario.horizon,
                    deterioration_multiplier=scenario.deterioration,
                    closure_probability=scenario.closures)
    trained = model()
    return {"scenario": scenario.model_dump(),
            "runs": {policy: run_episode(config, scenario.seed, policy, trained)
                     for policy in ("sac", "cp_sat", "threshold", "no_maintenance")}}


@app.get("/api/evaluation")
def evaluation():
    path = ARTIFACTS / "evaluation.json"
    if not path.exists():
        raise HTTPException(503, "No evaluation report. Run python -m railml.evaluate.")
    return json.loads(path.read_text())


@app.get("/api/training")
def training():
    runs = []
    for path in sorted(ARTIFACTS.glob("seed_*/training.csv")):
        with path.open() as f:
            rows = [{k: float(v) if v else None for k, v in row.items()}
                    for row in csv.DictReader(f)]
        runs.append({"seed": int(path.parent.name.split("_")[1]), "episodes": rows,
                     "validation": json.loads((path.parent / "validation.json").read_text())})
    return runs


@app.post("/api/export/{policy}")
def export(policy: Literal["sac", "cp_sat", "threshold", "no_maintenance"], scenario: Scenario):
    config = Config(horizon=scenario.horizon,
                    deterioration_multiplier=scenario.deterioration,
                    closure_probability=scenario.closures)
    result = run_episode(config, scenario.seed, policy, model())
    buffer = io.StringIO()
    fields = ["window", "action", "position", "next_position", "cost", "travel_km",
              "minutes_used", "available_minutes", "reward", "overdue", "mean_condition"]
    writer = csv.DictWriter(buffer, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(result["trace"])
    return Response(buffer.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": f'attachment; filename="{policy}-schedule.csv"'})
