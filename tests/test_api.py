import hashlib
import json
from pathlib import Path
from fastapi.testclient import TestClient
from railml.server import app

client = TestClient(app)


def test_home_health_and_research():
    assert client.get('/').status_code == 200
    assert client.get('/api/health').json()['trained_model']
    assert client.get('/api/research').json()['metadata']['algorithm'] == 'discrete SAC'


def test_simulation_and_export():
    params = {'seed': 200001, 'horizon': 10, 'deterioration': 1., 'closures': .15}
    response = client.post('/api/simulate', json=params)
    assert response.status_code == 200
    runs = response.json()['runs']
    assert set(runs) == {'sac', 'cp_sat', 'threshold', 'no_maintenance'}
    for run in runs.values():
        assert len(run['trace']) == 10
        assert run['initial'] == runs['sac']['initial']
        assert all(t['mask'][t['action']] for t in run['trace'])
    export = client.post('/api/export/sac', json=params)
    assert export.status_code == 200
    assert len(export.text.strip().splitlines()) == 11


def test_bad_inputs_and_results():
    assert client.post('/api/simulate', json={'horizon': 9999}).status_code == 422
    assert client.post('/api/simulate', json={'seed': -1}).status_code == 422
    assert client.post('/api/export/unknown', json={}).status_code == 422
    report = client.get('/api/evaluation').json()
    assert len(report['test_seeds']) == 20
    assert len(client.get('/api/training').json()) >= 1
    root = Path(__file__).resolve().parents[1] / 'artifacts'
    assert hashlib.sha256((root / 'demo.pt').read_bytes()).hexdigest() == report['checkpoint_sha256']
    tests = set(report['test_seeds'])
    for manifest in report['training_runs']:
        start, end = manifest['training_seed_range']
        assert not tests.intersection(range(start, end + 1))
