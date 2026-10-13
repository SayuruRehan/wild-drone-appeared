"""Mission API isolation, cancellation, archived evidence and scientific accounting."""

import gzip
import hashlib
import json
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip('fastapi')
pytest.importorskip('httpx')
from fastapi.testclient import TestClient  # noqa: E402

from wild_drone.webapp.server import (  # noqa: E402
    MissionManager,
    MissionRequest,
    create_app,
    verified_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]


def completed(client, identity):
    for _ in range(500):
        result = client.get(f'/api/missions/{identity}').json()
        if result['status'] in ('completed', 'failed', 'cancelled'):
            return result
        time.sleep(.01)
    pytest.fail('Mission did not finish')


def test_seeded_jobs_are_isolated_and_incremental():
    with TestClient(create_app()) as client:
        request = {'policy': 'random', 'grid': 20, 'world': 24000, 'battery': 80,
                   'recharge': 5, 'spread': .04}
        first = client.post('/api/missions', json=request)
        second = client.post('/api/missions', json=request)
        assert first.status_code == second.status_code == 202
        assert first.json()['id'] != second.json()['id']
        a, b = [completed(client, r.json()['id']) for r in (first, second)]
        assert a['status'] == b['status'] == 'completed'
        assert a['frames'] == b['frames'] and a['summary'] == b['summary']
        assert a['frames'][0]['step'] == 0 and a['frames'][-1]['terminal']
        assert a['frames'][-1]['lost'] == [True]*3
        assert a['summary']['weighted_freshness'] == sum(
            f['metrics']['weighted_freshness'] for f in a['frames'][1:])/160
        after = client.get(f'/api/missions/{a["id"]}?after_step=10').json()
        assert all(f['step'] > 10 for f in after['frames'])
        assert a['frames'][0]['batteries'] == [80]*3


@pytest.mark.parametrize('payload', [
    {'grid': 60}, {'battery': 0}, {'spread': float('nan')},
    {'policy': 'made-up'}, {'policy': 'dqn-0'}, {'world': -1}, {'unknown': 1},
])
def test_invalid_or_unavailable_missions_do_not_start(payload):
    with TestClient(create_app()) as client:
        if payload.get('spread') != payload.get('spread'):
            response = client.post('/api/missions', content='{"spread": NaN}',
                                   headers={'Content-Type': 'application/json'})
        else:
            response = client.post('/api/missions', json=payload)
        assert response.status_code == 422
        assert client.app.state.missions.jobs == {}


def test_unknown_job_and_missing_checkpoint_capabilities(tmp_path):
    available, reasons = verified_checkpoints(tmp_path)
    assert not available and len(reasons) == 9
    bad = tmp_path/'dqn/seed-0/policy.pt'
    bad.parent.mkdir(parents=True)
    bad.write_bytes(b'wrong checkpoint')
    assert 'SHA-256' in verified_checkpoints(tmp_path)[1]['dqn-0']
    with TestClient(create_app(checkpoint_root=tmp_path)) as client:
        caps = client.get('/api/capabilities').json()
        assert sum(c['available'] for c in caps['controllers']) == 5
        assert client.get('/api/missions/unknown').status_code == 404
        assert client.delete('/api/missions/unknown').status_code == 404


def test_frozen_controller_rejects_environment_intervention():
    manager = MissionManager()
    manager.checkpoints['dqn-0'] = Path('unused.pt')
    try:
        with pytest.raises(ValueError, match='unchanged 40'):
            manager.create(MissionRequest(policy='dqn-0', battery=120))
        assert not manager.jobs
    finally:
        manager.close()


def test_cancellation_and_capacity(monkeypatch):
    entered, release = threading.Event(), threading.Event()

    class WaitingPolicy:
        def act(self, env, observation):
            entered.set()
            release.wait(3)
            return 0

    monkeypatch.setattr('wild_drone.webapp.server.make_policy', lambda *args: WaitingPolicy())
    manager = MissionManager(workers=1, capacity=1)
    try:
        identity = manager.create(MissionRequest())
        assert entered.wait(2)
        with pytest.raises(OverflowError):
            manager.create(MissionRequest())
        manager.cancel(identity)
        release.set()
        for _ in range(200):
            if manager.read(identity, -1)['status'] == 'cancelled':
                break
            time.sleep(.01)
        result = manager.read(identity, -1)
        assert result['status'] == 'cancelled' and result['summary'] is None
        assert len(result['frames']) <= 2
    finally:
        release.set()
        manager.close()


def test_expired_jobs_cleanup():
    manager = MissionManager(ttl=0)
    try:
        identity = manager.create(MissionRequest(policy='random', grid=20))
        for _ in range(500):
            if manager.read(identity, -1)['status'] == 'completed':
                break
            time.sleep(.01)
        manager.create(MissionRequest(policy='random', grid=20, world=24001))
        with pytest.raises(KeyError):
            manager.read(identity, -1)
    finally:
        manager.close()


def test_exported_replay_catalogue_is_complete_and_tamper_evident():
    folder = ROOT/'web/public/data'
    evidence = json.loads((folder/'evidence.json').read_text())
    assert len(evidence['missions']) == 39
    assert {m['world'] for m in evidence['missions']} == {22000, 22009, 22019}
    count = 0
    charges = losses = False
    for entry in evidence['missions']:
        raw = (folder/entry['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        replay = json.loads(gzip.decompress(raw))
        assert replay['verified'] and replay['frames'][0]['step'] == 0
        assert replay['frames'][-1]['terminal']
        assert replay['summary'] == entry['summary']
        count += replay['summary']['steps']
        charges |= replay['summary']['charge_completions'] > 0
        losses |= replay['summary']['total_losses'] > 0
    assert count == evidence['verification']['steps'] == 11389
    assert charges and losses
