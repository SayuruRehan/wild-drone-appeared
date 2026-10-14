"""Local-only simulator service with isolated, cancellable mission jobs."""

import argparse
import hashlib
import io
import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import replace
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from wild_drone.baselines import make_policy
from wild_drone.config import EnvConfig
from wild_drone.environment import WildfireFleetEnv
from wild_drone.webapp.state import initial_packet, mission_summary, snapshot

ROOT = Path(__file__).resolve().parents[3]
ARCHIVE = ROOT/'docs/experiments/rl-main-2026-10-11'
BASELINES = ('greedy', 'sweep', 'random', 'masked_random', 'planner')


class MissionRequest(BaseModel):
    policy: str = 'greedy'
    world: int = Field(default=24000, ge=0, le=2**31-1)
    grid: int = Field(default=40)
    battery: float = Field(default=160, ge=40, le=240)
    recharge: float = Field(default=8, ge=1, le=16)
    spread: float = Field(default=.04, ge=0, le=.12)

    model_config = {'extra': 'forbid'}


def verified_policy(path, algorithm, seed):
    if path is None or not path.is_file():
        raise ValueError('Frozen checkpoint is not installed locally.')
    manifest = json.loads((ARCHIVE/f'fit-{algorithm}-{seed}.json').read_text())
    expected = manifest['files']['policy.pt']['sha256']
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError('Checkpoint SHA-256 differs from the frozen fit.')
    from wild_drone.masked_learning import load_masked_policy
    policy = load_masked_policy(io.BytesIO(contents))
    if (policy.algorithm != algorithm or policy.train_config.seed != seed or
            policy.env_config.to_dict() != manifest['env_config'] or
            policy.train_config.to_dict() != manifest['train_config']):
        raise ValueError('Checkpoint configuration differs from the frozen fit.')
    return policy


def verified_checkpoints(directory):
    """Only expose the nine frozen fits, with exact archived byte/config identity."""
    result, unavailable = {}, {}
    for algorithm in ('dqn', 'reinforce', 'a2c'):
        for seed in range(3):
            key = f'{algorithm}-{seed}'
            path = Path(directory)/algorithm/f'seed-{seed}'/'policy.pt' if directory else None
            try:
                verified_policy(path, algorithm, seed)
                result[key] = path
            except (ValueError, KeyError, RuntimeError, OSError) as exc:
                unavailable[key] = str(exc)
    return result, unavailable


class MissionManager:
    def __init__(self, checkpoints=None, *, workers=2, capacity=8, ttl=1800):
        self.checkpoints, self.unavailable = verified_checkpoints(checkpoints)
        self.executor = ThreadPoolExecutor(max_workers=workers)
        self.jobs, self.lock = {}, threading.Lock()
        self.capacity, self.ttl = capacity, ttl

    def create(self, request):
        if request.grid not in (20, 40):
            raise ValueError('Choose the 20×20 or 40×40 preset.')
        config = EnvConfig() if request.grid == 40 else EnvConfig(
            grid_size=20, horizon=160, battery_capacity=80, recharge_rate=5,
            sensor_radius=1, inspection_deadline=25, max_fuel=100)
        config = replace(config, battery_capacity=request.battery, recharge_rate=request.recharge,
                         spread_probability=request.spread)
        if request.policy not in BASELINES and request.policy not in self.checkpoints:
            raise ValueError('Controller unavailable. Install a verified frozen checkpoint for RL.')
        if request.policy in self.checkpoints and config != EnvConfig():
            raise ValueError('Frozen RL controllers require the unchanged 40×40 environment.')
        with self.lock:
            now = time.monotonic()
            self.jobs = {k: j for k, j in self.jobs.items() if
                         j['status'] in ('queued', 'running') or now-j['updated'] < self.ttl}
            if len(self.jobs) >= self.capacity:
                completed = [k for k, j in self.jobs.items()
                             if j['status'] not in ('queued', 'running')]
                if completed:
                    del self.jobs[min(completed, key=lambda k: self.jobs[k]['updated'])]
                else:
                    raise OverflowError('Mission queue is full; cancel or finish a run first.')
            identity = uuid.uuid4().hex
            self.jobs[identity] = {'status': 'queued', 'frames': [], 'packet': None,
                                   'summary': None, 'error': None, 'cancel': threading.Event(),
                                   'updated': now}
        self.executor.submit(self.run, identity, request, config)
        return identity

    def run(self, identity, request, config):
        try:
            if request.policy in self.checkpoints:
                algorithm, seed = request.policy.split('-')
                policy = verified_policy(self.checkpoints[request.policy], algorithm, int(seed))
            elif request.policy == 'masked_random':
                from wild_drone.masked_learning import MaskedRandomPolicy
                policy = MaskedRandomPolicy(request.world)
            else:
                policy = make_policy(request.policy, request.world)
            env = WildfireFleetEnv(config)
            obs, _ = env.reset(seed=request.world)
            with self.lock:
                job = self.jobs[identity]
                job['status'] = 'running'
                fitted = request.policy in self.checkpoints
                job['packet'] = initial_packet(
                    env, policy=request.policy.split('-')[0] if fitted else request.policy,
                    world=request.world, training_seed=int(seed) if fitted else None)
                job['frames'].append(snapshot(env))
            while not env._done and not job['cancel'].is_set():
                before = {'positions': env.positions.copy(), 'batteries': env.batteries.copy(),
                          'lost': env.lost.copy()}
                action = policy.act(env, obs)
                obs, _, _, _, info = env.step(action)
                frame = snapshot(env, info, action=action, previous=before)
                with self.lock:
                    job['frames'].append(frame)
                    job['updated'] = time.monotonic()
            with self.lock:
                job['status'] = 'cancelled' if job['cancel'].is_set() else 'completed'
                if job['status'] == 'completed':
                    job['summary'] = mission_summary(env, job['frames'])
                job['updated'] = time.monotonic()
        except Exception as exc:
            with self.lock:
                self.jobs[identity].update(status='failed', error=str(exc),
                                           updated=time.monotonic())

    def read(self, identity, after):
        with self.lock:
            if identity not in self.jobs:
                raise KeyError(identity)
            job = self.jobs[identity]
            return {k: job[k] for k in ('status', 'packet', 'summary', 'error')} | {
                'id': identity, 'frames': [f for f in job['frames'] if f['step'] > after]}

    def cancel(self, identity):
        with self.lock:
            if identity not in self.jobs:
                raise KeyError(identity)
            self.jobs[identity]['cancel'].set()

    def close(self):
        with self.lock:
            for job in self.jobs.values():
                job['cancel'].set()
        self.executor.shutdown(wait=True, cancel_futures=True)


def create_app(*, checkpoint_root=None, static_dir=None, manager=None):
    missions = manager or MissionManager(checkpoint_root)

    @asynccontextmanager
    async def lifespan(app):
        yield
        missions.close()

    app = FastAPI(title='Wild Drone Mission Control', lifespan=lifespan)
    app.state.missions = missions

    @app.exception_handler(RequestValidationError)
    async def invalid_settings(request, exc):
        # Do not echo nonfinite request inputs into a JSON error response.
        messages = [f'{e["loc"][-1]}: {e["msg"]}' for e in exc.errors()]
        return JSONResponse({'detail': '; '.join(messages)}, status_code=422)

    @app.get('/api/capabilities')
    def capabilities():
        return {'mode': 'local', 'controllers': [
            {'id': p, 'available': True, 'kind': 'baseline'} for p in BASELINES] + [
            {'id': p, 'available': p in missions.checkpoints, 'kind': 'rl',
             'reason': missions.unavailable.get(p)} for p in
            [f'{a}-{s}' for a in ('dqn', 'reinforce', 'a2c') for s in range(3)]],
            'max_horizon': 300, 'concurrent_missions': 2}

    @app.post('/api/missions', status_code=202)
    def create(request: MissionRequest):
        try:
            return {'id': missions.create(request)}
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except OverflowError as exc:
            raise HTTPException(429, str(exc)) from exc

    @app.get('/api/missions/{identity}')
    def read(identity: str, after_step: int = Query(-1, ge=-1, le=300)):
        try:
            return missions.read(identity, after_step)
        except KeyError as exc:
            raise HTTPException(404, 'Mission expired or unknown.') from exc

    @app.delete('/api/missions/{identity}', status_code=202)
    def cancel(identity: str):
        try:
            missions.cancel(identity)
            return {'status': 'cancellation requested'}
        except KeyError as exc:
            raise HTTPException(404, 'Mission expired or unknown.') from exc

    directory = Path(static_dir) if static_dir else ROOT/'web/dist'
    if directory.is_dir():
        app.mount('/', StaticFiles(directory=directory, html=True), name='frontend')
    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8000)
    parser.add_argument('--checkpoint-root', type=Path)
    parser.add_argument('--static-dir', type=Path)
    args = parser.parse_args()
    import uvicorn
    uvicorn.run(create_app(checkpoint_root=args.checkpoint_root, static_dir=args.static_dir),
                host=args.host, port=args.port)
