"""387 task-isolated reusable warm worker; all restart/setup costs retained.

Extends v1 with larger fixed-prior regressions and reuse of immutable model
state only. A deadline kills only this session's worker; the next task pays
for recreation. No support state, search, geometry or fast weights persist.
"""
import gc
import hashlib
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
import psutil
import deadline_prediction_worker_v1 as old


def fingerprint(context):
    model = context.get('model')
    if not hasattr(model, 'state_dict'): return None
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode()); digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def calculate(config, context, x, v, q, seed, emit):
    if config['kind'] != 'regression' or config['method'] not in ['prior16384_ridge', 'prior65536_ridge']:
        return old.calculate(config, context, x, v, q, seed, emit)
    regression = context['regression']; begin = time.perf_counter()
    fallback, _, fm = regression.regression(x, v, 'prior4096_ridge')
    fallback_prediction = np.clip(fallback(q), 0., 1.); fallback_seconds = time.perf_counter()-begin
    emit('fallback', fallback_prediction)
    count = int(config['method'].split('_')[0][5:]); tick = time.perf_counter()
    predict, metadata = regression.posterior.prior_moments(x, v, 4, count)
    prediction = np.clip(predict(q), 0., 1.)
    arrays = dict(x=x.copy(), v=v.copy(), q=q.copy(), prediction=prediction, fallback_prediction=fallback_prediction)
    metadata.update(features=count, large_prior_seconds=time.perf_counter()-tick,
        fallback_seconds=fallback_seconds, fallback_metadata=fm, fixed_bank_seed=731,
        query_targets_accessed=False, global_bp_used=False, feature_bank_reused=False,
        total_seconds=time.perf_counter()-begin)
    return prediction, arrays, metadata


def worker(connection, config, root):
    phase = 'setup'; generation = 0
    try:
        context = old.prepare(config, root); initial = fingerprint(context)
        connection.send(dict(kind='ready', generation=generation, model_fingerprint=initial))
        while True:
            phase = 'waiting'; command = connection.recv()
            if command['kind'] == 'close': break
            assert command['kind'] == 'run' and command['generation'] == generation
            phase = 'online'; started = time.perf_counter()
            def emit(kind, prediction):
                connection.send(dict(kind=kind, generation=generation, prediction=np.asarray(prediction, np.float64),
                    worker_seconds=time.perf_counter()-started))
            prediction, arrays, metadata = calculate(config, context, command['x'], command['v'], command['q'], command['seed'], emit)
            emit('final', prediction); phase = 'archive'
            request = connection.recv()
            if request['kind'] == 'archive': connection.send(dict(arrays=arrays, metadata=metadata))
            else: assert request['kind'] == 'discard'
            del command, request, prediction, arrays, metadata
            gc.collect()
            assert fingerprint(context) == initial, 'Shared model changed across task adaptation'
            generation += 1
            connection.send(dict(kind='ready', generation=generation, model_fingerprint=initial))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(dict(kind='error', phase=phase, generation=generation, traceback=traceback.format_exc()))
    finally:
        connection.close()


class Session:
    def __init__(self, config, root=None):
        self.config = config; self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
        self.process = None; self.parent = None; self.ready = None
        self.setups = []; self.closures = []; self.calls = 0
        self.controller = psutil.Process(os.getpid())

    def _start(self):
        assert self.process is None
        ctx = mp.get_context('spawn'); parent, child = ctx.Pipe(); start = time.perf_counter()
        process = ctx.Process(target=worker, args=(child, self.config, self.root), daemon=True)
        process.start(); child.close(); self.process = process; self.parent = parent
        try:
            if not parent.poll(60.): raise TimeoutError('Task-free worker setup exceeded 60 seconds')
            ready = parent.recv(); assert ready['kind'] == 'ready', ready
            self.ready = ready
            seconds = time.perf_counter()-start
            self.setups.append(dict(pid=process.pid, seconds=seconds, before_call=self.calls))
            return seconds
        except Exception:
            self._stop(force=True); raise

    def _stop(self, force=False):
        if self.process is None: return 0.
        start = time.perf_counter(); process = self.process
        if process.is_alive():
            if force: process.terminate()
            else: self.parent.send(dict(kind='close'))
        process.join(timeout=10.)
        if process.is_alive(): process.kill(); process.join(timeout=10.)
        assert not process.is_alive(), 'Owned worker survived cleanup'
        seconds = time.perf_counter()-start
        self.closures.append(dict(pid=process.pid, seconds=seconds, force=force, exitcode=process.exitcode))
        self.parent.close(); self.process = None; self.parent = None; self.ready = None
        return seconds

    def close(self):
        self._stop()
        return dict(setups=self.setups, closures=self.closures, calls=self.calls,
            total_setup_seconds=sum(s['seconds'] for s in self.setups),
            total_closure_seconds=sum(s['seconds'] for s in self.closures))

    def run(self, x, v, q, *, seed, budget, archive=True):
        assert budget >= 0 and x.shape == v.shape and x.ndim == q.ndim == 1
        assert all(np.isfinite(a).all() for a in [x, v, q])
        setup_seconds = self._start() if self.process is None else 0.
        assert self.process.is_alive() and self.ready['kind'] == 'ready'
        process = self.process; parent = self.parent; generation = self.ready['generation']; self.calls += 1
        monitored = psutil.Process(process.pid); initial_memory = monitored.memory_info()._asdict(); initial_cpu = monitored.cpu_times()
        events = []; resources = []; archive_data = None; received_final = False; error = None
        started = time.perf_counter()
        def sample():
            try:
                memory = monitored.memory_info()._asdict(); cpu = monitored.cpu_times()
                resources.append(dict(seconds=time.perf_counter()-started, worker_rss=memory['rss'],
                    controller_rss=self.controller.memory_info().rss, worker_peak_working_set_since_creation=memory.get('peak_wset'),
                    worker_cpu_seconds=(cpu.user+cpu.system)-(initial_cpu.user+initial_cpu.system)))
            except psutil.NoSuchProcess: pass
        try:
            parent.send(dict(kind='run', generation=generation, x=x, v=v, q=q, seed=seed))
            while True:
                remaining = budget-(time.perf_counter()-started)
                if remaining <= 0: break
                if not parent.poll(min(remaining, .005)):
                    sample(); continue
                event = parent.recv(); event['received_seconds'] = time.perf_counter()-started
                assert event['generation'] == generation, 'Stale event from another task'
                events.append(event)
                if event['kind'] == 'error': error = event; break
                if event['kind'] == 'final': received_final = True; break
            prediction, selected = old.choose(events, budget, q); sample()
            decision_seconds = time.perf_counter()-started
            transfer = 0.; ready_seconds = 0.; cleanup_seconds = 0.
            if received_final:
                tick = time.perf_counter(); parent.send(dict(kind='archive' if archive else 'discard'))
                if archive:
                    if not parent.poll(60.): raise TimeoutError('Completed archive transfer exceeded 60 seconds')
                    archive_data = parent.recv(); assert 'arrays' in archive_data and 'metadata' in archive_data, archive_data
                transfer = time.perf_counter()-tick; tick = time.perf_counter()
                if not parent.poll(60.): raise TimeoutError('Worker task-state cleanup exceeded 60 seconds')
                ready = parent.recv()
                assert ready['kind'] == 'ready' and ready['generation'] == generation+1, ready
                assert ready['model_fingerprint'] == self.ready['model_fingerprint']
                self.ready = ready; ready_seconds = time.perf_counter()-tick
            else:
                cleanup_seconds = self._stop(force=True)
            metadata = dict(config=self.config, worker_pid=process.pid, generation=generation,
                budget_seconds=budget, selected=selected, error=error, received_final=received_final,
                setup_seconds=setup_seconds, decision_seconds=decision_seconds,
                controller_overrun_seconds=max(0., decision_seconds-budget), archive_seconds=transfer,
                task_cleanup_seconds=ready_seconds, terminated_owned_worker=not received_final,
                cleanup_seconds=cleanup_seconds, worker_reused=generation > 0,
                model_fingerprint_verified=received_final, query_targets_accessed=False,
                initialization_receives_support=False, hard_realtime_guarantee=False,
                warm_runtime_setup_excluded_and_separately_charged=True, late_predictions_admitted=False,
                shared_scientific_task_state_reused=False, setup_worker_memory=initial_memory,
                resource_samples=resources, resource_poll_interval_seconds=.005,
                worker_max_sampled_rss=max((s['worker_rss'] for s in resources), default=initial_memory['rss']),
                controller_max_sampled_rss=max((s['controller_rss'] for s in resources), default=self.controller.memory_info().rss),
                memory_scope='Sampled RSS is a lower bound on online peak; lifetime peak_wset includes setup/prior tasks; shared pages not deduplicated')
            return prediction, metadata, events, archive_data
        except Exception:
            self._stop(force=True); raise
