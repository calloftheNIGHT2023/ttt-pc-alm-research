"""386 prototype warm-worker deadline boundary, without query-target access.

Only predictions RECEIVED by the controller by the deadline are admitted.
Setup, online receipt, cancellation/cleanup and archive transfer are separate
measured costs. This is not a hard real-time OS guarantee or a trained method.
"""
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback
import numpy as np
import psutil


def choose(events, budget, q):
    """Pure, independently replayable deadline rule; never inspect a query answer."""
    prediction = np.full(np.shape(q), .5, dtype=np.float64); selected = 'constant'
    for event in events:
        if event['received_seconds'] <= budget and event['kind'] in ['fallback', 'final']:
            a = np.asarray(event['prediction'], dtype=np.float64)
            assert a.shape == np.shape(q) and np.isfinite(a).all() and np.all((a >= 0) & (a <= 1))
            prediction = a.copy(); selected = event['kind']
    return prediction, selected


def prepare(config, root):
    """Task-free module/weight loading. This phase cannot receive support data."""
    kind = config['kind']
    if kind == 'probe': return {}
    import independent_hybrid_memory as regression
    context = dict(regression=regression)
    if kind == 'regional':
        import continuous_regional_prediction_v1 as model
        context['model'] = model
    elif kind == 'optimizer':
        import n24_optimizer_controls_v1 as model
        context['model'] = model
    elif kind == 'meta':
        import n24_meta_input_adapter_v1 as meta
        meta.torch.set_num_threads(1); meta.torch.set_num_interop_threads(1)
        models, manifest = meta.load(root)
        context.update(meta=meta, model=models[config['method']], model_manifest=manifest[config['method']])
    else: assert kind == 'regression'
    return context


def calculate(config, context, x, v, q, seed, emit):
    kind = config['kind']; arrays = dict(x=x.copy(), v=v.copy(), q=q.copy())
    if kind == 'probe':
        time.sleep(config.get('first_delay', 0.))
        emit('fallback', np.full_like(q, .25))
        if config.get('raise_after_fallback'): raise RuntimeError('Intentional deadline preflight error')
        time.sleep(config.get('final_delay', 0.))
        pred = np.full_like(q, .75); arrays['prediction'] = pred
        return pred, arrays, dict(probe_only=True)
    regression = context['regression']
    # Standalone regressions retain their normal fast path. All iterative/meta
    # candidates share the same explicitly paid prior4096 fallback, not free labels.
    if kind != 'regression':
        tick = time.perf_counter(); fallback, _, fallback_meta = regression.regression(x, v, 'prior4096_ridge')
        fallback_prediction = np.clip(fallback(q), 0., 1.)
        fallback_seconds = time.perf_counter()-tick
        emit('fallback', fallback_prediction)
        arrays['fallback_prediction'] = fallback_prediction
    if kind == 'regional':
        a, m = context['model'].fit(x, v, q, seed=seed, name=config['method'],
            schedule=config['schedule'], order_name=config['ordering'])
        arrays.update(a)
        # An incomplete union is not silently substituted for a complete posterior.
        if m['full_candidate_set_resolved'] and m['readout_available']:
            pred = a['readout_prediction']; m['selected_rule'] = 'fully_resolved_posterior'
        else:
            pred = fallback_prediction; m['selected_rule'] = 'paid_prior_fallback'
        m.update(fallback_seconds=fallback_seconds, fallback_metadata=fallback_meta)
    elif kind == 'optimizer':
        a, m = context['model'].fit(x, v, q, seed=seed, family=config['family'],
            steps=config['steps'], restarts=config['restarts'], readout=config['readout'])
        arrays.update(a); pred = a['prediction']
        m.update(fallback_seconds=fallback_seconds, fallback_metadata=fallback_meta)
    elif kind == 'meta':
        state = None; stages = [4, 8, 16, 24] if config.get('warm_trajectory') else [24]
        for n in stages: predict, state, m = context['meta'].fit(context['model'], x[:n], v[:n], state)
        pred = predict(q); arrays.update(context['meta'].arrays(state))
        m.update(stages=stages, support_exposures=sum(stages), model_manifest=context['model_manifest'],
            fallback_seconds=fallback_seconds, fallback_metadata=fallback_meta)
    else:
        assert kind == 'regression'
        if config['method'] == 'residual_linear_ridge':
            phi = np.column_stack([np.ones(len(x)), x]); y = v-regression.base.forward(x, np.zeros(4))
            w = np.linalg.solve(phi.T@phi+1e-4*np.eye(2), phi.T@y)
            pred = w[0]+q*w[1]+regression.base.forward(q, np.zeros(4))
            arrays['w'] = w; m = dict(persistent_state_bytes=w.nbytes, ridge_lambda=1e-4)
        else:
            predict, state, m = regression.regression(x, v, config['method'])
            pred = predict(q)
            if isinstance(state, dict): arrays.update({k:value for k, value in state.items() if isinstance(value, np.ndarray)})
    pred = np.clip(pred, 0., 1.); arrays['prediction'] = pred
    m.update(query_targets_accessed=False, output_clipping=[0., 1.])
    return pred, arrays, m


def worker(connection, config, root):
    phase = 'setup'
    try:
        context = prepare(config, root); connection.send(dict(kind='ready'))
        command = connection.recv()
        if command['kind'] == 'close': return
        assert command['kind'] == 'run'; phase = 'online'; started = time.perf_counter()
        def emit(kind, prediction):
            connection.send(dict(kind=kind, prediction=np.asarray(prediction, np.float64),
                worker_seconds=time.perf_counter()-started))
        prediction, arrays, metadata = calculate(config, context, command['x'], command['v'], command['q'], command['seed'], emit)
        emit('final', prediction); phase = 'archive'
        command = connection.recv()
        if command['kind'] == 'archive': connection.send(dict(arrays=arrays, metadata=metadata))
        else: assert command['kind'] == 'close'
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(dict(kind='error', phase=phase, traceback=traceback.format_exc()))
    finally:
        connection.close()


def execute(config, x, v, q, *, seed, budget, root=None, archive=True):
    assert budget >= 0 and x.shape == v.shape and x.ndim == q.ndim == 1
    assert np.isfinite(x).all() and np.isfinite(v).all() and np.isfinite(q).all()
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    ctx = mp.get_context('spawn'); parent, child = ctx.Pipe()
    process = ctx.Process(target=worker, args=(child, config, root), daemon=True)
    events = []; saved = None; error = None; terminated = False; received_final = False
    resource_samples = []; controller = psutil.Process(os.getpid())
    setup_start = time.perf_counter(); process.start(); child.close()
    try:
        if not parent.poll(60.): raise TimeoutError('Worker setup did not become ready within 60 seconds')
        ready = parent.recv()
        if ready['kind'] != 'ready': raise RuntimeError(ready)
        setup_seconds = time.perf_counter()-setup_start
        monitored = psutil.Process(process.pid)
        initial_cpu = monitored.cpu_times(); initial_memory = monitored.memory_info()._asdict()
        started = time.perf_counter()
        def sample_resources():
            try:
                memory = monitored.memory_info()._asdict(); cpu = monitored.cpu_times()
                resource_samples.append(dict(seconds=time.perf_counter()-started,
                    worker_rss=memory['rss'], controller_rss=controller.memory_info().rss,
                    worker_peak_working_set_since_creation=memory.get('peak_wset'),
                    worker_cpu_seconds=(cpu.user+cpu.system)-(initial_cpu.user+initial_cpu.system)))
            except psutil.NoSuchProcess:
                pass
        parent.send(dict(kind='run', x=x, v=v, q=q, seed=seed))
        while True:
            remaining = budget-(time.perf_counter()-started)
            if remaining <= 0: break
            if not parent.poll(min(remaining, .005)):
                sample_resources(); continue
            event = parent.recv(); event['received_seconds'] = time.perf_counter()-started
            events.append(event)
            if event['kind'] == 'error': error = event; break
            if event['kind'] == 'final': received_final = True; break
        prediction, selected = choose(events, budget, q)
        sample_resources()
        decision_seconds = time.perf_counter()-started
        archive_start = time.perf_counter(); archive_seconds = 0.
        if received_final:
            parent.send(dict(kind='archive' if archive else 'close'))
            if archive:
                if not parent.poll(60.): raise TimeoutError('Completed worker archive transfer timed out')
                saved = parent.recv(); assert 'arrays' in saved and 'metadata' in saved
                archive_seconds = time.perf_counter()-archive_start
        cleanup_start = time.perf_counter()
        if not received_final and process.is_alive(): process.terminate(); terminated = True
        process.join(timeout=10.)
        if process.is_alive(): process.kill(); process.join(timeout=10.)
        assert not process.is_alive(), 'Owned worker survived cleanup'
        cleanup_seconds = time.perf_counter()-cleanup_start
        metadata = dict(config=config, budget_seconds=budget, selected=selected, error=error,
            setup_seconds=setup_seconds, decision_seconds=decision_seconds,
            controller_overrun_seconds=max(0., decision_seconds-budget), cleanup_seconds=cleanup_seconds,
            archive_seconds=archive_seconds, terminated_owned_worker=terminated, worker_exitcode=process.exitcode,
            received_final=received_final, query_targets_accessed=False,
            hard_realtime_guarantee=False, warm_runtime_setup_excluded_and_separately_charged=True,
            receipt_and_prediction_copy_inside_online_clock=True, late_predictions_admitted=False,
            rss_peak_measured=False, initialization_receives_support=False,
            setup_worker_memory=initial_memory, resource_samples=resource_samples,
            worker_max_sampled_rss=max((s['worker_rss'] for s in resource_samples), default=initial_memory['rss']),
            controller_max_sampled_rss=max((s['controller_rss'] for s in resource_samples), default=controller.memory_info().rss),
            resource_poll_interval_seconds=.005,
            resource_scope='5ms observed process RSS lower bound to true online peak; Windows peak_wset includes setup; shared pages not deduplicated; CPU to last sample')
        return prediction, metadata, events, saved
    finally:
        if process.is_alive(): process.terminate(); process.join(timeout=10.)
        parent.close()
