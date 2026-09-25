"""417 frontier worker; frozen 402 worker is used unchanged for all old kinds."""
import gc
import multiprocessing as mp
import time
import traceback
import numpy as np
import prefix_deadline_worker_v2 as old


def worker(connection, config, root):
    if config['kind'] != 'frontier':
        return old.worker(connection, config, root)
    generation = 0; phase = 'setup'
    try:
        # Task-free imports. No observed support or query is available yet.
        import continuous_frontier_prediction_v1 as model
        identity = model.registry_identity()
        connection.send(dict(kind='ready', generation=0, model_fingerprint=None))
        while True:
            phase = 'waiting'; command = connection.recv()
            if command['kind'] == 'close':
                break
            assert command['kind'] == 'run' and command['generation'] == generation
            phase = 'online'; started = time.perf_counter(); ordinal = 0
            def emit(kind, prediction):
                nonlocal ordinal
                connection.send(dict(kind=kind, generation=generation, ordinal=ordinal,
                    prediction=np.asarray(prediction, np.float64), worker_seconds=time.perf_counter()-started))
                ordinal += 1
            arrays, metadata = model.fit(command['x'], command['v'], command['q'], seed=command['seed'],
                method=config['method'], search_seconds=config.get('search_seconds', .25),
                max_expanded=config.get('max_expanded', 65536), batch_queries=config.get('batch_queries', 32), emit=emit)
            tick = time.perf_counter()
            metadata = model.reader.serializable(metadata)
            metadata['certificate_serialization_seconds'] = time.perf_counter()-tick
            metadata['worker_end_to_end_seconds'] = time.perf_counter()-started
            emit('final', arrays['prediction']); phase = 'archive'; request = connection.recv()
            if request['kind'] == 'archive':
                connection.send(dict(arrays=arrays, metadata=metadata))
            else:
                assert request['kind'] == 'discard'
            del command, request, arrays, metadata
            gc.collect(); assert model.registry_identity() == identity
            generation += 1
            connection.send(dict(kind='ready', generation=generation, model_fingerprint=None))
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        connection.send(dict(kind='error', phase=phase, generation=generation, traceback=traceback.format_exc()))
    finally:
        connection.close()


class Session(old.Session):
    def _start(self):
        assert self.process is None
        ctx = mp.get_context('spawn'); parent, child = ctx.Pipe(); start = time.perf_counter()
        process = ctx.Process(target=worker, args=(child, self.config, self.root), daemon=True)
        process.start(); child.close(); self.process = process; self.parent = parent
        try:
            if not parent.poll(60.):
                raise TimeoutError('Task-free frontier worker setup exceeded 60 seconds')
            ready = parent.recv(); assert ready['kind'] == 'ready', ready
            self.ready = ready; seconds = time.perf_counter()-start
            self.setups.append(dict(pid=process.pid, seconds=seconds, before_call=self.calls))
            return seconds
        except Exception:
            self._stop(force=True)
            raise
