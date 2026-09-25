"""402 portfolio-only task-free runtime preload; frozen v1 math unchanged."""
import multiprocessing as mp
import time
import traceback
import prefix_deadline_worker_v1 as original


def worker(connection, config, root):
    try:
        if config['kind'] == 'portfolio':
            # No support/query data have been received. Single optimizer workers
            # already import this dependency in prepare before publishing ready.
            import n24_optimizer_controls_v1
            import candidate_set_readout_v1
    except Exception:
        connection.send(dict(kind='error', phase='setup', generation=0, traceback=traceback.format_exc()))
        connection.close()
        return
    original.worker(connection, config, root)


class Session(original.Session):
    def _start(self):
        assert self.process is None
        ctx = mp.get_context('spawn')
        parent_pipe, child = ctx.Pipe()
        start = time.perf_counter()
        process = ctx.Process(target=worker, args=(child, self.config, self.root), daemon=True)
        process.start()
        child.close()
        self.process, self.parent = process, parent_pipe
        try:
            if not parent_pipe.poll(60.):
                raise TimeoutError('Task-free prefix v2 worker setup exceeded 60 seconds')
            ready = parent_pipe.recv()
            assert ready['kind'] == 'ready', ready
            self.ready = ready
            seconds = time.perf_counter()-start
            self.setups.append(dict(pid=process.pid, seconds=seconds, before_call=self.calls))
            return seconds
        except Exception:
            self._stop(force=True)
            raise
