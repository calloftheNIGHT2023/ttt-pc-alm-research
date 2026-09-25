"""425 compiled reader with unchanged 418 receipts and deadline controller."""
import multiprocessing as mp
import time
import traceback
import frontier_deadline_worker_v2 as old


def worker(connection, config, root):
    if config['kind'] != 'compiled_frontier_receipt':
        return old.worker(connection, config, root)
    try:
        import compiled_continuous_frontier_v1 as model
        with model.installed():
            old.worker(connection, dict(config, kind='frontier_receipt'), root)
    except Exception:
        try:
            connection.send(dict(kind='error', phase='compiled_scope', generation=0, traceback=traceback.format_exc()))
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
                raise TimeoutError('Task-free compiled frontier worker setup exceeded 60 seconds')
            ready = parent.recv(); assert ready['kind'] == 'ready', ready
            self.ready = ready; seconds = time.perf_counter()-start
            self.setups.append(dict(pid=process.pid, seconds=seconds, before_call=self.calls))
            return seconds
        except Exception:
            self._stop(force=True)
            raise
