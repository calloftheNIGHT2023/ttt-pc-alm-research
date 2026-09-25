"""395 task-isolated worker extension for matched cohort and anytime portfolios."""
import gc
import multiprocessing as mp
import time
import traceback
import numpy as np
import deadline_prediction_worker_v2 as parent
import prefix_matched_controls_v1 as controls


def prepare(config,root):
    if config['kind'] not in ('cohort','portfolio'):
        return parent.old.prepare(config,root)
    import independent_hybrid_memory as regression
    context=dict(regression=regression)
    if config['kind']=='cohort':
        model,manifest=controls.load_model(config,root)
        context.update(model=model,model_manifest=manifest)
    return context


def calculate(config,context,x,v,q,seed,emit):
    if config['kind'] not in ('cohort','portfolio'):
        return parent.calculate(config,context,x,v,q,seed,emit)
    assert len(x) in controls.STAGES
    begin=time.perf_counter()
    fallback,_,fm=context['regression'].regression(x,v,'prior4096_ridge')
    fallback_prediction=np.clip(fallback(q),0.,1.)
    emit('fallback',fallback_prediction)
    fallback_seconds=time.perf_counter()-begin
    if config['kind']=='cohort':
        arrays,metadata=controls.fit_model(context['model'],x,v,q)
        metadata['model_manifest']=context['model_manifest']
    else:
        arrays,metadata=controls.fit_portfolio(x,v,q,seed=seed,name=config['name'],emit=emit)
    arrays['fallback_prediction']=fallback_prediction
    metadata.update(fallback_seconds=fallback_seconds,fallback_metadata=fm,
                    support_prefix=len(x),end_to_end_seconds=time.perf_counter()-begin)
    return arrays['prediction'],arrays,metadata


def worker(connection,config,root):
    phase='setup';generation=0
    try:
        context=prepare(config,root);initial=parent.fingerprint(context)
        connection.send(dict(kind='ready',generation=generation,model_fingerprint=initial))
        while True:
            phase='waiting';command=connection.recv()
            if command['kind']=='close':
                break
            assert command['kind']=='run' and command['generation']==generation
            phase='online';started=time.perf_counter();ordinal=0
            def emit(kind,prediction):
                nonlocal ordinal
                connection.send(dict(kind=kind,generation=generation,ordinal=ordinal,
                    prediction=np.asarray(prediction,np.float64),worker_seconds=time.perf_counter()-started))
                ordinal+=1
            prediction,arrays,metadata=calculate(config,context,command['x'],command['v'],command['q'],command['seed'],emit)
            emit('final',prediction);phase='archive';request=connection.recv()
            if request['kind']=='archive':
                connection.send(dict(arrays=arrays,metadata=metadata))
            else:
                assert request['kind']=='discard'
            del command,request,prediction,arrays,metadata
            gc.collect();assert parent.fingerprint(context)==initial
            generation+=1
            connection.send(dict(kind='ready',generation=generation,model_fingerprint=initial))
    except (EOFError,BrokenPipeError):
        pass
    except Exception:
        connection.send(dict(kind='error',phase=phase,generation=generation,traceback=traceback.format_exc()))
    finally:
        connection.close()


class Session(parent.Session):
    def _start(self):
        assert self.process is None
        ctx=mp.get_context('spawn');parent_pipe,child=ctx.Pipe();start=time.perf_counter()
        process=ctx.Process(target=worker,args=(child,self.config,self.root),daemon=True)
        process.start();child.close();self.process=process;self.parent=parent_pipe
        try:
            if not parent_pipe.poll(60.):
                raise TimeoutError('Task-free prefix worker setup exceeded 60 seconds')
            ready=parent_pipe.recv();assert ready['kind']=='ready',ready
            self.ready=ready;seconds=time.perf_counter()-start
            self.setups.append(dict(pid=process.pid,seconds=seconds,before_call=self.calls))
            return seconds
        except Exception:
            self._stop(force=True);raise
