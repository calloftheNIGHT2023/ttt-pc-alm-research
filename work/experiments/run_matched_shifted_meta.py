"""Fixed-budget meta controls on the study34 population, not fresh confirmation."""
import argparse,copy,hashlib,json,time,os
from pathlib import Path
import numpy as np
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
import matched_shifted_meta_models as m


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda');a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    device=torch.device(a.device)
    protocol=dict(**cfg,device=str(device),torch_version=torch.__version__,gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
                  source_sha256={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in [Path(__file__),Path(m.__file__)]},config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),audit=m.verify(),
                  task_scope='same shifted-tent prior; all test streams historically observed; outer training BP allowed and explicitly charged; no official TTT claim',
                  dtype='float64',warm_start='shallow carries fast parameters over 4,8,16,24 support prefixes; ridge refits closed form on full prefix')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    validation=m.fixed_batch(cfg['validation_seeds'],cfg['validation_queries'],device);rows=[];training=[];summaries=[]
    for c in cfg['configs']:
        torch.manual_seed(cfg['initial_seed']);model=m.make_model(c).to(device);trainable=[p for p in model.parameters() if p.requires_grad]
        gen=torch.Generator(device=device).manual_seed(cfg['train_seed']);best=float('inf');best_step=0;selected=None
        opt=torch.optim.Adam(trainable,lr=c['lr']) if trainable else None
        begin=time.perf_counter()
        for step in range(cfg['steps']+1 if trainable else 1):
            if step:
                model.train();batch=m.training_batch(cfg['batch_size'],cfg['train_queries'],gen,device)
                loss=m.trajectory_loss(model,batch);assert torch.isfinite(loss)
                opt.zero_grad(set_to_none=True);loss.backward();gn=torch.nn.utils.clip_grad_norm_(trainable,1.);opt.step()
            if step%cfg['validation_interval']==0 or step==cfg['steps']:
                model.eval()
                with torch.no_grad():vl=float(m.trajectory_loss(model,validation).cpu())
                if vl<best:best=vl;best_step=step;selected=copy.deepcopy(model.state_dict())
                log=dict(method=c['name'],step=step,validation_raw_query_mse=vl,best=best,
                         train_query_mse=float(loss.detach().cpu()) if step else None,gradient_norm=float(gn.cpu()) if step else None)
                training.append(log);print(json.dumps(log),flush=True)
                (a.out/'training.json').write_text(json.dumps(training,indent=2),encoding='utf-8')
        if device.type=='cuda':torch.cuda.synchronize()
        train_seconds=time.perf_counter()-begin
        model.load_state_dict(selected);model=model.cpu();model.eval();torch.save(selected,a.out/(c['name']+'.pt'))
        # Evaluate and time all methods on the same CPU as scalar ALM; GPU training is separate.
        data=m.fixed_batch(cfg['evaluation_seeds'],cfg['evaluation_queries'],'cpu')
        with torch.no_grad():
            dummy=model.adapt(data[0][:1,:4],data[1][:1,:4]);model.predict(dummy,data[2][:1])
            for i,seed in enumerate(cfg['evaluation_seeds']):
                x,v,q,target=[z[i:i+1] for z in data];state=None
                for n in m.STAGES:
                    start=time.perf_counter();state=model.adapt(x[:,:n],v[:,:n],state);fit=time.perf_counter()-start
                    start=time.perf_counter();pred=model.predict(state,q);read=time.perf_counter()-start
                    error=float((model.predict(state,x[:,:n]).clamp(0,1)-v[:,:n]).abs().max())
                    arrays=state if isinstance(state,tuple) else [state]
                    rows.append(dict(method=c['name'],seed=seed,n_context=n,query_mse=float((pred.clamp(0,1)-target).square().mean()),raw_query_mse=float((pred-target).square().mean()),
                                     adaptation_seconds=fit,read_queries_seconds=read,support_max_error=error,support_feasible=error<=m.EPS+1e-6,
                                     fast_state_bytes=sum(t.numel()*t.element_size() for t in arrays),shared_model_bytes=sum(t.numel()*t.element_size() for t in model.state_dict().values())))
        summaries.append(dict(method=c['name'],outer_training_seconds=train_seconds,outer_tasks=cfg['steps']*cfg['batch_size'] if trainable else 0,
                              outer_task_query_values=(cfg['steps']*cfg['batch_size']*cfg['train_queries']) if trainable else 0,
                              trainable_parameters=sum(p.numel() for p in trainable),selected_step=best_step,best_validation_raw_mse=best,
                              checkpoint_sha256=hashlib.sha256((a.out/(c['name']+'.pt')).read_bytes()).hexdigest()))
        (a.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(a.out/'training_summary.json').write_text(json.dumps(summaries,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed_method=c['name'],rows=len(rows),training_seconds=train_seconds)),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
