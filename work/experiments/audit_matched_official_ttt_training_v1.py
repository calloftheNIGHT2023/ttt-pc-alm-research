"""390 audit saved matched cohort, selected checkpoints and CPU old-validation replay.

Run after training finishes; no new research query targets and no training retry.
"""
from pathlib import Path
import hashlib
import json
import traceback

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'results/matched_official_ttt/training_v1'
OUT=ROOT/'results/matched_official_ttt/training_audit_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def main():
    import numpy as np
    import torch
    import matched_official_ttt_v1 as ttt
    import matched_shifted_meta_models as population
    torch.set_num_threads(1)
    finished=read(SOURCE/'summary.json');assert finished['passed']
    assert finished['trained_models']==8 and not (SOURCE/'failure.json').exists()
    for f,digest in finished['outputs_sha256'].items():
        assert sha(SOURCE/f)==digest
    protocol=read(SOURCE/'protocol.json')
    for f,digest in protocol['source_sha256'].items():
        assert sha(ROOT/f)==digest
    cohort=torch.load(SOURCE/'training_cohort.pt',map_location='cpu',weights_only=True)
    manifest=read(SOURCE/'cohort_manifest.json');assert sha(SOURCE/'training_cohort.pt')==manifest['checkpoint_sha256']
    names=['support_x','support_v','query_x','query_target'];fingerprints=[]
    generator=torch.Generator(device='cpu').manual_seed(protocol['train_seed'])
    for i in range(protocol['steps']):
        expected=population.training_batch(protocol['batch_size'],protocol['train_queries'],generator,'cpu')
        digest=hashlib.sha256()
        for name,tensor in zip(names,expected):
            actual=cohort[name][i];assert torch.equal(actual,tensor)
            array=actual.contiguous().numpy()
            digest.update(str((array.shape,array.dtype.str)).encode('ascii'));digest.update(array.tobytes())
        fingerprints.append(digest.hexdigest())
    assert fingerprints==manifest['batch_sha256']
    logs=read(SOURCE/'training_log.json');records=read(SOURCE/'training_summary.json')
    validation=population.fixed_batch(protocol['validation_seeds'],protocol['validation_queries'],'cpu')
    output=[]
    for cfg in protocol['configs']:
        name=cfg['name'];record=next(r for r in records if r['method']==name)
        checkpoint=SOURCE/(name+'.pt');consumed=SOURCE/(name+'_consumed_batches.json')
        assert sha(checkpoint)==record['checkpoint_sha256']
        assert sha(consumed)==record['consumed_batches_sha256'] and read(consumed)==fingerprints
        events=[r for r in logs if r['method']==name]
        assert [r['step'] for r in events]==list(range(0,protocol['steps']+1,protocol['validation_interval']))
        selected=min(events,key=lambda r:(r['validation_raw_mse'],r['step']))
        assert selected['step']==record['selected_step']
        assert selected['validation_raw_mse']==record['best_validation_raw_mse']
        if cfg['kind']=='official_ttt':
            model=ttt.MatchedOfficialTTT(cfg['head_dim'],cfg['inner_passes'],cfg['key_basis'],cfg['rate_mode']).double()
        else:
            model=population.make_model(cfg)
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
        model.eval();before={k:v.clone() for k,v in model.state_dict().items()}
        assert sum(p.numel() for p in model.parameters() if p.requires_grad)==record['trainable_parameters']
        x,v,q,target=validation
        with torch.no_grad():
            state=None;stage_losses=[]
            for n in ttt.STAGES:
                state=model.adapt(x[:,:n],v[:,:n],state)
                prediction=model.predict(state,q)
                assert torch.isfinite(prediction).all()
                # Predictor is read-only; query permutation does not change outputs.
                reverse=model.predict(state,q.flip(1)).flip(1)
                assert torch.allclose(prediction,reverse,rtol=1e-11,atol=1e-12)
                stage_losses.append(float((prediction-target).square().mean()))
        mse=float(np.mean(stage_losses));gap=abs(mse-record['best_validation_raw_mse'])
        assert gap<=2e-7, (name,mse,record['best_validation_raw_mse'],gap)
        assert all(torch.equal(v,before[k]) for k,v in model.state_dict().items())
        assert record['outer_tasks']==protocol['steps']*protocol['batch_size']
        assert record['unique_outer_query_values']==record['outer_tasks']*protocol['train_queries']
        assert record['outer_query_loss_exposures']==4*record['unique_outer_query_values']
        output.append(dict(method=name,passed=True,selected_step=record['selected_step'],
            validation_cpu_raw_mse=mse,stage_validation_raw_mse=stage_losses,
            cpu_vs_training_device_validation_gap=gap,
            exact_cohort_batches=len(fingerprints),slow_state_unchanged=True,query_permutation_readonly=True))
        print(json.dumps(output[-1]),flush=True)
    save(OUT/'models.json',output)
    save(OUT/'summary.json',dict(passed=True,models=len(output),
        regenerated_cohort_batches=protocol['steps'],exact_regenerated_tensor_arrays=4*protocol['steps'],
        matched_consumption_batches=len(output)*protocol['steps'],
        maximum_cpu_validation_gap=max(r['cpu_vs_training_device_validation_gap'] for r in output),
        original_training_not_rerun=True,new_task_query_targets_accessed=False,
        actual_downstream_training=False,source_sha256=sha(Path(__file__)),
        training_summary_sha256=sha(SOURCE/'summary.json'),
        outputs_sha256={'models.json':sha(OUT/'models.json')}))
    print(json.dumps(read(OUT/'summary.json')),flush=True)


if __name__=='__main__':
    assert (SOURCE/'summary.json').exists() and read(SOURCE/'summary.json')['passed'], 'Training must finish first'
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        main()
    except Exception:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
