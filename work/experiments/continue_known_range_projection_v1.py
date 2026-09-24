"""288 one-shot same-source preflight then supplementary all8192, no retries."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

import psutil
from known_range_projection_pipeline_v1 import complete,gate,now,paths,read,save,sha,source_hashes


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve()
    hashes=gate(root,'preflight')
    out=root/'results/known_range_projection/pipeline_execution_v1'
    out.mkdir(parents=True,exist_ok=False)
    stages=[]
    for stage in ['preflight','confirmation']:
        loc=paths(root,stage)
        stages.extend([(stage,'predict','known_range_projection_pipeline_v1.py',loc['projected']),
                       (stage,'evaluate','known_range_projection_pipeline_v1.py',loc['evaluation']),
                       (stage,'audit','audit_known_range_projection_v1.py',loc['audit'])])
    assert all(not folder.exists() for _,_,_,folder in stages), 'Preserve existing stage; no automatic resume/retry'
    save(out/'protocol.json',dict(created_utc=now(),source_sha256=hashes,no_automatic_retry=True,
         final_original_seal_sha256=sha(root/'results/round_287_audit_v6.json'),
         stages=[dict(stage=s,phase=p,script=f,output=str(d.relative_to(root))) for s,p,f,d in stages],
         supplementary_analysis=True,core_research_goal_complete=False))
    try:
        for stage,phase,script,folder in stages:
            assert source_hashes(root)==hashes
            command=[sys.executable,'-u',str(root/'work/experiments'/script),'--project',str(root),'--stage',stage]
            if phase!='audit': command+=['--phase',phase]
            child=subprocess.Popen(command,cwd=root,env=os.environ.copy())
            try: creation=psutil.Process(child.pid).create_time()
            except psutil.NoSuchProcess: creation=None
            label=stage+'_'+phase
            save(out/(label+'_start.json'),dict(utc=now(),pid=child.pid,create_time=creation,script=script,script_sha256=sha(root/'work/experiments'/script)))
            print(dict(stage=stage,phase=phase,event='started',pid=child.pid,utc=now()),flush=True)
            code=child.wait()
            save(out/(label+'_exit.json'),dict(utc=now(),returncode=code))
            assert code==0,(label,code)
            complete(folder)
            save(out/(label+'_complete.json'),dict(utc=now(),summary_sha256=sha(folder/'summary.json')))
        result=dict(passed=True,utc=now(),stages_completed=6,supplementary_analysis=True,
                    core_research_goal_complete=False,next='Inspect full supplementary tables and prepare all-method report; no original criterion replacement')
        save(out/'summary.json',result)
        print(result,flush=True)
    except BaseException as exc:
        save(out/'failure.json',dict(utc=now(),error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise


if __name__=='__main__': main()
