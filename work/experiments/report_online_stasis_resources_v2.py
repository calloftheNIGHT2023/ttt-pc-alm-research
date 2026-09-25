"""305 layout-only v2: move legend into empty upper-right plot space."""
import argparse
from pathlib import Path
from matplotlib.axes import Axes
import report_online_stasis_resources_v1 as original
from diagnose_gradient_flat_split_states_v1 import save,sha


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_resources/report_v2'
    out.mkdir(parents=True,exist_ok=False)
    saved=Axes.legend
    def legend(self,*args,**kwargs):
        kwargs['loc']='upper right'
        return saved(self,*args,**kwargs)
    try:
        Axes.legend=legend
        original.run(root,out)
        save(out/'layout_revision.json',dict(layout_only=True,change='legend lower-right to upper-right',
            source_sha256={Path(__file__).name:sha(Path(__file__)),Path(original.__file__).name:sha(Path(original.__file__))},
            previous_visual_review_sha256=sha(root/'results/online_stasis_resources/report_v1/visual_review.json')))
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise
    finally:
        Axes.legend=saved
