"""353 explicit adapter to frozen, audited 345 scoring arithmetic."""
from pathlib import Path
import traceback
import unvisited_online_suite_v1 as suite
import run_unvisited_online_v1 as pipeline
import evaluate_support_language_online_v1 as scoring

if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/suite.BASE/'development_evaluation_v1'; out.mkdir(parents=True, exist_ok=False)
    scoring.suite = suite; scoring.pipeline = pipeline
    try: scoring.run(root, out)
    except Exception:
        suite.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
