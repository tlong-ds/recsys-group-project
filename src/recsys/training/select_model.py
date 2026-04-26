import glob
import json
import os
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def select_best_model():
    metrics_files = glob.glob('metrics/experiments/*/*/evaluation_metrics.json')
    if not metrics_files:
        # Also check baseline metrics if they exist
        baseline_metrics = 'metrics/baseline/evaluation_metrics.json'
        if os.path.exists(baseline_metrics):
            metrics_files.append(baseline_metrics)
        else:
            logger.error("No evaluation metrics found.")
            return

    best_recall = -1
    best_file = None
    best_metrics = {}

    for f in metrics_files:
        try:
            with open(f, 'r') as jf:
                data = json.load(jf)
                # Assume Recall@20 is the primary metric
                recall = data.get('Recall@20', 0)
                if recall > best_recall:
                    best_recall = recall
                    best_file = f
                    best_metrics = data
        except Exception as e:
            logger.warning(f"Could not read {f}: {e}")

    if not best_file:
        logger.error("Could not find a best model.")
        return

    # Determine paths
    # If experiments: metrics/experiments/<data_version>/<model_profile>/evaluation_metrics.json
    # If baseline: metrics/baseline/evaluation_metrics.json
    parts = best_file.split(os.sep)
    if 'experiments' in parts:
        data_version = parts[2]
        model_profile = parts[3]
        model_source_dir = f"models/experiments/{data_version}/{model_profile}/latest"
    else:
        data_version = "baseline"
        model_profile = "baseline"
        model_source_dir = "models/trained/baseline/latest"

    summary = {
        "best_model": {
            "data_version": data_version,
            "model_profile": model_profile,
            "metrics": best_metrics,
            "source": model_source_dir
        }
    }

    os.makedirs('metrics', exist_ok=True)
    with open('metrics/best_model.json', 'w') as f:
        json.dump(summary, f, indent=4)
    
    logger.info(f"Best model: {data_version}/{model_profile} (Recall@20: {best_recall})")

    # Copy to latest
    target_dir = "models/trained/latest"
    if os.path.exists(target_dir):
        if os.path.islink(target_dir):
            os.unlink(target_dir)
        else:
            shutil.rmtree(target_dir)
    
    if os.path.exists(model_source_dir):
        shutil.copytree(model_source_dir, target_dir)
        logger.info(f"Copied {model_source_dir} to {target_dir}")
    else:
        logger.error(f"Source model directory {model_source_dir} does not exist!")

if __name__ == "__main__":
    select_best_model()
