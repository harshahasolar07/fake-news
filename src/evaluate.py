
import mlflow
import yaml
from mlflow.tracking import MlflowClient


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_production_model_metric(config_path='config.yaml'):
    """
    Get the F1 score of the current production model from Model Registry.
    
    Returns:
        float: F1 score of production model, or baseline threshold if no model in production
    """
    config = load_config(config_path)
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = MlflowClient()
    
    try:
        # Get production model
        model_name = config['model']['name']
        production_models = client.get_latest_versions(model_name, stages=["Production"])
        
        if not production_models:
            print("⚠️  No production model found. Using baseline threshold.")
            return config['model']['baseline_f1_threshold']
        
        # Get the latest production model
        production_model = production_models[0]
        run_id = production_model.run_id
        
        # Get metrics from the run
        run = client.get_run(run_id)
        f1_score = run.data.metrics.get('f1_score', 0.0)
        
        print(f"📊 Production model F1 score: {f1_score:.4f}")
        return f1_score
        
    except Exception as e:
        print(f"⚠️  Error getting production model: {e}")
        print(f"Using baseline threshold: {config['model']['baseline_f1_threshold']}")
        return config['model']['baseline_f1_threshold']


def get_latest_model_metric(config_path='config.yaml'):
    """
    Get the F1 score of the most recent model run.
    
    Returns:
        tuple: (f1_score, run_id) of the latest model
    """
    config = load_config(config_path)
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = MlflowClient()
    
    # Get the experiment
    experiment = client.get_experiment_by_name(config['mlflow']['experiment_name'])
    
    if not experiment:
        raise ValueError("No experiment found!")
    
    # Get all runs, sorted by start time (most recent first)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1
    )
    
    if not runs:
        raise ValueError("No runs found in experiment!")
    
    latest_run = runs[0]
    f1_score = latest_run.data.metrics.get('f1_score', 0.0)
    run_id = latest_run.info.run_id
    
    print(f"📊 Latest model F1 score: {f1_score:.4f}")
    
    return f1_score, run_id


def should_deploy_model(config_path='config.yaml'):
    """
    Determine if the latest model should be deployed based on F1 score comparison.
    
    This is the CONDITIONAL LOGIC GATE required by the assessment.
    
    Returns:
        tuple: (should_deploy: bool, new_f1: float, prod_f1: float, run_id: str)
    """
    print("\n" + "="*60)
    print("🔍 CONDITIONAL DEPLOYMENT LOGIC - MODEL COMPARISON")
    print("="*60)
    
    # Get production model F1 score
    prod_f1 = get_production_model_metric(config_path)
    
    # Get latest model F1 score
    new_f1, run_id = get_latest_model_metric(config_path)
    
    # Decision logic
    should_deploy = new_f1 >= prod_f1
    
    print(f"\n📈 Comparison Results:")
    print(f"   Production F1: {prod_f1:.4f}")
    print(f"   New Model F1:  {new_f1:.4f}")
    print(f"   Difference:    {new_f1 - prod_f1:+.4f}")
    
    if should_deploy:
        print(f"\n✅ DEPLOY: New model meets threshold (F1 >= {prod_f1:.4f})")
        print(f"   Proceeding to deployment stage...")
    else:
        print(f"\n❌ BLOCKED: New model below threshold (F1 < {prod_f1:.4f})")
        print(f"   Model will NOT be deployed.")
    
    print("="*60 + "\n")
    
    return should_deploy, new_f1, prod_f1, run_id


def promote_to_production(run_id, config_path='config.yaml'):
    """
    Promote a model to production stage in Model Registry.
    
    Args:
        run_id: MLflow run ID of the model to promote
    """
    config = load_config(config_path)
    
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    client = MlflowClient()
    
    model_name = config['model']['name']
    
    # Get all versions of the model
    versions = client.search_model_versions(f"name='{model_name}'")
    
    # Find the version corresponding to this run_id
    target_version = None
    for version in versions:
        if version.run_id == run_id:
            target_version = version.version
            break
    
    if target_version:
        # Transition to production
        client.transition_model_version_stage(
            name=model_name,
            version=target_version,
            stage="Production"
        )
        print(f"✅ Model version {target_version} promoted to Production!")
    else:
        print(f"⚠️  Could not find model version for run {run_id}")


if __name__ == "__main__":
    should_deploy, new_f1, prod_f1, run_id = should_deploy_model()
    
    if should_deploy:
        promote_to_production(run_id)