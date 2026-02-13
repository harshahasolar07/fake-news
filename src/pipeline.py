import os
import sys
from train import train_model
from evaluate import should_deploy_model, promote_to_production


def run_pipeline(model_type='logistic'):
    """
    Execute the complete MLOps pipeline:
    1. Train new model
    2. Evaluate against production baseline
    3. Conditionally deploy if metrics are better
    
    Args:
        model_type: Type of model to train ('logistic' or 'random_forest')
    """
    print("\n" + "🚀" + "="*58 + "🚀")
    print("   STARTING MLOPS PIPELINE - FAKE NEWS DETECTION")
    print("🚀" + "="*58 + "🚀\n")
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Step 1: Train new model
    print("📝 STEP 1: Training new model...")
    print("-" * 60)
    try:
        model, metrics = train_model(model_type=model_type)
        print(f"✅ Training completed successfully!")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        sys.exit(1)
    
    # Step 2: Evaluate and compare
    print("\n📊 STEP 2: Evaluating model performance...")
    print("-" * 60)
    try:
        should_deploy, new_f1, prod_f1, run_id = should_deploy_model()
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        sys.exit(1)
    
    # Step 3: Conditional deployment
    print("\n🎯 STEP 3: Deployment decision...")
    print("-" * 60)
    
    if should_deploy:
        print("Promoting model to production...")
        try:
            promote_to_production(run_id)
            print("✅ Model successfully deployed to production!")
            
            # Generate deployment artifacts
            print("\n📦 Generating deployment artifacts...")
            print("   - Model files saved in models/")
            print("   - API ready for serving")
            print("   - Run: python api/main.py to start API server")
            
        except Exception as e:
            print(f"⚠️  Deployment warning: {e}")
    else:
        print("❌ Deployment blocked - model does not meet threshold")
        print("   Suggestions:")
        print("   - Try different hyperparameters")
        print("   - Improve text preprocessing")
        print("   - Use a different model architecture")
        print("   - Collect more training data")
    
    print("\n" + "🏁" + "="*58 + "🏁")
    print("   PIPELINE EXECUTION COMPLETED")
    print("🏁" + "="*58 + "🏁\n")
    
    return should_deploy


if __name__ == "__main__":
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'logistic'
    run_pipeline(model_type=model_type)
 