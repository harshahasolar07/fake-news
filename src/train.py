
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
from datetime import datetime
import os

from data_preprocessing import TextPreprocessor, FeatureExtractor


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def train_model(config_path='config.yaml', model_type='logistic'):
    """
    Train a fake news detection model with MLflow tracking.
    
    Args:
        config_path: Path to configuration file
        model_type: Type of model ('logistic' or 'random_forest')
    
    Returns:
        Trained model and metrics
    """
    # Load configuration
    config = load_config(config_path)
    
    # Set up MLflow
    mlflow.set_tracking_uri(config['mlflow']['tracking_uri'])
    mlflow.set_experiment(config['mlflow']['experiment_name'])
    
    # Ensure models directory exists
    os.makedirs('models', exist_ok=True)
    
    # Start MLflow run
    with mlflow.start_run(run_name=f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
        
        # Log parameters
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("max_features", config['training']['max_features'])
        mlflow.log_param("ngram_range", config['training']['ngram_range'])
        
        # Load data
        print("Loading data...")
        df = pd.read_csv(config['data']['train_path'])
        
        # Preprocess text
        print("Preprocessing text...")
        preprocessor = TextPreprocessor()
        df = preprocessor.preprocess_dataframe(df, config['data']['text_column'])
        
        # Split data
        X = df['cleaned_text']
        y = df[config['data']['label_column']]
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=config['training']['test_size'],
            random_state=config['training']['random_state'],
            stratify=y
        )
        
        # Extract features
        print("Extracting features...")
        feature_extractor = FeatureExtractor(
            max_features=config['training']['max_features'],
            ngram_range=tuple(config['training']['ngram_range'])
        )
        
        X_train_vec = feature_extractor.fit_transform(X_train)
        X_test_vec = feature_extractor.transform(X_test)
        
        # Train model
        print(f"Training {model_type} model...")
        if model_type == 'logistic':
            model = LogisticRegression(
                max_iter=1000,
                random_state=config['training']['random_state'],
                C=1.0,
                solver='liblinear'
            )
            mlflow.log_param("C", 1.0)
            mlflow.log_param("solver", "liblinear")
            
        elif model_type == 'random_forest':
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=config['training']['random_state'],
                max_depth=20,
                min_samples_split=5
            )
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 20)
        
        model.fit(X_train_vec, y_train)
        
        # Evaluate
        print("Evaluating model...")
        y_pred = model.predict(X_test_vec)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted'),
            'recall': recall_score(y_test, y_pred, average='weighted'),
            'f1_score': f1_score(y_test, y_pred, average='weighted')
        }
        
        # Log metrics
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
            print(f"{metric_name}: {metric_value:.4f}")
        
        # Save artifacts
        print("Saving artifacts...")
        joblib.dump(preprocessor, 'models/preprocessor.pkl')
        joblib.dump(feature_extractor, 'models/feature_extractor.pkl')
        joblib.dump(model, 'models/model.pkl')
        
        # Log artifacts to MLflow
        mlflow.log_artifact('models/preprocessor.pkl')
        mlflow.log_artifact('models/feature_extractor.pkl')
        mlflow.sklearn.log_model(model, "model")
        
        # Log model to model registry
        model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
        mlflow.register_model(model_uri, config['model']['name'])
        
        print(f"✅ Model trained successfully! F1 Score: {metrics['f1_score']:.4f}")
        
        return model, metrics


if __name__ == "__main__":
    import sys
    model_type = sys.argv[1] if len(sys.argv) > 1 else 'logistic'
    train_model(model_type=model_type)