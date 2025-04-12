# train.py
import argparse
import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    args = parser.parse_args()
    
    # Read the train data
    train_data = pd.read_csv(os.path.join(args.train, 'train.csv'))
    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    
    # Train a simple model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Save the model
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
