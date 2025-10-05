import joblib
import pandas as pd
from ames_housing.model_training import get_final_pipeline
from ames_housing.data_loader import load_train_data

# Load the training data
train_df= load_train_data()

# Separate features and target
X = train_df.drop(columns='SalePrice')
y = train_df['SalePrice'].copy()

# Get the pipeline
pipeline = get_final_pipeline()

print("\nTraining the model, This may take few minutes!")
print("TRAINING...")
pipeline.fit(X, y)
print("Model trained successfully")

# Save the trained pipeline
joblib.dump(pipeline, "models/final_pipeline_v1.pkl")
print("Pipeline loaded successfully")