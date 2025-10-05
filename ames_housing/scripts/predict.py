import joblib
import logging
import pandas as pd
from ames_housing.data_loader import load_test_data
from ames_housing.model_inference import load_model

logging.basicConfig(level=logging.INFO)

pipeline = load_model()

test_df = load_test_data()

predictions = pipeline.predict(test_df)

submissions = pd.DataFrame({
    "Id": test_df.index,
    "SalePrice": predictions
})

submissions.to_csv("outputs/predictions/submission.csv", index = False)
logging.info("Predictions saved successfully")