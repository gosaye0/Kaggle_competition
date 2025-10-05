import pandas as pd
from sklearn.model_selection import cross_validate


from ames_housing.data_loader import load_train_data
from ames_housing.model_training import get_final_pipeline

train_df = load_train_data()
train_X = train_df.drop(columns = "SalePrice")
train_y = train_df["SalePrice"].copy()
model = get_final_pipeline()

cv_results = cross_validate(
    model,
    train_X,
    train_y,
    cv = 5,
    scoring=['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2'],
    return_train_score=True
)

cv_df = pd.DataFrame(cv_results)
cv_df.to_csv("outputs/reports/cv_fold_details.csv", index=False)
print("Fold details saved successfully.")


train_rmse = -cv_df["train_neg_root_mean_squared_error"].mean() 
test_rmse = -cv_df["test_neg_root_mean_squared_error"].mean()
train_mae = -cv_df["train_neg_mean_absolute_error"].mean()
test_mae = -cv_df["test_neg_mean_absolute_error"].mean()
train_r2 = cv_df["train_r2"].mean()
test_r2 = cv_df["test_r2"].mean()

train_rmse_std = -cv_df["train_neg_root_mean_squared_error"].std() 
test_rmse_std = -cv_df["test_neg_root_mean_squared_error"].std()
train_mae_std = -cv_df["train_neg_mean_absolute_error"].std()
test_mae_std = -cv_df["test_neg_mean_absolute_error"].std()
train_r2_std = cv_df["train_r2"].std()
test_r2_std = cv_df["test_r2"].std()


evaluation_results = pd.DataFrame({
    "RMSE": [f"{train_rmse:,.0f} ± {train_rmse_std:,.0f}", f"{test_rmse:,.0f} ± {test_rmse_std:,.0f}"],
    "MAE": [f"{train_mae:,.0f} ± {train_mae_std:,.0f}", f"{test_mae:,.0f} ± {test_mae_std:,.0f}"],
    "R2": [f"{train_r2:.2f} ± {train_r2_std:.4f}", f"{test_r2:.2f} ± {test_r2_std:.4f}"]
    },
    index = ["Train", "Test"])
evaluation_results.to_csv("outputs/reports/cv_results.csv")
print("Metric results saved successfully.")
