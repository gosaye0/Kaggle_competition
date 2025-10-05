import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from datetime import datetime


class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.n_features_in_ = None
        self.numeric_features_ = None
        self.categorical_features_ = None
        self.is_fitted_ = False

        self.learned_modes_ = {}
        self.learned_medians_ = {}

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=["object", "category"]).columns.tolist()

        # Learn medians for numeric features
        for feature in self.numeric_features_:
            self.learned_medians_[feature] = X[feature].median()

        # Learn modes for categorical features (handle all-NaN case)
        for feature in self.categorical_features_:
            mode = X[feature].mode()
            self.learned_modes_[feature] = mode[0] if not mode.empty else "None"

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        check_is_fitted(self, "is_fitted_")

        # Column validation
        for col in X.columns:
            if col not in self.feature_names_in_:
                raise ValueError(f"Unexpected column {col} not seen during fit")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        X_processed = X.copy()

        # Handling garage related features
        garage_cat = ["GarageType", "GarageFinish", "GarageQual", "GarageCond"]
        garage_num = ["GarageYrBlt", "GarageCars", "GarageArea"]
        mask_garage_na = X_processed['GarageType'].isna()
        X_processed.loc[mask_garage_na, garage_cat] = 'None'
        X_processed.loc[mask_garage_na, garage_num] = 0
        X_processed['GarageYrBlt'] = X_processed['GarageYrBlt'].fillna(X_processed['YearBuilt'])
        for f in garage_cat:
            X_processed[f] = X_processed[f].fillna(self.learned_modes_[f])
        for f in garage_num:
            X_processed[f] = X_processed[f].fillna(self.learned_medians_[f])

        # Handling basement related features
        basement_cat = ["BsmtQual", "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2"]
        basement_num = ["BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF", "BsmtFullBath", "BsmtHalfBath"]
        mask_bsmt_na = X_processed["BsmtQual"].isna()
        X_processed.loc[mask_bsmt_na, basement_cat] = "None"
        X_processed.loc[mask_bsmt_na, basement_num] = 0
        for f in basement_cat:
            X_processed[f] = X_processed[f].fillna(self.learned_modes_[f])
        for f in basement_num:
            X_processed[f] = X_processed[f].fillna(self.learned_medians_[f])

        # Handling other "Not Available" features
        other_cat = ["Alley", "FireplaceQu", "PoolQC", "Fence", "MiscFeature"]
        other_num = ["PoolArea", "Fireplaces", "MiscVal"]
        X_processed[other_cat] = X_processed[other_cat].fillna("None")
        X_processed[other_num] = X_processed[other_num].fillna(0)

        # Handling truly missing values
        true_missing_num = ["LotFrontage", "MasVnrArea"]
        true_missing_cat = ["Electrical", "Functional", "Utilities", "Exterior1st", "Exterior2nd"]
        X_processed["LotFrontage"] = X_processed["LotFrontage"].fillna(self.learned_medians_["LotFrontage"])

        # Vectorized MasVnrArea handling
        mask_masvnr_none = X_processed["MasVnrType"] == "None"
        X_processed.loc[mask_masvnr_none, "MasVnrArea"] = 0
        X_processed["MasVnrArea"] = X_processed["MasVnrArea"].fillna(self.learned_medians_["MasVnrArea"])

        for f in true_missing_cat:
            X_processed[f] = X_processed[f].fillna(self.learned_modes_[f])

        # Handling remaining Features
        handled_features = garage_cat + garage_num + basement_cat + basement_num + \
                           other_cat + other_num + true_missing_cat + true_missing_num
        remaining_features = [f for f in self.feature_names_in_ if f not in handled_features]

        remaining_num = X_processed[remaining_features].select_dtypes(include=[np.number]).columns.tolist()
        remaining_cat = X_processed[remaining_features].select_dtypes(include=['object', 'category']).columns.tolist()
        for f in remaining_num:
            X_processed[f] = X_processed[f].fillna(self.learned_medians_[f])
        for f in remaining_cat:
            X_processed[f] = X_processed[f].fillna(self.learned_modes_[f])

        return X_processed

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_in_)




class PureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, add_features = True, drop_features = True, columns_to_drop = None):
        self.add_features = add_features
        self.drop_features = drop_features
        self.columns_to_drop = columns_to_drop
        
        self.feature_names_in_ = None
        self.feature_names_out_ = None
        self.n_features_in_ = None
        self.numeric_features_ = None
        self.categorical_features_ = None
        self.is_fitted_ = False

    def _ordinal_encoding(self, X):
        ordinal_mappings = {
            "ExterQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "ExterCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "BsmtCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "HeatingQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "KitchenQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "FireplaceQu": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "GarageQual": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "GarageCond": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1, "None": 0},
            "PoolQC": {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "None": 0},
    
            "LotShape": {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1},
            "LandSlope": {"Gtl": 3, "Mod": 2, "Sev": 1},
            "Utilities": {"AllPub": 4, "NoSewr": 3, "NoSeWa": 2, "ELO": 1},
            "BsmtExposure": {"Gd": 4, "Av": 3, "Mn": 2, "No": 1, "None": 0},
            "Functional": {
                "Typ": 8, "Min1": 7, "Min2": 6,
                "Mod": 5, "Maj1": 4, "Maj2": 3,
                "Sev": 2, "Sal": 1
            },
            "GarageFinish": {"Fin": 3, "RFn": 2, "Unf": 1, "None": 0},
            "PavedDrive": {"Y": 3, "P": 2, "N": 1, 'None':0},
            "Fence": {"GdPrv": 4, "MnPrv": 3, "GdWo": 2, "MnWw": 1, "None": 0},
        }

        for feature, mapping in ordinal_mappings.items():
            if feature in X.columns:
                X[feature] = X[feature].map(mapping).fillna(0).astype(int)

        return X
    def _add_features(self, X):

        # Garage features
        X["GarageCars_per_Area"] = X["GarageCars"] / X["GarageArea"].replace(0, np.nan)
    
        # Living vs lot area
        X["GrLivArea_per_LotArea"] = X["GrLivArea"] / X["LotArea"].replace(0, np.nan)
    
        # Basement and floor ratios
        total_area = (X["TotalBsmtSF"] + X["1stFlrSF"] + X["2ndFlrSF"]).replace(0, np.nan)
        X["BsmtSF_ratio"] = X["TotalBsmtSF"] / total_area
        X["FirstFlr_ratio"] = X["1stFlrSF"] / total_area
        X["SecondFlr_ratio"] = X["2ndFlrSF"] / total_area
    
        # Rooms and bedrooms
        X["TotalRooms_per_TotalArea"] = X["TotRmsAbvGrd"] / total_area
        X["TotalArea_per_LotArea"] = total_area / X["LotArea"].replace(0, np.nan)
        X["Bedroom_ratio"] = X["BedroomAbvGr"] / X["TotRmsAbvGrd"].replace(0, np.nan)
        X["Bath_ratio"] = (X["FullBath"] + 0.5 * X["HalfBath"]) / X["TotRmsAbvGrd"].replace(0, np.nan)
        X["BsmtBath_ratio"] = (X["BsmtFullBath"] + 0.5 * X["BsmtHalfBath"]) / (X["TotalBsmtSF"].replace(0, np.nan))
    
        # Porch and outdoor features
        X["TotalPorchSF"] = X["OpenPorchSF"] + X["EnclosedPorch"] + X["3SsnPorch"] + X["ScreenPorch"]
        X["Porch_per_LotArea"] = X["TotalPorchSF"] / X["LotArea"].replace(0, np.nan)
        X["Porch_per_GrLivArea"] = X["TotalPorchSF"] / X["GrLivArea"].replace(0, np.nan)
    

        # Misc features
        X["Fireplaces_per_TotalRooms"] = X["Fireplaces"] / X["TotRmsAbvGrd"].replace(0, np.nan)
        X["GarageArea_per_LotArea"] = X["GarageArea"] / X["LotArea"].replace(0, np.nan)
    
        # Replace infinities or NaNs
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        return X

    def _drop_features(self, X):

        if self.columns_to_drop:
            for feature in self.columns_to_drop:
                if feature in X.columns:
                    X = X.drop(columns = feature)
        return X

    def _apply_transformations(self, X):
        X_processed = X.copy()

        X_processed = self._ordinal_encoding(X_processed)
        if self.add_features:
            X_processed = self._add_features(X_processed)
        if self.drop_features:
            X_processed = self._drop_features(X_processed)
        
        return X_processed

        
    def fit(self, X, y = None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.numeric_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=["object", "category"]).columns.tolist()

        X = self._apply_transformations(X)
        self.feature_names_out_ = X.columns.tolist()
        self.is_fitted_ = True
        
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        check_is_fitted(self, "is_fitted_")

        for col in X.columns:
            if col not in self.feature_names_in_:
                raise ValueError(f"Unexpected column {col} not seen during fit")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

 
        X = self._apply_transformations(X)
        X = X[self.feature_names_out_]
        return X

    def get_feature_names_out(self):
        return self.feature_names_out_
    

class FeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, drop_year_features=True, scale_features=True, transform_features=True):
        self.scale_features = scale_features
        self.transform_features = transform_features
        self.drop_year_features = drop_year_features

        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.numerical_features_ = None
        self.categorical_features_ = None
        self.feature_names_out_ = None
        self.encoders_ = {}
        self.scalers_ = {}

        self.features_to_scale = [
            "LotFrontage", "LotArea", "MasVnrArea",
            "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
            "1stFlrSF", "2ndFlrSF", "LowQualFinSF", "GrLivArea",
            "GarageArea",
            "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "3SsnPorch",
            "ScreenPorch", "PoolArea", "MiscVal"
        ]

        self.features_to_transform = [
            "LotFrontage","LotArea","MasVnrArea","BsmtFinSF1","BsmtFinSF2","BsmtUnfSF",
            "TotalBsmtSF","1stFlrSF","2ndFlrSF","LowQualFinSF","GrLivArea","WoodDeckSF",
            "OpenPorchSF","EnclosedPorch","3SsnPorch","ScreenPorch","PoolArea","MiscVal"
        ]
        
        self.age_features = []
        self.is_fitted_ = False

    def _transformation(self, X):
        X = X.copy()
        # log transform skewed features
        tr_features  = [feature for feature in self.features_to_transform if feature in X.columns]
        X[tr_features] = X[tr_features].clip(lower=0)
        X[tr_features] = X[tr_features].apply(np.log1p)

        self.age_features = []
        year_features = ['YearBuilt', 'YearRemodAdd', 'GarageYrBlt', 'YrSold']
        current_year = datetime.now().year
        if 'YearBuilt' in X.columns:
            X['HouseAge'] = current_year - X['YearBuilt'].clip(lower=0)
            self.age_features.append('HouseAge')
        if 'YearRemodAdd' in X.columns:
            X['RemodAge'] = current_year - X['YearRemodAdd'].clip(lower=0)
            self.age_features.append('RemodAge')
        if 'GarageYrBlt' in X.columns:
            X['GarageAge'] = current_year - X['GarageYrBlt'].clip(lower=0)
            self.age_features.append('GarageAge')
        if 'YrSold' in X.columns:
            X['YearsSinceSold'] = current_year - X['YrSold'].clip(lower=0)
            self.age_features.append('YearsSinceSold')

        if self.age_features:
            X[self.age_features] = X[self.age_features].apply(np.log1p)
        if self.drop_year_features:
            X = X.drop(columns=year_features, errors="ignore")

        return X
        
    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)
        self.numerical_features_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_features_ = X.select_dtypes(include=['object', 'category']).columns.tolist()

        X_processed = X.copy()

        if self.categorical_features_:
            ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
            self.encoders_['ohe'] = ohe.fit(X_processed[self.categorical_features_])
            encoded = self.encoders_['ohe'].transform(X[self.categorical_features_])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoders_['ohe'].get_feature_names_out(self.categorical_features_),
                index=X.index
            )
            X_processed = X_processed.drop(columns=self.categorical_features_, errors="ignore")
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
        if self.transform_features:
            X_processed = self._transformation(X_processed)
        if self.scale_features:
            existing_features_to_scale = [feature for feature in self.features_to_scale if feature in X_processed.columns]
            cols_to_scale = existing_features_to_scale + self.age_features if self.transform_features else existing_features_to_scale
            sc = StandardScaler()
            self.scalers_['sc'] = sc.fit(X_processed[cols_to_scale])
            scaled = self.scalers_['sc'].transform(X_processed[cols_to_scale])
            scaled_df = pd.DataFrame(scaled, columns=cols_to_scale, index=X.index)
            X_processed[cols_to_scale] = scaled_df

        self.feature_names_out_ = X_processed.columns.tolist()

        self.is_fitted_ = True
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")

        check_is_fitted(self, "is_fitted_")

        for col in X.columns:
            if col not in self.feature_names_in_:
                raise ValueError(f"Unexpected column {col} not seen during fit")
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {X.shape[1]}")

        X_processed = X.copy()
        if self.categorical_features_:
            encoded = self.encoders_['ohe'].transform(X[self.categorical_features_])
            encoded_df = pd.DataFrame(
                encoded,
                columns=self.encoders_['ohe'].get_feature_names_out(self.categorical_features_),
                index=X.index
            )
            X_processed = X_processed.drop(columns=self.categorical_features_, errors="ignore")
            X_processed = pd.concat([X_processed, encoded_df], axis=1)
        if self.transform_features:
            X_processed = self._transformation(X_processed)

        # Scale
        if self.scale_features:
            existing_features_to_scale = [feature for feature in self.features_to_scale if feature in X_processed.columns]
            cols_to_scale = existing_features_to_scale + self.age_features if self.transform_features else existing_features_to_scale
            scaled = self.scalers_['sc'].transform(X_processed[cols_to_scale])
            scaled_df = pd.DataFrame(scaled, columns=cols_to_scale, index=X.index)
            X_processed[cols_to_scale] = scaled_df

        # Reorder to match training
        final_cols = [c for c in self.feature_names_out_ if c in X_processed.columns]
        X_processed = X_processed[final_cols]

        return X_processed
    
    def get_feature_names_out(self):
        return self.feature_names_out_