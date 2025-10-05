from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from ames_housing.preprocessing import MissingValueHandler, PureTransformer, FeatureTransformer

def get_final_pipeline():
    """Returns the final trained pipeline with preprocessing + XGB model."""
    final_model = Pipeline([
        ('missingvaluehandler', MissingValueHandler()),
        ('puretransformer', PureTransformer(add_features=False)),
        ('featuretransformer', FeatureTransformer(
            scale_features=False,
            transform_features=False,
            drop_year_features=False
        )),
        ('model', XGBRegressor(
            n_estimators=326,
            max_depth=3,
            learning_rate=0.0708694921012168,
            subsample=0.7503553140583077,
            colsample_bytree=0.9458641568561671,
            random_state=42,
            verbosity=0
        ))
    ])
    return final_model
