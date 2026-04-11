from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import PowerTransformer, StandardScaler

def build_no_pipeline(numeric_features):
    numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('power', PowerTransformer(method='yeo-johnson')),('scaler', StandardScaler()) ])
    
    preprocessor = ColumnTransformer(transformers=[('num', numeric_transformer, numeric_features)]) 
    return preprocessor
