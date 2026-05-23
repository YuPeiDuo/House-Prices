import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score

df_train = pd.read_csv("D:/Machine_learning_Kaggle/house_price_predict/train.csv")
df_train_clean = df_train.copy()


features = ['MSSubClass','MSZoning','LotArea','LotShape','LotConfig','Neighborhood','HouseStyle','OverallQual','OverallCond','YearBuilt','YearRemodAdd','Exterior1st','Exterior2nd','MasVnrType','MasVnrArea','ExterQual','Foundation','BsmtExposure','BsmtFinType1','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','HeatingQC','1stFlrSF','GrLivArea','FullBath','KitchenQual','TotRmsAbvGrd','Fireplaces','GarageArea','MoSold','YrSold','SaleCondition']
X = df_train_clean[features]
y = df_train_clean['SalePrice']

X_train,X_val,y_train,y_val = train_test_split(X,y,test_size=0.2,random_state=42)

numeric_features = ['MSSubClass','OverallQual','OverallCond','YearBuilt','YearRemodAdd','MasVnrArea','BsmtFinSF1','BsmtUnfSF','TotalBsmtSF','1stFlrSF','GrLivArea','FullBath','TotRmsAbvGrd','Fireplaces','GarageArea','MoSold','YrSold']
categorical_features = ['MSZoning', 'LotShape', 'LotConfig', 'Neighborhood', 'HouseStyle','Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'Foundation','BsmtExposure', 'BsmtFinType1', 'HeatingQC', 'KitchenQual', 'SaleCondition']
numeric_transformer = Pipeline([('imputer',SimpleImputer(strategy='median')),('scaler',StandardScaler())])
categorical_transformer = Pipeline([('imputer',SimpleImputer(strategy='most_frequent')),('onehot',OneHotEncoder(drop='first',handle_unknown='ignore'))])

preprocessor = ColumnTransformer([
    ('num',numeric_transformer,numeric_features),
    ('cat',categorical_transformer,categorical_features)
])

pipeline = Pipeline([('preprocessor',preprocessor),('regressor',RandomForestRegressor(random_state=42))])
pipeline.fit(X_train,y_train)

preprocessor_fitted = pipeline.named_steps['preprocessor'] 
feature_names = preprocessor_fitted.get_feature_names_out()
rf_model = pipeline.named_steps['regressor']
importances = rf_model.feature_importances_

indices = np.argsort(importances)[::-1]
cumsum = np.cumsum(importances[indices]) 
threshold = 0.99
n_selected = np.argmax(cumsum >= threshold) + 1
selected_feature_names = feature_names[indices[:n_selected]]

original_feature_importance = {}
for name, imp in zip(feature_names, importances):
    if name.startswith('num__'):
        orig = name[5:]
    elif name.startswith('cat__'):
        orig = name[5:].split('_')[0]
    else:
        orig = name
    original_feature_importance[orig] = original_feature_importance.get(orig,0.0) + imp

sorted_orig = sorted(original_feature_importance.items(), key=lambda x: x[1], reverse=True)

orig_names = [k for k, v in sorted_orig]
orig_imps = [v for k, v in sorted_orig]
cumsum_orig = np.cumsum(orig_imps)
n_selected_orig = np.argmax(cumsum_orig >= threshold) + 1
selected_original_features = orig_names[:n_selected_orig]

features_selected = selected_original_features
X_selected = df_train_clean[features_selected]
X_train_sel, X_val_sel, y_train_sel, y_val_sel = train_test_split(X_selected, y, test_size=0.2, random_state=42)

numeric_features_selected = [f for f in features_selected if f in numeric_features]
categorical_features_selected = [f for f in features_selected if f in categorical_features]

preprocessor_sel = ColumnTransformer([
    ('num', numeric_transformer, numeric_features_selected),
    ('cat', categorical_transformer, categorical_features_selected)
])
pipeline_sel = Pipeline([
    ('preprocessor', preprocessor_sel),
    ('regressor', RandomForestRegressor(random_state=42))
])

pipeline_sel.fit(X_train_sel, y_train_sel)
y_pred_sel = pipeline_sel.predict(X_val_sel)
rmse_sel = np.sqrt(mean_squared_error(y_val_sel, y_pred_sel))
r2_sel = r2_score(y_val_sel, y_pred_sel)

param_grid = {
    'regressor__n_estimators': [50, 100, 150],
    'regressor__max_depth': [5, 10, None],
    'regressor__min_samples_split': [2, 5, 10]
}
grid = GridSearchCV(pipeline_sel, param_grid, cv=5, 
                    scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X_train_sel, y_train_sel)

best_model = grid.best_estimator_
y_pred_val = best_model.predict(X_val_sel)
rmse_val = np.sqrt(mean_squared_error(y_val_sel, y_pred_val))
r2_val = r2_score(y_val_sel, y_pred_val)

y_pred = pipeline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

df_test = pd.read_csv("D:/Machine_learning_Kaggle/house_price_predict/test.csv")
X_test = df_test[features_selected]  

y_pred = best_model.predict(X_test)

submission = pd.DataFrame({
    'Id': df_test['Id'],
    'SalePrice': y_pred
})

submission.to_csv('D:/Machine_learning_Kaggle/house_price_predict/submission.csv', index=False)