# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
from car_data_prep import prepare_data

# Load the dataset
df = pd.read_csv('https://raw.githubusercontent.com/raza783/cars-project-part-2/main/dataset.csv')  # עדכן את הנתיב לקובץ הנתונים שלך

# Process the data
df_processed = prepare_data(df)

# Define relevant columns
features = ['manufactor', 'Year', 'model', 'Hand', 'Gear', 'capacity_Engine', 'Engine_type', 
            'Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Km', 'Test_days', 
            'Supply_score', 'Pic_num', 'Color']
target = 'Price'

X = df_processed[features]
y = df_processed[target]

# Define the preprocessing pipelines for numeric and categorical data
numeric_features = ['Year', 'Hand', 'capacity_Engine', 'Km', 'Test_days', 'Supply_score', 'Pic_num']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

categorical_features = ['manufactor', 'model', 'Gear', 'Engine_type', 'Prev_ownership', 'Curr_ownership', 'Area', 'City', 'Color']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

# Create the ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Create a pipeline that includes the preprocessor and the model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', ElasticNet(random_state=42))])

# Define the parameter grid for searching
param_grid = {
    'regressor__alpha': [0.01, 0.1, 1.0, 10.0, 100.0],
    'regressor__l1_ratio': [0.1, 0.5, 0.7, 0.9, 1.0]
}

# Define GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='neg_root_mean_squared_error', cv=10, n_jobs=-1)

# Perform the grid search
grid_search.fit(X, y)

# Display the best parameters found
print("Best parameters found: ", grid_search.best_params_)
print("Best RMSE: ", -grid_search.best_score_)

# Train the model with the best parameters
best_model = grid_search.best_estimator_

# Save the trained model to a file
joblib.dump(best_model, 'trained_model.pkl')
