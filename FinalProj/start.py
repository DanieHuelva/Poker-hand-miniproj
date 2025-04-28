# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.decomposition import PCA


data = pd.read_csv("FinalProj/forestfires.csv")

# Quick overview
print(data.head())
print(data.info())

# Identify features and target
target = 'area'
X = data.drop(columns=[target])
y = data[target]

# Handle skewness in target (OPTIONAL but smart)
y = np.log1p(y)  # log(1 + area) to handle severe skewness

# Identify categorical and numerical features
categorical_features = ['month', 'day']
numerical_features = [col for col in X.columns if col not in categorical_features]

# Build preprocessing pipeline
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# (Optional) Dimensionality Reduction setup
pca = PCA(n_components=0.95)  # keep 95% variance

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(random_state=42)
}

# Train and evaluate models
for name, model in models.items():
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('pca', pca),
                               ('model', model)])
    
    # 5-fold cross validation
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X_train, y_train, scoring='neg_root_mean_squared_error', cv=cv)
    
    print(f"{name}: Average RMSE = {-scores.mean():.4f}")

# Pick best model (e.g., Random Forest)
final_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                 ('pca', pca),
                                 ('model', RandomForestRegressor(random_state=42))])

final_pipeline.fit(X_train, y_train)

# Predict and evaluate on test set
y_pred = final_pipeline.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print("\nFinal Model Evaluation on Test Set:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")

# Visualization: Residual Plot
plt.scatter(y_test, y_pred)
plt.xlabel("Actual log(1 + Area)")
plt.ylabel("Predicted log(1 + Area)")
plt.title("Actual vs Predicted (Random Forest)")
plt.show()

# Visualization: Residuals
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Distribution (Random Forest)")
plt.show()