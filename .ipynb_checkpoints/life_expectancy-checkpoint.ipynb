# 📦 Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# 📥 Load the dataset
df = pd.read_csv('Life Expectancy Data.csv')
df.columns = df.columns.str.strip()  # Remove extra spaces in column names

# 📌 Visualize missing values
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.tight_layout()
plt.show()

# 🧹 Drop missing values for simplicity
df = df.dropna()

# 🎯 Define features and target
X_full = df.drop(columns=['Life expectancy '])
y = df['Life expectancy ']

# Encode categorical variables if present
X_full = pd.get_dummies(X_full, drop_first=True)

# 🔍 Select top 5 features using SelectKBest
selector = SelectKBest(score_func=f_regression, k=5)
X_top5_array = selector.fit_transform(X_full, y)
selected_features = X_full.columns[selector.get_support()]
X_top5 = X_full[selected_features]
print("Top 5 Selected Features:", selected_features.tolist())

# 🧪 Train-test split for both sets
X_top5_train, X_top5_test, y_train, y_test = train_test_split(X_top5, y, test_size=0.2, random_state=42)
X_full_train, X_full_test, _, _ = train_test_split(X_full, y, test_size=0.2, random_state=42)

# ⚖️ Scaling for linear regression
scaler = StandardScaler()
X_top5_train_scaled = scaler.fit_transform(X_top5_train)
X_top5_test_scaled = scaler.transform(X_top5_test)
X_full_train_scaled = scaler.fit_transform(X_full_train)
X_full_test_scaled = scaler.transform(X_full_test)

# --- MODELS ON TOP 5 FEATURES ---

# 📈 Linear Regression
lr_model_top5 = LinearRegression()
lr_model_top5.fit(X_top5_train_scaled, y_train)
y_pred_lr = lr_model_top5.predict(X_top5_test_scaled)

# 🌲 Random Forest (default)
rf_model_base = RandomForestRegressor(random_state=42)
rf_model_base.fit(X_top5_train, y_train)
y_pred_rf_base = rf_model_base.predict(X_top5_test)

# 🌳 Decision Tree (default)
dt_model_base = DecisionTreeRegressor(random_state=42)
dt_model_base.fit(X_top5_train, y_train)
y_pred_dt_base = dt_model_base.predict(X_top5_test)

# 📊 Baseline Evaluation with extended diagnostics
model_scores = []

def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    residuals = y_true - y_pred
    model_scores.append({'Model': name, 'RMSE': rmse, 'R2': r2})
    print(f"\n{name} Evaluation")
    print("RMSE:", rmse)
    print("R2 Score:", r2)
    print("Mean Residual:", residuals.mean())
    print("Residual Std Dev:", residuals.std())
    if hasattr(y_pred, 'shape'):
        plt.figure(figsize=(6, 4))
        sns.histplot(residuals, kde=True, bins=30)
        plt.title(f'Residual Distribution - {name}')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.show()

print("\n--- Baseline Model Performance (Top 5 Features) ---")
evaluate_model("Linear Regression", y_test, y_pred_lr)
evaluate_model("Random Forest (Base)", y_test, y_pred_rf_base)
evaluate_model("Decision Tree (Base)", y_test, y_pred_dt_base)

# --- HYPERPARAMETER TUNING ON TOP 5 FEATURES ---

# Random Forest
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(RandomForestRegressor(random_state=42), rf_params, cv=3, n_jobs=-1)
rf_grid.fit(X_top5_train, y_train)
rf_model = rf_grid.best_estimator_
y_pred_rf = rf_model.predict(X_top5_test)
print("\nBest RF Params (Top 5):", rf_grid.best_params_)

# Decision Tree
dt_params = {
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}
dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=3, n_jobs=-1)
dt_grid.fit(X_top5_train, y_train)
dt_model = dt_grid.best_estimator_
y_pred_dt = dt_model.predict(X_top5_test)
print("\nBest DT Params (Top 5):", dt_grid.best_params_)

# 📊 Tuned Evaluation (Top 5)
print("\n--- Tuned Model Performance (Top 5 Features) ---")
evaluate_model("Random Forest (Tuned)", y_test, y_pred_rf)
evaluate_model("Decision Tree (Tuned)", y_test, y_pred_dt)

# 🔁 Compare performance using ALL features
lr_full = LinearRegression()
lr_full.fit(X_full_train_scaled, y_train)
y_pred_full = lr_full.predict(X_full_test_scaled)
print("\n--- Linear Regression (All Features) ---")
evaluate_model("Linear Regression (All Features)", y_test, y_pred_full)

# 📊 Compare all model performances
score_df = pd.DataFrame(model_scores)
print("\nModel Comparison Summary:")
print(score_df.sort_values(by='RMSE'))

# 📈 Feature importance plot for best model (Random Forest tuned)
plt.figure(figsize=(8, 6))
importances = pd.Series(rf_model.feature_importances_, index=X_top5.columns)
importances.sort_values().plot(kind='barh')
plt.title('Feature Importances - Random Forest (Top 5 Features)')
plt.tight_layout()
plt.show()

# 💾 Save best model
joblib.dump(rf_model, 'random_forest_life_expectancy.pkl')
print("\nRandom Forest model saved as 'random_forest_life_expectancy.pkl'")

# 📤 Export predictions
predictions_df = pd.DataFrame({
    'Actual': y_test,
    'LR_Pred_Top5': y_pred_lr,
    'RF_Tuned_Top5': y_pred_rf,
    'DT_Tuned_Top5': y_pred_dt,
    'LR_Full': y_pred_full
})
predictions_df.to_csv('life_expectancy_predictions.csv', index=False)
print("Predictions exported to 'life_expectancy_predictions.csv'")

# 📉 Visualizations (optional)
def plot_predictions(y_true, y_pred, title):
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=y_true, y=y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title(f'{title} - Actual vs. Predicted')
    plt.tight_layout()
    plt.show()

plot_predictions(y_test, y_pred_rf, 'Random Forest (Tuned, Top 5)')
