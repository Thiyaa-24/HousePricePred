import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ==========================================
# 1. LOAD DATA
# ==========================================

file_path = r"D:\62-DS\HousePricePred\house_pred_modified_data.csv"

df = pd.read_csv(file_path)

print("Original shape:", df.shape)


# ==========================================
# 2. REMOVE ZERO PRICE
# ==========================================

df = df[df["price"] > 0].copy()

print("Shape after removing zero prices:", df.shape)


# ==========================================
# 3. DATE PROCESSING
# ==========================================

df["date"] = pd.to_datetime(df["date"])

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df = df.drop("date", axis=1)


# ==========================================
# 4. REMOVE UNNECESSARY / LEAKAGE COLUMNS
# ==========================================

df = df.drop(
    ["street", "price_per_sqft"],
    axis=1
)


# ==========================================
# 5. ONE-HOT ENCODING
# ==========================================

df = pd.get_dummies(
    df,
    columns=["city", "statezip"],
    drop_first=True
)


# ==========================================
# 6. FEATURES AND TARGET
# ==========================================

X = df.drop("price", axis=1)
y = df["price"]


# ==========================================
# 7. TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# ==========================================
# 8. DEFINE MODELS
# ==========================================

models = {
    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Lasso Regression": Lasso(alpha=1000, max_iter=10000),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.05,
        random_state=42
    )
}


# ==========================================
# 9. TRAIN AND EVALUATE
# ==========================================

results = []

for name, model in models.items():

    print("\nTraining:", name)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })


# ==========================================
# 10. COMPARISON TABLE
# ==========================================

results_df = pd.DataFrame(results)

results_df = results_df.sort_values(
    "R2 Score",
    ascending=False
)

print("\n==========================================")
print("       MODEL COMPARISON RESULTS")
print("==========================================")

print(
    results_df.to_string(index=False)
)


# ==========================================
# 11. BEST MODEL
# ==========================================

best_model = results_df.iloc[0]

print("\n==========================================")
print("             BEST MODEL")
print("==========================================")

print("Model:", best_model["Model"])
print("R² Score:", best_model["R2 Score"])
print("RMSE:", best_model["RMSE"])
print("MAE:", best_model["MAE"])


# ==========================================
# 12. R² COMPARISON GRAPH
# ==========================================

plt.figure(figsize=(10, 6))

plt.bar(
    results_df["Model"],
    results_df["R2 Score"]
)

plt.xlabel("Model")
plt.ylabel("R² Score")
plt.title("R² Score Comparison of Regression Models")

plt.xticks(
    rotation=30,
    ha="right"
)

plt.tight_layout()
plt.show()