import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import joblib
import os
import warnings

warnings.filterwarnings('ignore')

# Set figure aesthetics
sns.set_theme(style="whitegrid")
plt.rcParams.update({'figure.dpi': 300, 'figure.figsize': (10, 6)})

print("==============================================")
print(" EV Charging Cost Prediction - Data Science Pipeline")
print("==============================================\n")

# ---------------------------------------------------------
# STEP 1: LOAD THE DATASET
# ---------------------------------------------------------
print("1. Loading the dataset...")
data_path = 'Ev charging dataset.csv'
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Dataset not found at {data_path}. Please check the path.")

df = pd.read_csv(data_path)
print(f"   => Dataset loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.\n")

# ---------------------------------------------------------
# STEP 2: EXPLORATORY DATA ANALYSIS (EDA) & VISUALIZATION
# ---------------------------------------------------------
print("2. Performing Exploratory Data Analysis (EDA)...")
eda_dir = 'eda_plots'
os.makedirs(eda_dir, exist_ok=True)

# Graph 1: Distribution of Charging Cost
plt.figure()
sns.histplot(df['Charging Cost (USD)'], bins=30, kde=True, color='blue')
plt.title('Distribution of Charging Cost (USD)')
plt.xlabel('Charging Cost (USD)')
plt.ylabel('Frequency')
plt.savefig(f'{eda_dir}/target_distribution.png')
plt.close()
print(f"   => Saved distribution plot to '{eda_dir}/target_distribution.png'")

# Graph 2: Energy Consumed vs Charging Cost
plt.figure()
sns.scatterplot(data=df, x='Energy Consumed (kWh)', y='Charging Cost (USD)', alpha=0.6, color='green')
plt.title('Energy Consumed vs Charging Cost')
plt.xlabel('Energy Consumed (kWh)')
plt.ylabel('Charging Cost (USD)')
plt.savefig(f'{eda_dir}/energy_vs_cost.png')
plt.close()
print(f"   => Saved scatter plot to '{eda_dir}/energy_vs_cost.png'")

# Graph 3: Average Charging Cost by Charger Type
plt.figure()
sns.barplot(data=df, x='Charger Type', y='Charging Cost (USD)', ci=None, palette='viridis')
plt.title('Average Charging Cost by Charger Type')
plt.xlabel('Charger Type')
plt.ylabel('Average Cost (USD)')
plt.savefig(f'{eda_dir}/cost_by_charger_type.png')
plt.close()
print(f"   => Saved bar chart to '{eda_dir}/cost_by_charger_type.png'")

# Graph 4: Average Cost by Time of Day
plt.figure()
sns.barplot(data=df, x='Time of Day', y='Charging Cost (USD)', ci=None, palette='magma')
plt.title('Average Charging Cost by Time of Day')
plt.xlabel('Time of Day')
plt.ylabel('Average Cost (USD)')
plt.savefig(f'{eda_dir}/cost_by_time.png')
plt.close()
print(f"   => Saved bar chart to '{eda_dir}/cost_by_time.png'")

# Graph 5: Correlation Heatmap
numeric_cols = df.select_dtypes(include=[np.number]).columns
plt.figure(figsize=(12, 8))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig(f'{eda_dir}/correlation_heatmap.png')
plt.close()
print(f"   => Saved correlation heatmap to '{eda_dir}/correlation_heatmap.png'\n")

# ---------------------------------------------------------
# STEP 3: PREPROCESSING & FEATURE ENGINEERING
# ---------------------------------------------------------
print("3. Preprocessing Data and Engineering Features...")

# We will drop columns that shouldn't be used for predicting individual sessions based on identifiers
# Or at least separate them. User ID and Station ID are identifiers.
# Charging Start Time and End Time can be used to engineer features but raw strings are dropped.
columns_to_drop = ['User ID', 'Charging Station ID', 'Charging Start Time', 'Charging End Time']

# Let's engineer features first before dropping
# Convert to datetime and calculate duration explicitly just in case
# (We already have Charging Duration (hours) but let's make an Hour of Day feature)
df['Charging Start Time'] = pd.to_datetime(df['Charging Start Time'])
df['Start_Hour'] = df['Charging Start Time'].dt.hour
df['Is_Weekend'] = df['Charging Start Time'].dt.dayofweek // 5  # 5,6 are weekend (1), else 0

# Now drop raw times and identifiers
df_processed = df.drop(columns=columns_to_drop)

# Define Target and Features
target_col = 'Charging Cost (USD)'

# Before splitting, let's drop any rows where the target is entirely missing!
df_processed = df_processed.dropna(subset=[target_col])

X = df_processed.drop(columns=[target_col])
y = df_processed[target_col]

# Identify numerical and categorical columns
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

print(f"   => Numerical Features: {len(num_features)}")
print(f"   => Categorical Features: {len(cat_features)}")

# Build Preprocessing Pipeline
# Numerical: Impute missing with median -> Standard Scaler
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Impute missing with mode (most_frequent) -> One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine into a ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_features),
        ('cat', categorical_transformer, cat_features)
    ])

# Pre-split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("   => Split data into 80% Training and 20% Testing sets.\n")

# ---------------------------------------------------------
# STEP 4: MODEL TRAINING
# ---------------------------------------------------------
print("4. Training Machine Learning Models...")

models = {
    'Baseline (Mean)': DummyRegressor(strategy='mean'),
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = {}
best_r2 = -float('inf')
best_model_name = ""
best_model_pipeline = None

for name, model in models.items():
    print(f"   => Training {name}...")
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('model', model)])
    
    # Train
    pipeline.fit(X_train, y_train)
    
    # Predict
    y_pred = pipeline.predict(X_test)
    
    # Evaluate
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    results[name] = {'R2': r2, 'RMSE': rmse, 'MAE': mae}
    
    # Keep track of the best model (excluding baseline)
    if name != 'Baseline (Mean)' and r2 > best_r2:
        best_r2 = r2
        best_model_name = name
        best_model_pipeline = pipeline

print("\n---------------------------------------------------------")
print("5. MODEL EVALUATION RESULTS")
print("---------------------------------------------------------")
print(f"{'Model':<20} | {'R2 Score':<10} | {'RMSE':<10} | {'MAE':<10}")
print("-" * 59)
for name, metrics in results.items():
    print(f"{name:<20} | {metrics['R2']:<10.4f} | {metrics['RMSE']:<10.4f} | {metrics['MAE']:<10.4f}")
print("-" * 59)

# ---------------------------------------------------------
# STEP 6: SAVE BEST MODEL & FEATURE IMPORTANCES
# ---------------------------------------------------------
print(f"\n6. Saving the best model ({best_model_name}) and extracting insights...")

# Extract Feature Importances if the model supports it
# Extract encoded column names
preprocessor_fitted = best_model_pipeline.named_steps['preprocessor']
model_fitted = best_model_pipeline.named_steps['model']

# Get categorical feature names after one-hot encoding
cat_encoder = preprocessor_fitted.named_transformers_['cat'].named_steps['onehot']
cat_feature_names = cat_encoder.get_feature_names_out(cat_features).tolist()
all_feature_names = num_features + cat_feature_names

# Plot feature importances for Tree-based models
if hasattr(model_fitted, 'feature_importances_'):
    importances = model_fitted.feature_importances_
    indices = np.argsort(importances)[::-1][:15] # Top 15
    
    plt.figure(figsize=(10, 6))
    plt.title(f"Top 15 Feature Importances ({best_model_name})")
    plt.bar(range(15), importances[indices], align="center", color='purple')
    plt.xticks(range(15), [all_feature_names[i] for i in indices], rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f'{eda_dir}/feature_importances.png')
    plt.close()
    print(f"   => Saved feature importances plot to '{eda_dir}/feature_importances.png'")

# Save the Pipeline (Includes preprocessing AND model)
model_filepath = 'ev_cost_predictor.pkl'
joblib.dump(best_model_pipeline, model_filepath)
print(f"   => Model saved successfully to '{model_filepath}'")

print("\nPipeline Execution Completed Successfully!")
print("==============================================")
