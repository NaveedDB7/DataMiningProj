import pandas as pd
import joblib
import numpy as np

print("==============================================")
print(" EV Charging Cost Prediction & Analysis")
print("==============================================\n")

print("1. Loading the trained model from 'ev_cost_predictor.pkl'...")
try:
    pipeline = joblib.load('ev_cost_predictor.pkl')
    print("   => Model loaded successfully.")
except FileNotFoundError:
    print("   => ERROR: 'ev_cost_predictor.pkl' not found. Please run ev_cost_prediction.py first.")
    exit(1)

print("\n2. Loading dataset to pick a sample for prediction...")
df_full = pd.read_csv('Ev charging dataset.csv')

# Preprocessing analogous to the training script to drop right columns
columns_to_drop = ['User ID', 'Charging Station ID', 'Charging Start Time', 'Charging End Time']

# Engineer features so the pipeline can process it
df_full['Charging Start Time'] = pd.to_datetime(df_full['Charging Start Time'])
df_full['Start_Hour'] = df_full['Charging Start Time'].dt.hour
df_full['Is_Weekend'] = df_full['Charging Start Time'].dt.dayofweek // 5

# Get 5 samples
sample_df = df_full.sample(n=5, random_state=42)

# Save actual targets for comparison
actual_costs = sample_df['Charging Cost (USD)'].values

# Prepare features
features = sample_df.drop(columns=columns_to_drop + ['Charging Cost (USD)'])

print("   => Simulating predictions on 5 random real-world sessions.\n")

print("3. Making Predictions and Analysis...")
predictions = pipeline.predict(features)

results = []
for i in range(len(sample_df)):
    pred = predictions[i]
    act = actual_costs[i]
    diff = pred - act
    dur = features.iloc[i]['Charging Duration (hours)']
    energy = features.iloc[i]['Energy Consumed (kWh)']
    ctype = features.iloc[i]['Charger Type']
    tod = features.iloc[i]['Time of Day']
    
    print(f"--- Session {i+1} ---")
    print(f"  * Key Context: {energy:.2f} kWh consumed over {dur:.2f} hours using {ctype} during the {tod}.")
    print(f"  * Actual Cost:   ${act:.2f}")
    if np.isnan(pred):
        print(f"  * Predicted Cost: ERROR (NaN)")
    else:    
        print(f"  * Predicted Cost: ${pred:.2f} (Diff: ${diff:+.2f})")
    
    # Analysis narrative (what, why, how)
    print("  * Analysis:")
    print(f"    - WHAT: The model predicted a cost of ${pred:.2f}.")
    print(f"    - HOW: The data was preprocessed (missing values imputed, categorical variables one-hot encoded, " 
          f"numeric features standardized) and fed into the trained {type(pipeline.named_steps['model']).__name__}.")
    print(f"    - WHY: The predicted value is derived by finding a linear weighting (if linear regression) or " 
          f"tree traversal logic (if trees). Features such as using a '{ctype}' and consuming '{energy:.1f}' kWh heavily "
          f"influenced this specific cost output based on historical trends.\n")

    results.append({
        'Session_Index': i+1,
        'Energy_Consumed_kWh': energy,
        'Charger_Type': ctype,
        'Actual_Cost_USD': act,
        'Predicted_Cost_USD': pred,
        'Difference': diff
    })

# Save results to CSV
output_file = 'prediction_analysis_results.csv'
pd.DataFrame(results).to_csv(output_file, index=False)
print(f"4. Saved detailed prediction results to '{output_file}'")
print("\nProcess Completed Successfully.")
print("==============================================")
