EV Charging Cost Prediction

A machine learning pipeline that predicts the cost of an EV charging session based on session details like energy consumed, charger type, and time of day.

Requirements
Python 3.8+ with the following libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, and joblib. Install them with:
pip install pandas numpy matplotlib seaborn scikit-learn joblib

How to Run
Open your terminal, navigate to the project folder, and run the scripts in this order:

**python ev_cost_prediction.py**

This loads the dataset, performs EDA, trains 5 models, and saves the best one as ev_cost_predictor.pkl. It also generates an eda_plots/ folder with 6 visualisation charts and prints a model comparison table.

Once that's done, run:

**python predict_and_analyze.py**

This loads the saved model, runs predictions on 5 sample sessions, prints the actual vs predicted cost for each, and saves the full results to prediction_analysis_results.csv.

Important
Both scripts must be run from the same folder as the dataset file. The filename must be exactly: Ev charging dataset.csv (case-sensitive).
