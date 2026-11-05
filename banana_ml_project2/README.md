# Banana KMnO4 - Remaining Shelf-Life Prediction

Machine-learning model for predicting the remaining shelf life of bananas using storage day, ethylene concentration and KMnO4 treatment dosage.

Objective
Predict:

How many days remain before a banana becomes over-ripe or decayed?

The project combines experimental banana-storage data with machine-learning models to estimate remaining shelf life.

Inputs
The models use:

Storage day
Ethylene concentration (ppm)
KMnO4 dosage (0 g, 10 g, or 20 g)
Target:

Remaining shelf life (days)
Peel colour and quality stage are excluded because spoilage is defined using those variables.

Models Evaluated
Linear Regression
Decision Tree Regression
Random Forest Regression
K-Nearest Neighbours (KNN)
Support Vector Regression (SVR)
Dummy Regressor (benchmark)
Random Forest was selected as the primary model because it provides feature-importance analysis.

Validation
Models are evaluated using GroupKFold cross-validation grouped by banana ID.

Performance metrics:

MAE (Mean Absolute Error)
RMSE (Root Mean Squared Error)
R² (Coefficient of Determination)
Project Structure
config.py            Configuration settings
data.py              Data loading and cleaning
dataset.py           Shelf-life dataset construction
eda.py               Exploratory analysis and plots
model.py             Machine-learning models
train_evaluate.py    Validation and evaluation
main.py              Main workflow
predict.py           Prediction utility
app.py               Streamlit application
DATA.csv             Experimental dataset
Running
pip install -r requirements.txt
python main.py
Run the web application:

streamlit run app.py
Outputs
Running main.py generates:

trend_plots.png
predicted_vs_actual.png
feature_importance.png
banana_remaining_days_rf_model.joblib
Notes
Bananas that do not reach spoilage are treated as censored observations and excluded from model training.
This is a proof-of-concept study based on a small experimental dataset.
Results should be interpreted together with the limitations discussed in the dissertation.
Author
Bornwise Koketso Maenetja
Final-Year Chemical Engineering Project
University of Johannesburg
