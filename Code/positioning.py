#!/usr/bin/env python
# coding: utf-8

# # ML and Ensemble

# In[17]:


import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression


# Suppress warnings
warnings.filterwarnings('ignore')

# Define the variable for the filename
N = "F04_Device2_2GHz"

# Load datasets dynamically based on N
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_training_{N}.csv', header=None)
test_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_test_{N}.csv', header=None)
# Prepare features and target variables
X_train = train_data.iloc[:, 6:-1]
X_test = test_data.iloc[:, 6:-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize base models
base_models = {
    "KNN": KNeighborsRegressor(n_neighbors=3),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(random_state=42, n_estimators=100)
}

# Step 1: Generate Out-of-Fold Predictions for Training Meta-Learner
kf = KFold(n_splits=5, shuffle=True, random_state=42)
meta_features = np.zeros((len(X_train), len(base_models)))  # Meta-features for training
test_meta_features = np.zeros((len(X_test), len(base_models)))  # Meta-features for test set

for i, (name, model) in enumerate(base_models.items()):
    test_preds_folds = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        val_preds = model.predict(X_fold_val)
        meta_features[val_idx, i] = val_preds  # Store OOF predictions
        
        # Predict on test set for this fold
        test_preds = model.predict(X_test_scaled)
        test_preds_folds.append(test_preds)
    
    # Average predictions across all folds for the test set
    test_meta_features[:, i] = np.mean(test_preds_folds, axis=0)

# Step 2: Train Meta-Learner
meta_learner = LinearRegression()
meta_learner.fit(meta_features, y_train)

# Step 3: Predict Using Meta-Learner
final_predictions = meta_learner.predict(test_meta_features)

# Step 4: Evaluate Ensemble
ensemble_mae = mean_absolute_error(y_test, final_predictions)

# Step 5: Save Results
results = "Base Model MAEs:\n"
for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    results += f"{name}: {mae:.2f}\n"
results += f"Meta-Learner Weighted Ensemble MAE: {ensemble_mae:.2f}\n"

# Save results to file
output_filename = f"results_meta_learner_{N}.txt"
with open(output_filename, "w") as file:
    file.write(results)

# Print results
print(f"Results saved to {output_filename}")
print(results)


# In[5]:


import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the variable for the filename
N = "CX1_Device1_5GHz"

# Load datasets dynamically based on N
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_training_{N}.csv', header=None)
test_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_test_{N}.csv', header=None)

# Prepare features and target variables
X_train = train_data.iloc[:, 6:-1]
X_test = test_data.iloc[:, 6:-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(random_state=42, n_estimators=100)

# Train and predict for each model
models = {
    "KNN": knn,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb
}

predictions = {}
mae_scores = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    predictions[name] = preds
    mae_scores[name] = mean_absolute_error(y_test, preds)

# Calculate weights (inverse of MAE)
weights = {name: 1 / mae for name, mae in mae_scores.items()}
total_weight = sum(weights.values())

# Weighted Ensemble Prediction
final_prediction = np.zeros_like(y_test, dtype=float)
for name, preds in predictions.items():
    final_prediction += preds * weights[name]

final_prediction /= total_weight

# Evaluate the Weighted Ensemble
ensemble_mae = mean_absolute_error(y_test, final_prediction)

# Prepare results for saving
results = "Model MAEs:\n"
for name, mae in mae_scores.items():
    results += f"{name}: {mae:.2f}\n"
results += f"Weighted Ensemble MAE: {ensemble_mae:.2f}\n"

# Save results to a file
output_filename = f"results_positioning_v3{N}.txt"
with open(output_filename, "w") as file:
    file.write(results)

# Print confirmation and results
print(f"Results saved to {output_filename}")
print(results)


# In[33]:


import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the variable for the filename
N = "F04_Device1_5GHz"

# Load datasets dynamically based on N
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_training_{N}.csv', header=None)
test_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_test_{N}.csv', header=None)

# Prepare features and target variables
X_train = train_data.iloc[:, 6:-1]
X_test = test_data.iloc[:, 6:-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define parameter grids for grid search
param_grids = {
    "KNN": {"n_neighbors": [3, 5, 7, 9]},
    "Decision Tree": {"max_depth": [None, 5, 10, 20], "min_samples_split": [2, 5, 10]},
    "Random Forest": {"n_estimators": [50, 100, 200], "max_depth": [None, 10, 20]},
    "XGBoost": {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1, 0.2], "max_depth": [3, 5, 7]}
}

# Initialize base models
base_models = {
    "KNN": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Random Forest": RandomForestRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42)
}

# Perform grid search for each base model
optimized_models = {}
for name, model in base_models.items():
    print(f"Optimizing {name}...")
    grid = GridSearchCV(estimator=model, param_grid=param_grids[name], scoring='neg_mean_absolute_error', cv=5)
    grid.fit(X_train_scaled, y_train)
    optimized_models[name] = grid.best_estimator_
    print(f"Best parameters for {name}: {grid.best_params_}")

# Step 1: Generate Out-of-Fold Predictions for Training Meta-Learner
kf = KFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((len(X_train), len(optimized_models)))  # Meta-features for training
test_meta_features = np.zeros((len(X_test), len(optimized_models)))  # Meta-features for test set

for i, (name, model) in enumerate(optimized_models.items()):
    test_preds_folds = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        val_preds = model.predict(X_fold_val)
        meta_features[val_idx, i] = val_preds  # Store OOF predictions
        
        # Predict on test set for this fold
        test_preds = model.predict(X_test_scaled)
        test_preds_folds.append(test_preds)
    
    # Average predictions across all folds for the test set
    test_meta_features[:, i] = np.mean(test_preds_folds, axis=0)

# Step 2: Test Alternative Meta-Learners
meta_learners = {
    "Linear Regression": LinearRegression(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
}

results = "Meta-Learner Performance:\n"
mae_results = {}
best_mae = float('inf')
best_meta_learner = None

for name, meta_learner in meta_learners.items():
    # Train meta-learner
    meta_learner.fit(meta_features, y_train)
    
    # Predict using meta-learner
    final_predictions = meta_learner.predict(test_meta_features)
    
    # Evaluate
    mae = mean_absolute_error(y_test, final_predictions)
    mae_results[name] = mae
    results += f"{name}: MAE = {mae:.2f}\n"
    
    # Track best meta-learner
    if mae < best_mae:
        best_mae = mae
        best_meta_learner = name
        best_predictions = final_predictions

# Add base model MAEs to results
mae_scores = {}
for name, model in optimized_models.items():
    preds = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, preds)
    mae_scores[name] = mae
    results += f"{name}: MAE = {mae:.2f}\n"

# Step 3: Simple Weighted Ensemble
weights = {name: 1 / mae for name, mae in mae_scores.items()}
total_weight = sum(weights.values())

weighted_ensemble_predictions = np.zeros(len(y_test))
for name, model in optimized_models.items():
    weighted_ensemble_predictions += model.predict(X_test_scaled) * weights[name]

weighted_ensemble_predictions /= total_weight
weighted_ensemble_mae = mean_absolute_error(y_test, weighted_ensemble_predictions)
results += f"Weighted Ensemble MAE: {weighted_ensemble_mae:.2f}\n"

# Save results
output_filename = f"results_meta_learners_weighted_grid_search{N}.txt"
with open(output_filename, "w") as file:
    file.write(results)

print(f"Results saved to {output_filename}")
print(results)


# In[59]:


import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the variable for the filename
N = "CX1_Device1_2GHz"

# Load datasets dynamically based on N
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_training_{N}.csv', header=None)
test_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_test_{N}.csv', header=None)

# Prepare features and target variables
X_train = train_data.iloc[:, 6:-1]
X_test = test_data.iloc[:, 6:-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(random_state=42, n_estimators=100)

# Base Models
base_models = {
    "KNN": knn,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb
}

# Generate meta-features using out-of-fold predictions
kf = KFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((len(X_train), len(base_models)))
test_meta_features = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    test_preds_folds = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        val_preds = model.predict(X_fold_val)
        meta_features[val_idx, i] = val_preds
        
        # Predict on test set for this fold
        test_preds_folds.append(model.predict(X_test_scaled))
    
    # Average predictions across all folds for the test set
    test_meta_features[:, i] = np.mean(test_preds_folds, axis=0)

# Step 1: Train and Evaluate Meta-Learners
meta_learners = {
    "Linear Regression": LinearRegression(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
}

results = "Meta-Learner Performance:\n"
meta_predictions = {}
best_mae = float('inf')
best_meta_learner = None

for name, meta_learner in meta_learners.items():
    meta_learner.fit(meta_features, y_train)
    preds = meta_learner.predict(test_meta_features)
    meta_predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    results += f"{name}: MAE = {mae:.2f}\n"
    
    # Track the best meta-learner
    if mae < best_mae:
        best_mae = mae
        best_meta_learner = name
        best_meta_predictions = preds

# Step 2: Weighted Ensemble Prediction
predictions = {}
mae_scores = {}

for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    predictions[name] = preds
    mae_scores[name] = mean_absolute_error(y_test, preds)

weights = {name: 1 / mae for name, mae in mae_scores.items()}
total_weight = sum(weights.values())

weighted_ensemble_predictions = np.zeros(len(y_test))
for name, preds in predictions.items():
    weighted_ensemble_predictions += preds * weights[name]

weighted_ensemble_predictions /= total_weight
weighted_ensemble_mae = mean_absolute_error(y_test, weighted_ensemble_predictions)

results += f"Weighted Ensemble MAE: {weighted_ensemble_mae:.2f}\n"

# Save results to file
output_filename = f"results_meta_learners_v3{N}.txt"
with open(output_filename, "w") as file:
    file.write(results)

print(f"Results saved to {output_filename}")
print(results)



# In[71]:


import matplotlib.pyplot as plt
import pandas as pd

# Group y_test and predictions by unique values of y_test
def group_predictions(y_true, predictions):
    """Group predictions by unique y_test values."""
    grouped_data = pd.DataFrame({"y_true": y_true})
    for name, preds in predictions.items():
        grouped_data[name] = preds
    grouped = grouped_data.groupby("y_true").mean().reset_index()  # Group by unique y_test and average predictions
    return grouped

# Combine all predictions
all_predictions = {**predictions, "Weighted Ensemble": weighted_ensemble_predictions, "Meta-Learner": best_meta_predictions}

# Group the predictions
grouped_results = group_predictions(y_test, all_predictions)

# Plot Real vs. Predicted Values (Discrete)
plt.figure(figsize=(14, 8))

# Plot True Values
plt.scatter(grouped_results.index, grouped_results["y_true"], label="True Values", color="black", s=100, marker="o")

# Plot Predicted Values for each algorithm
for column in grouped_results.columns[1:]:
    plt.scatter(grouped_results.index, grouped_results[column], label=column, alpha=0.7, s=80)

plt.title("Real vs. Predicted Values")
plt.xlabel("Sample Index")
plt.ylabel("Value")
plt.legend()
plt.tight_layout()
output_filename = "real_vs_predicted_plots_all_cx1.png"
plt.savefig(output_filename, dpi=300)
plt.show()


# In[70]:


import matplotlib.pyplot as plt
import pandas as pd

# Group y_test and predictions by unique values of y_test
def group_predictions(y_true, predictions):
    """Group predictions by unique y_test values."""
    grouped_data = pd.DataFrame({"y_true": y_true})
    for name, preds in predictions.items():
        grouped_data[name] = preds
    grouped = grouped_data.groupby("y_true").mean().reset_index()  # Group by unique y_test and average predictions
    return grouped

# Combine all predictions
all_predictions = {**predictions, "Weighted Ensemble": weighted_ensemble_predictions, "Meta-Learner": best_meta_predictions}

# Group the predictions
grouped_results = group_predictions(y_test, all_predictions)

# Prepare subplots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()  # Flatten the axes for easy indexing
algorithms = grouped_results.columns[1:]  # Exclude 'y_true'

# Plot for each algorithm
for i, column in enumerate(algorithms):
    ax = axes[i]
    ax.scatter(grouped_results.index, grouped_results["y_true"], label="True Values", color="black", s=100, marker="o")
    ax.scatter(grouped_results.index, grouped_results[column], label=column, alpha=0.7, s=80)
    ax.set_title(f"True vs. Predicted: {column}")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Value")
    ax.legend()

# Remove empty subplots if any
if len(algorithms) < len(axes):
    for j in range(len(algorithms), len(axes)):
        fig.delaxes(axes[j])

plt.tight_layout()
output_filename = "real_vs_predicted_plots_CX1.png"
plt.savefig(output_filename, dpi=300)
plt.show()


# In[72]:


import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the variable for the filename
N = "CX1_Device2_dual"

# Load datasets dynamically based on N
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_training_{N}.csv', header=None)
test_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_test_{N}.csv', header=None)

# Prepare features and target variables
X_train = train_data.iloc[:, 6:-1]
X_test = test_data.iloc[:, 6:-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(random_state=42, n_estimators=100)

# Base Models
base_models = {
    "KNN": knn,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb
}

# Step 1: Generate meta-features using out-of-fold predictions
kf = KFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((len(X_train), len(base_models)))
test_meta_features = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    test_preds_folds = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        val_preds = model.predict(X_fold_val)
        meta_features[val_idx, i] = val_preds
        
        # Predict on test set for this fold
        test_preds_folds.append(model.predict(X_test_scaled))
    
    # Average predictions across all folds for the test set
    test_meta_features[:, i] = np.mean(test_preds_folds, axis=0)

# Step 2: Train and Evaluate Meta-Learners
meta_learners = {
    "Linear Regression": LinearRegression(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
}

results = "Meta-Learner Performance:\n"
meta_predictions = {}
best_mae = float('inf')
best_meta_learner = None

for name, meta_learner in meta_learners.items():
    start_time = time.time()
    meta_learner.fit(meta_features, y_train)
    preds = meta_learner.predict(test_meta_features)
    meta_predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    elapsed_time = (time.time() - start_time) / len(y_test)  # Time per sample
    results += f"{name}: MAE = {mae:.2f}, Time per sample = {elapsed_time:.6f} seconds\n"
    
    # Track the best meta-learner
    if mae < best_mae:
        best_mae = mae
        best_meta_learner = name
        best_meta_predictions = preds

# Step 3: Weighted Ensemble Prediction
predictions = {}
mae_scores = {}
ensemble_start_time = time.time()

for name, model in base_models.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    elapsed_time = (time.time() - start_time) / len(y_test)  # Time per sample
    predictions[name] = preds
    mae_scores[name] = mean_absolute_error(y_test, preds)
    results += f"{name}: MAE = {mae_scores[name]:.2f}, Time per sample = {elapsed_time:.6f} seconds\n"

weights = {name: 1 / mae for name, mae in mae_scores.items()}
total_weight = sum(weights.values())

weighted_ensemble_predictions = np.zeros(len(y_test))
for name, preds in predictions.items():
    weighted_ensemble_predictions += preds * weights[name]

weighted_ensemble_predictions /= total_weight
weighted_ensemble_mae = mean_absolute_error(y_test, weighted_ensemble_predictions)
ensemble_elapsed_time = (time.time() - ensemble_start_time) / len(y_test)  # Time per sample
results += f"Weighted Ensemble MAE: {weighted_ensemble_mae:.2f}, Time per sample = {ensemble_elapsed_time:.6f} seconds\n"

# Save results to file
output_filename = f"results_meta_learners_v3{N}.txt"
with open(output_filename, "w") as file:
    file.write(results)

print(f"Results saved to {output_filename}")
print(results)


# # With Swarm
# 

# In[74]:


import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the variable for the filename
N = "CX1_Device2_dual"

# Load datasets dynamically based on N
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\SWRM_processed_training_CX1_Device2_5GHz.csv', header=None)
test_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\SWRM_processed_testing_CX1_Device2_5GHz.csv', header=None)

# Prepare features and target variables
X_train = train_data.iloc[:, 6:-1]
X_test = test_data.iloc[:, 6:-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(random_state=42, n_estimators=100)

# Base Models
base_models = {
    "KNN": knn,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb
}

# Step 1: Generate meta-features using out-of-fold predictions
kf = KFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((len(X_train), len(base_models)))
test_meta_features = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    test_preds_folds = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        val_preds = model.predict(X_fold_val)
        meta_features[val_idx, i] = val_preds
        
        # Predict on test set for this fold
        test_preds_folds.append(model.predict(X_test_scaled))
    
    # Average predictions across all folds for the test set
    test_meta_features[:, i] = np.mean(test_preds_folds, axis=0)

# Step 2: Train and Evaluate Meta-Learners
meta_learners = {
    "Linear Regression": LinearRegression(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
}

results = "Meta-Learner Performance:\n"
meta_predictions = {}
best_mae = float('inf')
best_meta_learner = None

for name, meta_learner in meta_learners.items():
    start_time = time.time()
    meta_learner.fit(meta_features, y_train)
    preds = meta_learner.predict(test_meta_features)
    meta_predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    elapsed_time = (time.time() - start_time) / len(y_test)  # Time per sample
    results += f"{name}: MAE = {mae:.2f}, Time per sample = {elapsed_time:.6f} seconds\n"
    
    # Track the best meta-learner
    if mae < best_mae:
        best_mae = mae
        best_meta_learner = name
        best_meta_predictions = preds

# Step 3: Weighted Ensemble Prediction
predictions = {}
mae_scores = {}
ensemble_start_time = time.time()

for name, model in base_models.items():
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    elapsed_time = (time.time() - start_time) / len(y_test)  # Time per sample
    predictions[name] = preds
    mae_scores[name] = mean_absolute_error(y_test, preds)
    results += f"{name}: MAE = {mae_scores[name]:.2f}, Time per sample = {elapsed_time:.6f} seconds\n"

weights = {name: 1 / mae for name, mae in mae_scores.items()}
total_weight = sum(weights.values())

weighted_ensemble_predictions = np.zeros(len(y_test))
for name, preds in predictions.items():
    weighted_ensemble_predictions += preds * weights[name]

weighted_ensemble_predictions /= total_weight
weighted_ensemble_mae = mean_absolute_error(y_test, weighted_ensemble_predictions)
ensemble_elapsed_time = (time.time() - ensemble_start_time) / len(y_test)  # Time per sample
results += f"Weighted Ensemble MAE: {weighted_ensemble_mae:.2f}, Time per sample = {ensemble_elapsed_time:.6f} seconds\n"

# Save results to file
output_filename = f"results_meta_learners_v3_swarm{N}.txt"
with open(output_filename, "w") as file:
    file.write(results)

print(f"Results saved to {output_filename}")
print(results)


# # PLOT CDF

# In[78]:


import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.distributions.empirical_distribution import ECDF

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the variable for the filename
N = "CX1_Device1_2GHz"

# Load datasets dynamically based on N
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_training_{N}.csv', header=None)
test_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_test_{N}.csv', header=None)

# Prepare features and target variables
X_train = train_data.iloc[:, 6:-1]
X_test = test_data.iloc[:, 6:-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(random_state=42, n_estimators=100)

# Base Models
base_models = {
    "KNN": knn,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb
}

# Step 1: Generate meta-features using out-of-fold predictions
kf = KFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((len(X_train), len(base_models)))
test_meta_features = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    test_preds_folds = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        val_preds = model.predict(X_fold_val)
        meta_features[val_idx, i] = val_preds
        
        # Predict on test set for this fold
        test_preds_folds.append(model.predict(X_test_scaled))
    
    # Average predictions across all folds for the test set
    test_meta_features[:, i] = np.mean(test_preds_folds, axis=0)

# Step 2: Train and Evaluate Meta-Learners
meta_learners = {
    "Linear Regression": LinearRegression(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
}

results = "Meta-Learner Performance:\n"
meta_predictions = {}
best_mae = float('inf')
best_meta_learner = None

for name, meta_learner in meta_learners.items():
    meta_learner.fit(meta_features, y_train)
    preds = meta_learner.predict(test_meta_features)
    meta_predictions[name] = preds
    mae = mean_absolute_error(y_test, preds)
    results += f"{name}: MAE = {mae:.2f}\n"
    
    # Track the best meta-learner
    if mae < best_mae:
        best_mae = mae
        best_meta_learner = name
        best_meta_predictions = preds

# Step 3: Weighted Ensemble Prediction
predictions = {}
mae_scores = {}

for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    predictions[name] = preds
    mae_scores[name] = mean_absolute_error(y_test, preds)

weights = {name: 1 / mae for name, mae in mae_scores.items()}
total_weight = sum(weights.values())

weighted_ensemble_predictions = np.zeros(len(y_test))
for name, preds in predictions.items():
    weighted_ensemble_predictions += preds * weights[name]

weighted_ensemble_predictions /= total_weight
weighted_ensemble_mae = mean_absolute_error(y_test, weighted_ensemble_predictions)

results += f"Weighted Ensemble MAE: {weighted_ensemble_mae:.2f}\n"

# Calculate Positioning Errors
positioning_errors = {
    name: np.abs(y_test - preds) for name, preds in {**predictions, "Weighted Ensemble": weighted_ensemble_predictions, "Best Meta-Learner": best_meta_predictions}.items()
}

# Plot Empirical CDF of Positioning Errors
plt.figure(figsize=(10, 6))
for name, errors in positioning_errors.items():
    ecdf = ECDF(errors)
    plt.plot(ecdf.x, ecdf.y, label=name)

plt.title("Empirical CDF of Positioning Errors {N}")
plt.xlabel("Positioning Error (meters)")
plt.ylabel("Cumulative Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"Empirical_CDF_Positioning_Errors_{N}.png")
plt.show()

# Save results to file
output_filename = f"results_meta_learners_v3{N}.txt"
with open(output_filename, "w") as file:
    file.write(results)

print(f"Results saved to {output_filename}")
print(results)


# In[77]:


import pandas as pd
import numpy as np
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from statsmodels.distributions.empirical_distribution import ECDF

# Suppress warnings
warnings.filterwarnings('ignore')

# Define the variable for the filename
N = "CX1_Device1_2GHz"

# Load datasets dynamically based on N
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_training_{N}.csv', header=None)
test_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_test_{N}.csv', header=None)

# Prepare features and target variables
X_train = train_data.iloc[:, 6:-1]
X_test = test_data.iloc[:, 6:-1]
y_train = train_data.iloc[:, -1]
y_test = test_data.iloc[:, -1]

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
knn = KNeighborsRegressor(n_neighbors=3)
dt = DecisionTreeRegressor(random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
xgb = XGBRegressor(random_state=42, n_estimators=100)

# Base Models
base_models = {
    "KNN": knn,
    "Decision Tree": dt,
    "Random Forest": rf,
    "XGBoost": xgb
}

# Step 1: Generate meta-features using out-of-fold predictions
kf = KFold(n_splits=10, shuffle=True, random_state=42)
meta_features = np.zeros((len(X_train), len(base_models)))
test_meta_features = np.zeros((len(X_test), len(base_models)))

for i, (name, model) in enumerate(base_models.items()):
    test_preds_folds = []
    for train_idx, val_idx in kf.split(X_train_scaled):
        X_fold_train, X_fold_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        model.fit(X_fold_train, y_fold_train)
        val_preds = model.predict(X_fold_val)
        meta_features[val_idx, i] = val_preds
        
        # Predict on test set for this fold
        test_preds_folds.append(model.predict(X_test_scaled))
    
    # Average predictions across all folds for the test set
    test_meta_features[:, i] = np.mean(test_preds_folds, axis=0)

# Step 2: Train and Evaluate Meta-Learners
meta_learners = {
    "Linear Regression": LinearRegression(),
    "Neural Network": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
}

meta_predictions = {}
for name, meta_learner in meta_learners.items():
    meta_learner.fit(meta_features, y_train)
    preds = meta_learner.predict(test_meta_features)
    meta_predictions[name] = preds

# Step 3: Weighted Ensemble Prediction
predictions = {}
for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    predictions[name] = model.predict(X_test_scaled)

weights = {name: 1 / mean_absolute_error(y_test, preds) for name, preds in predictions.items()}
total_weight = sum(weights.values())

weighted_ensemble_predictions = np.zeros(len(y_test))
for name, preds in predictions.items():
    weighted_ensemble_predictions += preds * weights[name]

weighted_ensemble_predictions /= total_weight
predictions["Weighted Ensemble"] = weighted_ensemble_predictions

# Positioning Errors
positioning_errors = {
    name: np.abs(y_test - preds) for name, preds in predictions.items()
}
positioning_errors["Meta-Learner"] = np.abs(y_test - meta_predictions["Neural Network"])

# Subplots for ECDF
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()

comparison_pairs = [
    ("KNN", "Meta-Learner"),
    ("Decision Tree", "Meta-Learner"),
    ("Random Forest", "Meta-Learner"),
    ("Weighted Ensemble", "Meta-Learner")
]

for i, (alg1, alg2) in enumerate(comparison_pairs):
    ax = axes[i]
    ecdf1 = ECDF(positioning_errors[alg1])
    ecdf2 = ECDF(positioning_errors[alg2])
    
    ax.plot(ecdf1.x, ecdf1.y, label=f"{alg1} ECDF")
    ax.plot(ecdf2.x, ecdf2.y, label=f"{alg2} ECDF")
    
    ax.set_title(f"ECDF: {alg1} vs. {alg2}")
    ax.set_xlabel("Positioning Error (meters)")
    ax.set_ylabel("Cumulative Probability")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.savefig(f"Empirical_CDF_Positioning_Errors_Subplots_{N}.png")
plt.show()


# # POWER MAP

# In[82]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load training data dynamically
N = "CX1_Device1_2GHz"
train_data = pd.read_csv(f'C:\\Users\\admin\\M_final\\UTMInDualSymFi\\All_pos_data_training_{N}.csv', header=None)

# Extract positions and RSSI values
positions = train_data.iloc[:, -1]
rssi_values = train_data.iloc[:, 6:-1]

# Convert RSSI from percentage to dBm using the given formula
rssi_dBm = -100 + (rssi_values * (50 / 100))

# Identify the 4 APs with the highest average RSSI across all positions
top_aps_indices = rssi_dBm.mean(axis=0).nlargest(4).index

# Filter the RSSI values for the top 4 APs
top_aps_rssi = rssi_dBm.iloc[:, top_aps_indices]

# Compute position-wise mean RSSI for the top 4 APs
position_mean_rssi = top_aps_rssi.groupby(positions).mean()

# Plot position-wise mean RSSI values for the top 4 APs
plt.figure(figsize=(12, 8))
for ap in position_mean_rssi.columns:
    plt.plot(position_mean_rssi.index, position_mean_rssi[ap], label=f"AP {ap}", marker='o')

plt.title("Position-wise Mean RSSI Values of Top 4 APs (2.4 GHz)")
plt.xlabel("Position")
plt.ylabel("RSSI (dBm)")
plt.legend(title="Access Points")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"Position_wise_Mean_RSSI_{N}.png")
plt.show()



# In[ ]:




