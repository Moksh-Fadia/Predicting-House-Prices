import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
# GridSearchCV: Tests all possible combinations of hyperparameters
# RandomizedSearchCV: Randomly samples combinations from a range of hyperparameters. Faster for large datasets.
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("housing.csv")

print(data.isnull().sum())      # Checking for missing values


data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())     # Handling missing values

# Handling categorical data (ocean_proximity)
data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True)   # Machine learning models can only process numerical data. So, we convert text categories to numbers.
# pd.get_dummies(): This function converts categorical text data into numerical data by creating a new column for each category with a 0 or 1


# Feature Engineering
data['rooms_per_household'] = data['total_rooms'] / data['households']
data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
data['population_per_household'] = data['population'] / data['households']


X = data.drop(columns=["median_house_value"])   # drops this column and takes rest of the other columns as input for prediction
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Feature Scaling (StandardScaler for Linear Regression)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Dictionary to store evaluation results from the models. basically the models' performances
results = {}

# Function to evaluate models
def evaluate_model(model, model_name):
    model.fit(X_train, y_train)    # train the model to learn the relationship between the features (X_train) and the target variable (y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100  # Convert R² Score to a percentage

    results[model_name] = [mse, r2, accuracy]     # key = model_name, value = ([mse, r2, accuracy]) in the dict named results
    
    # Displaying a few actual vs predicted values
    comparison = pd.DataFrame({"Actual": y_test[:5], "Predicted": y_pred[:5]})
    print(f"\n{model_name} - First 5 Actual vs. Predicted Values:")
    print(comparison)


# List of models to test
models = {      # another dict which stores all the ML models we want to test ( "key": value() )
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    "Support Vector Regressor (SVR)": SVR()
}


# Hyperparameter grids for each model
param_grids = {
    "Linear Regression": {},    # leave the dictionary empty because it doesn't really have hyperparameters that require tuning
    "Decision Tree Regressor": {
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "Random Forest Regressor": {
        "n_estimators": [50, 100],    # controls how many trees are built
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    },
    "Gradient Boosting Regressor": {
        "learning_rate": [0.01, 0.1],      # controls how much the model adjusts during learning
        "n_estimators": [50, 100],
        "max_depth": [3, 5]
    },
    "K-Nearest Neighbors Regressor": {
        "n_neighbors": [3, 5],
        "weights": ['uniform', 'distance'],
        "p": [1, 2]  
    },
    "Support Vector Regressor (SVR)": {
        "C": [0.1, 1],      # C and epsilon control how strict the model is in fitting the data
        "epsilon": [0.1, 0.2],
        "kernel": ['linear', 'rbf']
    }
}

# Dictionary to store best models
best_models = {}

# Looping Through Each Model to Tune Hyperparameters
for model_name, model in models.items():
    print(f"\nTuning hyperparameters for {model_name}...")

    if param_grids[model_name]:  # Only tune if there are hyperparameters specified
        search = RandomizedSearchCV(
            estimator=model,    # The model we're tuning (e.g., RandomForestRegressor())
            param_distributions=param_grids[model_name],    # the hyperparameters we want to test from the dict
            n_iter=5,    # Number of random combinations to test
            scoring='r2',    # Using R² score to evaluate the model's performance
            cv=2,    # 2-fold Cross-Validation ie. It splits the data into 3 parts and tests each one separately.
            n_jobs=-1,    # Use all available CPU cores for faster processing
            random_state=42,      # Ensure reproducibility of results ie. consistent results each time you run it
            verbose=1       # Display progress message in the console
        )
        search.fit(X_train, y_train)    # train the model with the training data (X_train, y_train) using all the hyperparameter combinations
        best_models[model_name] = search.best_estimator_    # .best_estimator_: The model with the best hyperparameters.
        print(f"Best parameters for {model_name}: {search.best_params_}")   # .best_params_: The exact hyperparameter values that performed the best.
        print(f"Best R² Score: {search.best_score_}\n")    # .best_score_: The highest R² score achieved with those hyperparameters.
    else:       # If a model doesn't have hyperparameters to tune (like Linear Regression), we just train it normally and store it in best_models
        model.fit(X_train, y_train)
        best_models[model_name] = model  # For models without hyperparameters


# Evaluate each model
for model_name, model in models.items():    # looping over all the models defined in the dict
    evaluate_model(model, model_name)

# Display results
final_results = pd.DataFrame(results, index=["MSE", "R² Score", "Accuracy (%)"]).T.sort_values(by="R² Score", ascending=False)
print("\nComparison of Models:\n")
print(final_results)


# Visualization:
# Convert results to DataFrame for easy plotting
results_df = pd.DataFrame(results, index=["MSE", "R² Score", "Accuracy (%)"]).T

# Plotting R² Score Comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df["R² Score"], palette="viridis")
plt.title("R² Score Comparison of Models")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.show()

# Plotting Mean Squared Error (MSE) Comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df["MSE"], palette="plasma")
plt.title("Mean Squared Error Comparison of Models")
plt.ylabel("MSE")
plt.xticks(rotation=45)
plt.show()

# Plotting Accuracy (%) Comparison
plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df["Accuracy (%)"], palette="cividis")
plt.title("Accuracy Comparison of Models")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.show()




