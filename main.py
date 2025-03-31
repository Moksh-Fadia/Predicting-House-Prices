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
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("housing.csv")

print(data.isnull().sum())   


data['total_bedrooms'] = data['total_bedrooms'].fillna(data['total_bedrooms'].median())     

data = pd.get_dummies(data, columns=["ocean_proximity"], drop_first=True) 

data['rooms_per_household'] = data['total_rooms'] / data['households']
data['bedrooms_per_room'] = data['total_bedrooms'] / data['total_rooms']
data['population_per_household'] = data['population'] / data['households']


X = data.drop(columns=["median_house_value"])   
y = data["median_house_value"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

results = {}

def evaluate_model(model, model_name):
    model.fit(X_train, y_train)  
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    accuracy = r2 * 100  

    results[model_name] = [mse, r2, accuracy]    

    comparison = pd.DataFrame({"Actual": y_test[:5], "Predicted": y_pred[:5]})
    print(f"\n{model_name} - First 5 Actual vs. Predicted Values:")
    print(comparison)

models = {     
    "Linear Regression": LinearRegression(),
    "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
    "Random Forest Regressor": RandomForestRegressor(random_state=42),
    "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=42),
    "K-Nearest Neighbors Regressor": KNeighborsRegressor(),
    "Support Vector Regressor (SVR)": SVR()
}

param_grids = {
    "Linear Regression": {},   
    "Decision Tree Regressor": {
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2]
    },
    "Random Forest Regressor": {
        "n_estimators": [50, 100],    
        "max_depth": [None, 10],
        "min_samples_split": [2, 5]
    },
    "Gradient Boosting Regressor": {
        "learning_rate": [0.01, 0.1],      
        "n_estimators": [50, 100],
        "max_depth": [3, 5]
    },
    "K-Nearest Neighbors Regressor": {
        "n_neighbors": [3, 5],
        "weights": ['uniform', 'distance'],
        "p": [1, 2]  
    },
    "Support Vector Regressor (SVR)": {
        "C": [0.1, 1],    
        "epsilon": [0.1, 0.2],
        "kernel": ['linear', 'rbf']
    }
}

best_models = {}

for model_name, model in models.items():
    print(f"\nTuning hyperparameters for {model_name}...")

    if param_grids[model_name]:  
        search = RandomizedSearchCV(
            estimator=model,    
            param_distributions=param_grids[model_name],  
            n_iter=5,   
            scoring='r2',    
            cv=2,    
            n_jobs=-1,   
            random_state=42,    
            verbose=1   
        )
        search.fit(X_train, y_train)   
        best_models[model_name] = search.best_estimator_    
        print(f"Best parameters for {model_name}: {search.best_params_}")  
        print(f"Best R² Score: {search.best_score_}\n")    
    else:      
        model.fit(X_train, y_train)
        best_models[model_name] = model 

for model_name, model in models.items():
    evaluate_model(model, model_name)

final_results = pd.DataFrame(results, index=["MSE", "R² Score", "Accuracy (%)"]).T.sort_values(by="R² Score", ascending=False)
print("\nComparison of Models:\n")
print(final_results)


# Visualization:

results_df = pd.DataFrame(results, index=["MSE", "R² Score", "Accuracy (%)"]).T

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df["R² Score"], palette="viridis")
plt.title("R² Score Comparison of Models")
plt.ylabel("R² Score")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df["MSE"], palette="plasma")
plt.title("Mean Squared Error Comparison of Models")
plt.ylabel("MSE")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(12, 6))
sns.barplot(x=results_df.index, y=results_df["Accuracy (%)"], palette="cividis")
plt.title("Accuracy Comparison of Models")
plt.ylabel("Accuracy (%)")
plt.xticks(rotation=45)
plt.show()




