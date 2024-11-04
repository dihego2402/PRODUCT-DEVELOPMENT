import pandas as pd
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib

# Cargar parámetros
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Cargar datos de entrenamiento
train_data = pd.read_csv("data/train_data.csv")

# Separar variables independientes y dependientes
X_train = train_data.drop(columns=[params['target']])
y_train = train_data[params['target']]

# Inicializar modelos
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=params['random_forest']['n_estimators']),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=params['gradient_boosting']['n_estimators'])
}

# Entrenar y seleccionar el mejor modelo
best_model = None
best_mse = float('inf')
for model_name, model in models.items():
    model.fit(X_train, y_train)
    predictions = model.predict(X_train)
    mse = mean_squared_error(y_train, predictions)
    if mse < best_mse:
        best_mse = mse
        best_model = model
        best_model_name = model_name

# Guardar el mejor modelo
joblib.dump(best_model, "models/best_model.pkl")
print(f"Mejor modelo: {best_model_name} con MSE: {best_mse}")