import pandas as pd
import yaml
import joblib
from sklearn.metrics import mean_squared_error

# Cargar parámetros
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Cargar datos de prueba y modelo entrenado
test_data = pd.read_csv("data/test_data.csv")
model = joblib.load("models/best_model.pkl")

# Separar variables independientes y dependientes
X_test = test_data.drop(columns=[params['target']])
y_test = test_data[params['target']]

# Predicciones y evaluación
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)

# Guardar métricas en JSON
metrics = {"MSE": mse}
with open("metrics.json", "w") as file:
    json.dump(metrics, file)

print(f"MSE en conjunto de prueba: {mse}")