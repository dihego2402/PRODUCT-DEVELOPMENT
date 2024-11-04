import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import yaml

# Cargar parámetros
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Cargar datos
data = pd.read_csv("data/data.csv")

# Preprocesamiento
numeric_features = params['preprocessing']['numeric_features']
categorical_features = params['preprocessing']['categorical_features']

# Normalización para numéricas
scaler = StandardScaler()
data[numeric_features] = scaler.fit_transform(data[numeric_features])

# Codificación OneHot para categóricas
encoder = OneHotEncoder(sparse=False)
encoded_features = encoder.fit_transform(data[categorical_features])
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
data = data.drop(columns=categorical_features)
data = pd.concat([data, encoded_df], axis=1)

# Separación de datos en entrenamiento y prueba
train_data, test_data = train_test_split(data, test_size=params['split']['test_size'], random_state=params['split']['random_state'])
train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)
