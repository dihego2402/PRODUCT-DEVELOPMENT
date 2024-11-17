import os
import pandas as pd
from preprocessing import preprocess_data

def batch_predict(input_folder, output_folder, model, target_column):
    for file in os.listdir(input_folder):
        if file.endswith('.parquet'):
            input_path = os.path.join(input_folder, file)
            data = pd.read_parquet(input_path)
            X, _ = preprocess_data(data, target_column)
            predictions = model.predict_proba(X)
            
            # Guardar predicciones
            output_path = os.path.join(output_folder, f"predictions_{file}")
            pd.DataFrame(predictions).to_parquet(output_path)