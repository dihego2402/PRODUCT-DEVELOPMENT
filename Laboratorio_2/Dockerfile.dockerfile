FROM python:3.9-slim

# Instalar dependencias
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copiar archivos
COPY . /app
WORKDIR /app

# Exponer el puerto para la API
EXPOSE 8000

# Comando por defecto
CMD ["python", "app.py"]