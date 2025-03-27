# Gebruik een lichte Python-image als basis
FROM python:3.9-slim

# Stel de werkdirectory in
WORKDIR /app

# Kopieer requirements.txt en installeer de benodigde pakketten
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Kopieer de hele app-directory
COPY app/ app/

# Zorg ervoor dat de MLflow-tracking directory bestaat
RUN mkdir -p /app/mlruns

# Exposeer poort 8000 voor de API
EXPOSE 8000

# Start de FastAPI-server via Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
