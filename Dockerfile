# Use a lightweight Python image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app and models folder
COPY app/ app/
COPY models/ models/
COPY datafiles/ datafiles/
COPY visualizations/ visualizations/

# Expose the port for FastAPI
EXPOSE 8000

# Run the FastAPI server with Uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


