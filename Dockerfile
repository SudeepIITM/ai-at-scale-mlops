# Use slim Python base
FROM python:3.10-slim

WORKDIR /app

# Copy project files
COPY . .

# Install deps (minimal for API + MLflow client)
RUN pip install --no-cache-dir fastapi uvicorn mlflow pandas pydantic requests scikit-learn

# Set MLflow tracking URI (points to server at host machine)
ENV MLFLOW_TRACKING_URI=http://host.docker.internal:5000

EXPOSE 8000

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
