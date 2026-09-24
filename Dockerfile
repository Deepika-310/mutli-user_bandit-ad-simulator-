FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Store the SQLite file on a volume so data survives container restarts
ENV BANDIT_DB=/data/bandit.db
RUN mkdir -p /data && useradd --create-home appuser && chown -R appuser /data /app
USER appuser

EXPOSE 8000 8501
# Default: the REST API. docker-compose overrides the command for the dashboard.
CMD ["uvicorn", "--factory", "service.main:create_app", "--host", "0.0.0.0", "--port", "8000"]
