FROM python:3.10-slim

WORKDIR /app

ADD requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

ADD api.py .
ADD ai_id_vehicle.keras .
RUN mkdir -p ai_id_vehicle/1/
COPY ai_id_vehicle.keras ./ai_id_vehicle/1/

EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]

