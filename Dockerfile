FROM python:3.12-slim

WORKDIR /srv

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY suspension_model.py .
COPY ["Cubi Point Daily Data.csv", "./"]
COPY ["May-November Suspensions - Sheet1.csv", "./"]
COPY app/ ./app/

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
