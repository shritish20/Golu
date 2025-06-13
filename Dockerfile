FROM python:3.9-slim-buster
 WORKDIR /app
 COPY requirements.txt .
 RUN pip install--no-cache-dir-r requirements.txt
 COPY . .
 CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port",
 "8000"]
 To build and run the Docker container:
 # Build the image
 docker build-t voluguard-api .
 # Run the container
 docker run-d-p 8000:8000--env-file .env voluguard-api
