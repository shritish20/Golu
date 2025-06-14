#!/bin/bash

# Start Uvicorn, listening on all interfaces, port 8000
# 'main:app' assumes your FastAPI app instance is named 'app' in 'main.py'
# If your app is in a different file (e.g., `your_api_file.py`), change it to `your_api_file:app`
exec uvicorn main:app --host 0.0.0.0 --port 8000
