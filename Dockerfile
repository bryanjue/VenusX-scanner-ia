# Use the official Python image
FROM python:3.11-slim

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire source code
COPY . .

EXPOSE 5000

# Start the application pointing to app/main.py
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5000"]