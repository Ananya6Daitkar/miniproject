FROM python:3.11-slim
WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir -r requirements-lock.txt
COPY railml railml
COPY web web
COPY artifacts artifacts
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["python", "-m", "uvicorn", "railml.server:app", "--host", "0.0.0.0", "--port", "8000"]
