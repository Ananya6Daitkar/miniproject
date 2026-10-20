FROM python:3.11-slim
WORKDIR /app
COPY requirements-lock.txt .
RUN pip install --no-cache-dir torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements-lock.txt
COPY railml railml
COPY web web
COPY artifacts artifacts
RUN useradd --create-home appuser
USER appuser
EXPOSE 8000
CMD ["sh", "-c", "exec python -m uvicorn railml.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
