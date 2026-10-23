FROM python:3.12-slim
RUN useradd --create-home appuser
WORKDIR /app
COPY requirements.txt pyproject.toml ./
COPY src ./src
COPY api ./api
RUN pip install --no-cache-dir -r requirements.txt
USER appuser
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
