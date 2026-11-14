FROM python:3.11-slim
WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
COPY src ./src
COPY reports/baseline ./reports/baseline
ENV PYTHONPATH=/app/src
EXPOSE 8000
# To serve a fine-tuned model: add torch+transformers, mount its folder and set GRIEVANCE_MODEL_DIR.
CMD ["uvicorn", "grievance.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
