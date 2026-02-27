FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8050 \
    PREFLIGHT_DATA_DIR=/tmp/preflight-data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN addgroup --system app && adduser --system --ingroup app --home /home/app app \
    && mkdir -p /tmp/preflight-data \
    && chown -R app:app /app /tmp/preflight-data

EXPOSE 8050
USER app

# If wsgi.py defines server = <flask_app> or server = <dash_app.server>
CMD ["sh", "-c", "gunicorn wsgi:server --bind 0.0.0.0:${PORT:-8050} --workers 2 --threads 4 --timeout 180 --access-logfile - --error-logfile -"]
