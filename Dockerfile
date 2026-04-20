FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SYNAPSEMEM_DATA_DIR=/app/data

WORKDIR /app

COPY backend/pyproject.toml ./backend/pyproject.toml
RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    "sqlalchemy>=2.0" \
    "aiosqlite>=0.20" \
    "pydantic>=2.9" \
    "pydantic-settings>=2.5" \
    "httpx>=0.27" \
    "networkx>=3.3" \
    "python-multipart>=0.0.9" \
    "pytest>=8.3" \
    "pytest-asyncio>=0.24"

COPY backend /app/backend
COPY frontend /app/frontend

RUN mkdir -p /app/data

EXPOSE 5555
WORKDIR /app/backend
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "5555"]
