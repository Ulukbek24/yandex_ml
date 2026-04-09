FROM python:3.11-slim

WORKDIR /app

# Зависимости системы
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код проекта
COPY . .

# Папки для данных
RUN mkdir -p /app/index /app/logs

# Переменные среды по умолчанию
ENV KNOWLEDGE_BASE_DIR=/app/knowledge_base
ENV INDEX_DIR=/app/index
ENV LLM_MODEL=gpt-4o-mini
ENV EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV TOP_K=5

# Порт для FastAPI (если понадобится API-режим)
EXPOSE 8080

# По умолчанию — CLI-бот
CMD ["python", "rag_bot.py"]
