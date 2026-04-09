# RAG-бот: Velthari Chronicles

RAG-чатбот для вселенной "Velthari Chronicles". Отвечает на вопросы по базе знаний, отклоняет вопросы вне базы и блокирует попытки prompt injection.

## Быстрый старт

```bash
# 1. Установить зависимости
pip install -r requirements.txt

# 2. Создать .env из примера
cp .env.example .env
# Вписать OPENAI_API_KEY в .env

# 3. Построить индекс
python build_index.py

# 4. Запустить бота
python rag_bot.py          # интерактивный режим
python rag_bot.py --demo   # демо-режим (10 вопросов)
```

## Через Docker

```bash
# Собрать образ
docker compose build

# Построить индекс
docker compose --profile build run index-builder

# Запустить бота
docker compose run rag-bot

# Запустить автообновление индекса (ежедневно в 06:00)
docker compose --profile updater up index-updater
```

## Структура проекта

```
├── knowledge_base/          # База знаний (41 .md файл)
├── index/                   # FAISS индекс (создаётся при build_index.py)
├── logs/                    # Логи запросов и обновлений
├── scripts/
│   └── replace_terms.py     # Скрипт замены терминов
├── build_index.py           # Задание 2: построение индекса
├── rag_bot.py               # Задания 3–5: RAG-бот + безопасность
├── update_index.py          # Задание 6: автообновление индекса
├── evaluate.py              # Задание 7: оценка качества
├── golden_questions.txt     # Тестовый набор вопросов
├── Project_template.md      # Подробное описание решений
├── terms_map.json           # Маппинг терминов (43 пары)
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

## Оценка качества

```bash
python evaluate.py
```

Выводит отчёт: % правильных ответов, заблокированных атак, пробелов в базе знаний.

## Обновление индекса вручную

```bash
python update_index.py
```

Добавляет только новые/изменённые файлы, не перестраивает индекс с нуля.
