"""
Задание 6: Автоматическое ежедневное обновление базы знаний.

Что делает:
    - Отслеживает новые/изменённые файлы в knowledge_base/
    - Добавляет их в существующий FAISS индекс (не перестраивает с нуля)
    - Логирует процесс обновления
    - Может запускаться по cron (см. README)

Запуск вручную:    python update_index.py
Запуск по расписанию: python update_index.py --schedule (каждый день в 06:00)
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
INDEX_DIR = os.getenv("INDEX_DIR", "./index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
STATE_FILE = Path(INDEX_DIR) / "file_state.json"
LOG_DIR = "./logs"

# Настройка логгера
Path(LOG_DIR).mkdir(exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(f"{LOG_DIR}/update_index.log", encoding="utf-8"),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def load_file_state() -> dict:
    """Загружаем состояние файлов (mtime) для отслеживания изменений."""
    if STATE_FILE.exists():
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_file_state(state: dict):
    """Сохраняем текущее состояние файлов."""
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def get_new_or_changed_files(kb_dir: str, state: dict) -> list[Path]:
    """Находим файлы, которые появились или изменились с прошлого запуска."""
    new_files = []
    for ext in ["*.md", "*.txt"]:
        for fp in Path(kb_dir).glob(ext):
            mtime = fp.stat().st_mtime
            key = str(fp)
            if key not in state or state[key] != mtime:
                new_files.append(fp)
    return new_files


def update_index():
    """Основная функция обновления индекса."""
    start_time = time.time()
    log.info("=" * 50)
    log.info("Запуск обновления индекса")

    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

    # Проверяем, существует ли индекс
    if not Path(INDEX_DIR).exists():
        log.warning("Индекс не найден. Запустите build_index.py для первоначального создания.")
        return

    # Загружаем состояние файлов
    state = load_file_state()
    new_files = get_new_or_changed_files(KNOWLEDGE_BASE_DIR, state)

    if not new_files:
        log.info("Новых или изменённых файлов нет. Индекс актуален.")
        elapsed = time.time() - start_time
        _write_update_log(0, elapsed, [])
        return

    log.info(f"Найдено новых/изменённых файлов: {len(new_files)}")
    for fp in new_files:
        log.info(f"  → {fp.name}")

    # Загружаем эмбеддинги и существующий индекс
    log.info(f"Загружаю модель: {EMBEDDING_MODEL}")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.load_local(
        INDEX_DIR, embeddings, allow_dangerous_deserialization=True
    )

    # Разбиваем новые файлы на чанки
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", " ", ""],
    )

    new_docs = []
    for fp in new_files:
        with open(fp, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            continue

        chunks = splitter.split_text(content)
        for i, chunk in enumerate(chunks):
            new_docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": str(fp),
                    "filename": fp.name,
                    "title": fp.stem.replace("_", " ").title(),
                    "chunk_index": i,
                    "updated_at": datetime.now().isoformat(),
                }
            ))

    if new_docs:
        log.info(f"Добавляю {len(new_docs)} новых чанков в индекс...")
        vectorstore.add_documents(new_docs)
        vectorstore.save_local(INDEX_DIR)
        log.info("Индекс обновлён и сохранён.")

    # Обновляем состояние файлов
    for fp in new_files:
        state[str(fp)] = fp.stat().st_mtime
    save_file_state(state)

    elapsed = time.time() - start_time
    _write_update_log(len(new_docs), elapsed, [fp.name for fp in new_files])
    log.info(f"Готово. Время: {elapsed:.1f}с, добавлено чанков: {len(new_docs)}")


def _write_update_log(new_chunks: int, elapsed: float, files: list):
    """Пишем лог обновления в JSON для аналитики."""
    Path(LOG_DIR).mkdir(exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "event": "index_update",
        "new_chunks": new_chunks,
        "files_processed": files,
        "elapsed_seconds": round(elapsed, 2),
    }
    log_path = Path(LOG_DIR) / "update_history.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_scheduled():
    """Запуск по расписанию через schedule."""
    import schedule
    log.info("Запуск в режиме расписания. Обновление каждый день в 06:00.")
    schedule.every().day.at("06:00").do(update_index)

    # Выполняем сразу при старте
    update_index()

    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", action="store_true",
                        help="Запустить в режиме расписания (06:00 ежедневно)")
    args = parser.parse_args()

    if args.schedule:
        run_scheduled()
    else:
        update_index()
