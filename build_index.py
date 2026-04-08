"""
Задание 3: Создание векторного индекса из базы знаний.

Запуск:
    python build_index.py

Что делает:
    1. Читает все .md и .txt файлы из папки knowledge_base/
    2. Разбивает на чанки по 500 токенов с перекрытием 50
    3. Генерирует эмбеддинги через sentence-transformers (локально, бесплатно)
    4. Сохраняет FAISS индекс в папку index/
"""

import os
import time
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "./knowledge_base")
INDEX_DIR = os.getenv("INDEX_DIR", "./index")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 500      # токенов на чанк
CHUNK_OVERLAP = 50    # перекрытие для сохранения контекста


def load_documents(kb_dir: str) -> list[dict]:
    """Читает все документы из базы знаний."""
    docs = []
    kb_path = Path(kb_dir)

    for ext in ["*.md", "*.txt"]:
        for file_path in sorted(kb_path.glob(ext)):
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if not content:
                continue

            docs.append({
                "source": str(file_path),
                "filename": file_path.name,
                "title": file_path.stem.replace("_", " ").title(),
                "content": content,
            })

    return docs


def build_index():
    print("=" * 60)
    print("Построение векторного индекса базы знаний")
    print("=" * 60)

    # Ленивый импорт — чтобы не падать при проверке скрипта без зависимостей
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.schema import Document

    start_time = time.time()

    # 1. Загрузка документов
    print(f"\n[1/4] Загрузка документов из {KNOWLEDGE_BASE_DIR}...")
    raw_docs = load_documents(KNOWLEDGE_BASE_DIR)
    print(f"      Найдено документов: {len(raw_docs)}")

    # 2. Разбиение на чанки
    print(f"\n[2/4] Разбиение на чанки (размер={CHUNK_SIZE}, перекрытие={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n\n", "\n", " ", ""],
    )

    langchain_docs = []
    for doc in raw_docs:
        chunks = splitter.split_text(doc["content"])
        for i, chunk in enumerate(chunks):
            langchain_docs.append(Document(
                page_content=chunk,
                metadata={
                    "source": doc["source"],
                    "filename": doc["filename"],
                    "title": doc["title"],
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                }
            ))

    print(f"      Всего чанков: {len(langchain_docs)}")

    # 3. Генерация эмбеддингов
    print(f"\n[3/4] Загрузка модели эмбеддингов: {EMBEDDING_MODEL}...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    print("      Модель загружена. Генерирую эмбеддинги...")

    # 4. Создание и сохранение индекса
    print(f"\n[4/4] Создание FAISS индекса и сохранение в {INDEX_DIR}...")
    vectorstore = FAISS.from_documents(langchain_docs, embeddings)

    Path(INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vectorstore.save_local(INDEX_DIR)

    # Сохраняем метаинфо
    elapsed = time.time() - start_time
    meta = {
        "documents": len(raw_docs),
        "chunks": len(langchain_docs),
        "embedding_model": EMBEDDING_MODEL,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "build_time_seconds": round(elapsed, 2),
    }
    with open(Path(INDEX_DIR) / "meta.json", "w") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Готово!")
    print(f"   Документов: {len(raw_docs)}")
    print(f"   Чанков: {len(langchain_docs)}")
    print(f"   Время: {elapsed:.1f} сек")
    print(f"   Индекс сохранён: {INDEX_DIR}/")

    # Быстрый тест
    print("\n[Тест] Проверяю поиск по индексу...")
    results = vectorstore.similarity_search("Krath Velgor", k=3)
    for i, r in enumerate(results, 1):
        preview = r.page_content[:80].replace("\n", " ")
        print(f"   {i}. [{r.metadata['title']}] {preview}...")

    return vectorstore


if __name__ == "__main__":
    build_index()
