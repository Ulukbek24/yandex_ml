"""
Задание 7: Аналитика покрытия и качества базы знаний.

Что делает:
    1. Загружает "золотые вопросы" из golden_questions.txt
    2. Задаёт каждый вопрос боту
    3. Оценивает ответ: правильный / неверный / заблокирован
    4. Сохраняет результаты в logs/evaluation.jsonl
    5. Выводит итоговый отчёт

Запуск:
    python evaluate.py
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

INDEX_DIR = os.getenv("INDEX_DIR", "./index")
LOG_DIR = "./logs"
GOLDEN_FILE = "./golden_questions.txt"
EVAL_LOG = f"{LOG_DIR}/evaluation.jsonl"


def load_golden_questions(path: str) -> list[dict]:
    """Парсим файл с золотыми вопросами."""
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("|")
            if len(parts) >= 2:
                questions.append({
                    "question": parts[0].strip(),
                    "expected": parts[1].strip(),  # answer / no_answer / blocked / partial
                    "keywords": parts[2].strip().split(", ") if len(parts) > 2 else [],
                })
    return questions


def evaluate_answer(answer: str, expected: str, keywords: list[str]) -> dict:
    """Оцениваем ответ бота."""
    answer_lower = answer.lower()

    if expected == "blocked":
        correct = "заблоки" in answer_lower or "манипуляц" in answer_lower or "не могу" in answer_lower
        return {"correct": correct, "reason": "security_filter" if correct else "filter_missed"}

    if expected == "no_answer":
        correct = "не знаю" in answer_lower or "нет информации" in answer_lower or "отсутствует" in answer_lower
        return {"correct": correct, "reason": "no_answer_detected" if correct else "hallucination_risk"}

    if expected in ("answer", "partial"):
        if "не знаю" in answer_lower:
            return {"correct": False, "reason": "unexpected_no_answer"}
        # Проверяем наличие ключевых слов
        found_keywords = [kw for kw in keywords if kw.lower() in answer_lower]
        coverage = len(found_keywords) / len(keywords) if keywords else 1.0
        correct = coverage >= 0.5  # хотя бы половина ключевых слов
        return {
            "correct": correct,
            "keyword_coverage": round(coverage, 2),
            "found": found_keywords,
            "missed": [kw for kw in keywords if kw.lower() not in answer_lower],
            "reason": "keywords_found" if correct else "low_keyword_coverage",
        }

    return {"correct": False, "reason": "unknown_expected_type"}


def run_evaluation():
    print("=" * 60)
    print("Автоматическое тестирование качества RAG-бота")
    print("=" * 60)

    # Импортируем бот (без --demo флага)
    from rag_bot import RAGBot
    bot = RAGBot()

    questions = load_golden_questions(GOLDEN_FILE)
    print(f"\nЗагружено вопросов: {len(questions)}\n")

    results = []
    correct_count = 0
    blocked_count = 0

    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['question'][:60]}...")

        start = time.time()
        answer = bot.ask(q["question"])
        elapsed = time.time() - start

        eval_result = evaluate_answer(answer, q["expected"], q["keywords"])
        correct = eval_result.get("correct", False)

        status = "✅" if correct else "❌"
        if "заблоки" in answer.lower():
            status = "🛡️"
            blocked_count += 1

        print(f"   {status} Ожидалось: {q['expected']} | {eval_result.get('reason', '')}")

        if correct:
            correct_count += 1

        entry = {
            "timestamp": datetime.now().isoformat(),
            "question": q["question"],
            "expected": q["expected"],
            "answer": answer[:500],
            "correct": correct,
            "elapsed": round(elapsed, 2),
            "eval_details": eval_result,
        }
        results.append(entry)
        time.sleep(0.5)  # не спамим API

    # Сохраняем результаты
    Path(LOG_DIR).mkdir(exist_ok=True)
    with open(EVAL_LOG, "a", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Итоговый отчёт
    total = len(questions)
    accuracy = correct_count / total * 100 if total else 0

    print("\n" + "=" * 60)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    print(f"Всего вопросов:    {total}")
    print(f"Правильных:        {correct_count} ({accuracy:.0f}%)")
    print(f"Заблокировано:     {blocked_count}")
    print(f"Неправильных:      {total - correct_count - blocked_count}")

    # Анализ пробелов
    gaps = [r for r in results if not r["correct"] and r["expected"] in ("answer", "partial")]
    if gaps:
        print(f"\n⚠️  Пробелы в базе знаний ({len(gaps)} вопросов без ответа):")
        for g in gaps:
            print(f"   • {g['question'][:70]}")

    print(f"\n📁 Лог сохранён: {EVAL_LOG}")
    return results


if __name__ == "__main__":
    run_evaluation()
