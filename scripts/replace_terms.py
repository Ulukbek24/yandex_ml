"""
Скрипт замены терминов Star Wars → вселенная Velthari Chronicles.
Запускать однократно при создании базы знаний из оригинальных источников.
"""
import json
import os
import re
from pathlib import Path


def load_terms_map(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["terms"]


def replace_in_text(text: str, terms: dict) -> str:
    """Заменяем термины в тексте. Сначала длинные, чтобы не срубить части длинных фраз."""
    sorted_terms = sorted(terms.items(), key=lambda x: len(x[0]), reverse=True)
    for original, replacement in sorted_terms:
        # word boundary чтобы не заменять части слов
        pattern = re.compile(re.escape(original), re.IGNORECASE)
        def _replace(m):
            match_text = m.group()
            # Сохраняем регистр первой буквы
            if match_text[0].isupper():
                return replacement[0].upper() + replacement[1:]
            return replacement
        text = pattern.sub(_replace, text)
    return text


def process_directory(input_dir: str, output_dir: str, terms: dict):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    processed = 0
    for file_path in input_path.glob("**/*.md"):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        new_content = replace_in_text(content, terms)

        rel_path = file_path.relative_to(input_path)
        out_file = output_path / rel_path
        out_file.parent.mkdir(parents=True, exist_ok=True)

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(new_content)

        processed += 1
        print(f"✓ {file_path.name}")

    print(f"\nГотово. Обработано файлов: {processed}")


if __name__ == "__main__":
    import sys
    root = Path(__file__).parent.parent

    terms_map_path = root / "terms_map.json"
    terms = load_terms_map(str(terms_map_path))

    # Если передан путь к исходникам — обрабатываем его
    # Иначе просто выводим словарь замен
    if len(sys.argv) > 1:
        input_dir = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else str(root / "knowledge_base")
        process_directory(input_dir, output_dir, terms)
    else:
        print("Использование: python replace_terms.py <input_dir> [output_dir]")
        print(f"\nЗагружено {len(terms)} правил замены из terms_map.json")
        print("Пример:")
        for k, v in list(terms.items())[:5]:
            print(f"  '{k}' → '{v}'")
