"""CLI entry point. stdout contains only the result, stderr contains safe errors."""

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

from .export import markdown
from .llm import DeepSeek
from .models import AppError, read_product
from .service import generate


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Черновик B2B-карточки через DeepSeek API")
    parser.add_argument("input", type=Path, help="JSON одного товара")
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    parser.add_argument("--output", type=Path, help="Файл результата; по умолчанию stdout")
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", help="Переопределить DEEPSEEK_MODEL")
    parser.add_argument("--validate-only", action="store_true", help="Проверить вход без обращения к API")
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    try:
        if args.output and args.output.resolve() in {args.input.resolve(), args.env_file.resolve()}:
            raise AppError("Результат нельзя записывать поверх входного файла или .env")
        # Bounded read before parsing; UTF-8 BOM from Windows editors is accepted.
        with args.input.open("rb") as stream:
            content = stream.read(100_001)
        if len(content) > 100_000:
            raise AppError("Входной JSON превышает 100 КБ")
        product = read_product(content.decode("utf-8-sig"))
        if args.validate_only:
            result = {"valid": True, "missing_fields": product.missing()}
            rendered = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
        else:
            load_dotenv(args.env_file, override=False)
            result = generate(product, DeepSeek(model=args.model))
            rendered = (markdown(result) if args.format == "markdown" else
                        json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        return 0
    except (AppError, OSError, UnicodeError) as error:
        message = str(error) if isinstance(error, AppError) else "Не удалось прочитать или записать UTF-8 файл."
        print(f"Ошибка: {message}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
