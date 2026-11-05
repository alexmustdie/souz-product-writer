"""Draft, validate, review, then assemble facts without model intervention."""

import json
import re
from datetime import datetime, timezone

from pydantic import ValidationError

from .models import AppError, Draft, Product, Review, strict_json, validation_summary
from .prompts import PROMPT_VERSION, REVIEWER, WRITER


def numeric_tokens(text: str) -> set[str]:
    # Compare whole decimal tokens, not substrings (1 != 10).
    return {token.replace(",", ".") for token in re.findall(r"\d+(?:[.,]\d+)?", text)}


def validate_draft(raw: str, product: Product) -> Draft:
    parsed = None
    try:
        parsed = strict_json(raw)
        draft = Draft.model_validate(parsed)
    except (ValueError, RecursionError) as error:
        detail = validation_summary(error) if isinstance(error, ValidationError) else "ожидался корректный JSON"
        if isinstance(parsed, dict) and isinstance(parsed.get("seo_description"), str):
            detail += f"; длина seo_description={len(parsed['seo_description'].strip())}, требуется 800–1200"
        raise AppError(f"Некорректный черновик: {detail}") from None
    source = json.dumps({"name": product.name, "manufacturer": product.manufacturer,
                         "specs": product.specs, "description": product.description}, ensure_ascii=False)
    extra = numeric_tokens(draft.model_dump_json()) - numeric_tokens(source)
    if extra:
        raise AppError("В тексте появились числа, отсутствующие в исходных фактах")
    return draft


def assemble(product: Product, draft: Draft, *, mode: str, model: str, calls: list) -> dict:
    keys = dict.fromkeys([*product.specs, *product.expected_specs])
    features = [{"name": key, "value": product.specs.get(key) or "данных нет",
                 "source": f"specs.{key}" if product.specs.get(key) else None} for key in keys]
    missing = product.missing()
    return {
        "status": "draft", "needs_human_review": True,
        "product_name": product.name, "manufacturer": product.manufacturer or "данных нет",
        **draft.model_dump(), "key_specs": features,
        "missing_fields": missing,
        "warnings": [f"{field}: данных нет, требуется уточнение" for field in missing],
        "sources": product.sources,
        "meta": {"mode": mode, "model": model, "prompt_version": PROMPT_VERSION,
                 "generated_at": datetime.now(timezone.utc).isoformat(),
                 "title_characters": len(draft.seo_title),
                 "description_characters": len(draft.seo_description), "api_calls": calls},
    }


def generate(product: Product, llm) -> dict:
    feedback = ""
    previous_draft = ""
    for attempt in range(2):
        payload = json.dumps({"product": product.facts(), "missing_fields": product.missing(),
                              "correction": feedback, "previous_draft": previous_draft}, ensure_ascii=False)
        raw = llm.complete(WRITER, payload)
        try:
            draft = validate_draft(raw, product)
            review_raw = llm.complete(REVIEWER, json.dumps(
                {"product": product.facts(), "draft": draft.model_dump()}, ensure_ascii=False))
            try:
                review = Review.model_validate(strict_json(review_raw))
            except (ValueError, RecursionError):
                raise AppError("Некорректный JSON проверки фактов") from None
            if review.unsupported_claims or review.contradictions:
                raise AppError("Проверка фактов: " + "; ".join(review.unsupported_claims + review.contradictions))
            return assemble(product, draft, mode="live_api", model=llm.model, calls=llm.calls)
        except AppError as error:
            feedback = str(error)
            previous_draft = raw
            if attempt == 1:
                # Don't export questionable prose or raw model responses.
                missing = product.missing()
                missing_note = " Недостающие поля: " + "; ".join(missing) + "." if missing else ""
                raise AppError("Не удалось получить проверенный черновик за 2 попытки. "
                               "Уточните исходные данные или повторите запуск. "
                               + (feedback if feedback.startswith("Некорректный черновик") else
                                  "Проверка содержания или формата не пройдена.") + missing_note) from None
    raise AssertionError("unreachable")
