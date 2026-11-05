"""Input contract and strict validation; never echo raw input in errors."""

import json
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator


class AppError(Exception):
    """An error safe to show to the CLI user."""


class Model(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, str_strip_whitespace=True)


Text = Annotated[str, Field(min_length=1, max_length=2000)]


class Product(Model):
    name: Annotated[str, Field(min_length=1, max_length=200)]
    manufacturer: Annotated[str, Field(max_length=200)] | None = None
    specs: dict[Text, Annotated[str, Field(max_length=2000)] | None] = Field(default_factory=dict, max_length=50)
    description: Annotated[str, Field(max_length=15000)] | None = None
    expected_specs: list[Text] = Field(default_factory=list, max_length=50)
    sources: list[Annotated[str, Field(max_length=2000)]] = Field(default_factory=list, max_length=10)

    @field_validator("manufacturer", "description")
    @classmethod
    def empty_is_missing(cls, value):
        return value or None

    @field_validator("specs")
    @classmethod
    def normalize_specs(cls, values):
        unknown = {"", "данных нет", "нет данных", "не указано", "неизвестно"}
        return {key: None if value is None or value.casefold() in unknown else value
                for key, value in values.items()}

    @field_validator("specs", mode="before")
    @classmethod
    def reject_colliding_keys(cls, values):
        if isinstance(values, dict):
            keys = [key.strip() for key in values if isinstance(key, str)]
            if len(keys) != len(set(keys)):
                raise ValueError("Названия характеристик совпадают после удаления пробелов")
        return values

    def missing(self) -> list[str]:
        result = []
        if not self.manufacturer:
            result.append("Производитель")
        if not self.description:
            result.append("Описание / назначение")
        if not any(self.specs.values()):
            result.append("Технические характеристики")
        for key in dict.fromkeys([*self.specs, *self.expected_specs]):
            if not self.specs.get(key):
                result.append(key)
        return result

    def facts(self) -> dict:
        # URLs are provenance, never fetched or used as additional knowledge by the LLM.
        return self.model_dump(exclude={"sources"})


class Draft(Model):
    seo_title: Annotated[str, Field(min_length=1, max_length=70)]
    seo_description: Annotated[str, Field(min_length=800, max_length=1200)]
    search_queries: list[Annotated[str, Field(min_length=3, max_length=120)]] = Field(min_length=3, max_length=5)
    commercial: Annotated[str, Field(min_length=20, max_length=650)]

    @field_validator("search_queries")
    @classmethod
    def distinct_queries(cls, values):
        if len({v.casefold() for v in values}) != len(values):
            raise ValueError("Поисковые запросы должны различаться")
        return values


class Review(Model):
    unsupported_claims: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(max_length=20)
    contradictions: list[Annotated[str, Field(min_length=1, max_length=1000)]] = Field(max_length=20)


def strict_json(text: str):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Повторяющиеся ключи JSON")
            result[key] = value
        return result

    def invalid_constant(_):
        raise ValueError("Недопустимая константа JSON")

    return json.loads(text, object_pairs_hook=unique, parse_constant=invalid_constant)


def validation_summary(error: ValidationError) -> str:
    # No input values or provider response bodies in public errors.
    return "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['type']}"
                     for e in error.errors(include_input=False, include_url=False))


def read_product(text: str) -> Product:
    if len(text.encode("utf-8")) > 100_000:
        raise AppError("Входной JSON превышает 100 КБ")
    try:
        return Product.model_validate(strict_json(text))
    except (ValueError, RecursionError) as error:
        detail = validation_summary(error) if isinstance(error, ValidationError) else "ошибка синтаксиса"
        raise AppError(f"Некорректный входной JSON: {detail}") from None
