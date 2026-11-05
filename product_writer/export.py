"""Plain, copyable Markdown; escape source-controlled Markdown markup."""

import html
import re


def text(value: str) -> str:
    value = html.escape(value, quote=False)
    return re.sub(r"([\\`*_{}\[\]()#+.!|>~-])", r"\\\1", value)


def markdown(result: dict) -> str:
    lines = [f"# {text(result['seo_title'])}", "", "Статус: черновик, требуется проверка менеджером.",
             "", "## SEO-описание", "", text(result["seo_description"]), "", "## Характеристики", "",
             f"1. Производитель: {text(result['manufacturer'])}"]
    for index, spec in enumerate(result["key_specs"], 2):
        lines.append(f"{index}. {text(spec['name'])}: {text(spec['value'])}")
    lines += ["", "## Поисковые запросы", ""]
    lines += [f"{index}. {text(query)}" for index, query in enumerate(result["search_queries"], 1)]
    lines += ["", "## Для менеджера", "", text(result["commercial"]), "", "## Требует уточнения", ""]
    lines += ([f"{i}. {text(w)}" for i, w in enumerate(result["warnings"], 1)] or
              ["Пропуски в переданных полях не обнаружены. Проверьте соответствие конкретной модификации."])
    lines += ["", "## Источники", ""]
    lines += ([f"{i}. {text(url)}" for i, url in enumerate(result["sources"], 1)] or ["Не переданы."])
    meta = result["meta"]
    lines += ["", "## Проверка формата", "", f"Заголовок: {meta['title_characters']} символов. "
              f"Описание: {meta['description_characters']} символов.", "",
              f"Режим: {text(meta['mode'])}. Модель: {text(meta['model'])}.", ""]
    return "\n".join(lines)
