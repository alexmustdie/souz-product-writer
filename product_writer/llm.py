"""DeepSeek JSON mode over the official OpenAI-compatible API."""

import os

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

from .models import AppError


class DeepSeek:
    def __init__(self, api_key: str | None = None, model: str | None = None, client=None):
        key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not key and client is None:
            raise AppError("Не задан DEEPSEEK_API_KEY. Заполните .env или переменную окружения.")
        self.model = model or os.getenv("DEEPSEEK_MODEL", "deepseek-flash")
        self.client = client or OpenAI(
            api_key=key, base_url="https://api.deepseek.com", timeout=60, max_retries=2
        )
        self.calls = []

    def complete(self, system: str, payload: str) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": payload}],
                response_format={"type": "json_object"},
                extra_body={"thinking": {"type": "disabled"}},
                max_tokens=3500,
                temperature=0,
            )
        except APITimeoutError:
            raise AppError("Истекло время ожидания DeepSeek. Попробуйте позже.") from None
        except APIConnectionError:
            raise AppError("Не удалось подключиться к DeepSeek. Проверьте сеть.") from None
        except APIStatusError as error:
            messages = {400: "API отклонил параметры запроса.", 401: "Ключ DeepSeek не принят.",
                        402: "Недостаточно средств на счете DeepSeek.", 403: "Доступ к DeepSeek запрещен.",
                        404: "Модель или API не найдены.", 422: "API не принял параметры модели.",
                        429: "Превышен лимит запросов DeepSeek."}
            message = messages.get(error.status_code, "Ошибка сервиса DeepSeek.")
            raise AppError(f"{message} HTTP {error.status_code}") from None

        if not isinstance(response, ChatCompletion):
            raise AppError("DeepSeek вернул некорректный формат ответа API.")
        try:
            # The SDK constructs models permissively. Validate the whole envelope
            # before dereferencing choices, messages or usage from the network.
            response = ChatCompletion.model_validate(response.model_dump(warnings=False), strict=True)
        except ValidationError:
            raise AppError("DeepSeek вернул поврежденную структуру ответа API.") from None
        self.calls.append({"id": response.id, "model": response.model,
                           "usage": response.usage.model_dump() if response.usage else None})
        if not response.choices:
            raise AppError("DeepSeek вернул ответ без вариантов текста.")
        choice = response.choices[0]
        if choice.finish_reason != "stop":
            raise AppError("DeepSeek не завершил ответ (лимит токенов или отказ). Результат не сохранен.")
        content = choice.message.content if choice.message else None
        if not isinstance(content, str) or not content.strip():
            raise AppError("DeepSeek вернул пустой ответ. Попробуйте повторить запрос.")
        return content
