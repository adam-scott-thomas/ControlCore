# ghostrouter

[English](README.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · **Русский** · [Deutsch](README.de.md)

Обучающийся `LLM`-маршрутизатор. Интеллектуальная маршрутизация между 10+ провайдерами с фолбэком, предохранителями, отслеживанием бюджета и пост-модельной редакцией вывода.

## Зачем это нужно

В каждом проекте имена моделей и `API`-клиенты зашиты жёстко. Когда модель работает медленно, упирается в ограничения скорости или отказывает в запросе, вызов просто падает. ghostrouter автоматически направляет каждый вызов в наиболее подходящую доступную модель — предохранители, цепочки фолбэков и пост-модельная редакция встроены по умолчанию.

## Установка

```bash
pip install ghostrouter
```

## Быстрый старт

```python
import asyncio
from ghostrouter.config import initialize_controlcore
from ghostrouter.adapters.executor import execute_call
from ghostrouter.schemas import ControlCoreCall, Caller, Intent, Target

# Initialize registries (reads env vars for API keys)
config, model_registry, adapter_registry = initialize_controlcore()

# Build a call
call = ControlCoreCall(
    caller=Caller(handle="my-app", account_id="00000000-0000-0000-0000-000000000000"),
    intent=Intent(**{"class": "lookup"}),           # `class` is aliased; use dict unpacking
    target=Target(type="model", alias="claude"),
    prompt="What is the capital of France?",
)

# Execute with automatic fallback
result, trace = asyncio.run(execute_call(call, model_registry, adapter_registry))
print(result.answer)
```

Либо запустите как демона и обращайтесь по `HTTP`:

```bash
ghostrouter serve            # binds to localhost:8265
ghostrouter run claude "Explain recursion"
ghostrouter result <job_id> --poll
```

## Архитектура

```
call → bouncer → eligibility filter → routing → adapter → redaction → result
                                          ↓
                                   circuit breaker
                                          ↓
                                   fallback chain
```

1. **Маршрутизация** — оценивает подходящие модели по уровню доверия, стоимости и истории задержек
2. **Фильтр пригодности** — отбирает модели по намерению, уровню подробности и требованиям к детерминизму
3. **Фолбэк** — пробует модели по порядку и переключается при таймауте, ошибке, отказе или ограничении скорости
4. **Адаптер** — тонкая прослойка для каждого провайдера; состояние между провайдерами не разделяется
5. **Редакция** — применяется к выходу модели (а не к промпту), удаляя утёкшие секреты, адреса электронной почты и номера телефонов

## Провайдеры

| Провайдер | Модели |
|----------|--------|
| OpenAI | GPT-4, GPT-4o, o1, o1-mini |
| Anthropic | Claude Sonnet, Opus, Haiku |
| Google | Gemini 1.5 Pro/Flash, Gemini 2.0 |
| xAI | Grok, Grok-2 |
| Mistral | Mistral Large/Medium/Small, Codestral |
| Groq | Llama-70B, Llama-8B, Mixtral, Gemma |
| Together | Llama-405B, Qwen-72B, DeepSeek-V3 |
| DeepSeek | DeepSeek Chat, Coder, Reasoner |
| Perplexity | Sonar (search-augmented) |
| Ollama | Любая локальная модель |

Чтобы включить провайдера, задайте соответствующую переменную окружения `*_API_KEY`.

## Ключевые возможности

- **Предохранители** — автоматически срабатывают при повторяющихся сбоях, восстанавливаются через полуоткрытые пробные запросы
- **Уровни доверия** — моделям присваиваются уровни доверия; вызовы могут требовать минимальный уровень
- **Пост-модельная редакция** — вывод сканируется и очищается перед возвратом вызывающей стороне
- **Трассы выполнения** — каждый вызов фиксирует, какие модели были задействованы, их тайминги и результаты
- **Строгие схемы** — `Pydantic v2` во всех компонентах; строгая валидация на каждой границе
- **Асинхронность из коробки** — на базе `httpx` + `asyncio`; запускается в `Starlette` с минимальными накладными расходами

## Интеграция со Spine (опционально)

ghostrouter поддерживает [maelspine](https://github.com/adam-scott-thomas/maelspine) для беспортового (без `import`) доступа к реестрам во всём крупном приложении:

```python
from ghostrouter.boot import boot

core = boot()  # idempotent singleton

# Any module, anywhere:
from spine import Core
registry = Core.instance().get("model_registry")
```

## HTTP API

| Метод | Путь | Описание |
|--------|------|-------------|
| POST | `/call` | Отправить вызов |
| GET | `/result/{job_id}` | Опрос результата |
| GET | `/health` | Состояние и статистика задач |
| GET | `/jobs` | Список последних задач |

## Часть стека GhostLogic

```
maelspine   — frozen capability registry
ghostrouter   — LLM orchestration gateway  ← you are here
ghostserver — evidence server (Blackbox)
```

## Лицензия

Apache-2.0 — Copyright 2026 Adam Thomas / GhostLogic LLC
