# ghostrouter

[English](README.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md) · [한국어](README.ko.md) · [Русский](README.ru.md) · **Deutsch**

Der lernende `LLM`-Router. Intelligentes Routing über 10+ Anbieter hinweg, mit Fallback, Circuit Breakern, Budget-Tracking und Post-Model-Redaction.

## Warum

Jedes Projekt codiert Modellnamen und `API`-Clients fest ein. Wird ein Modell langsam, rate-limitiert oder lehnt es eine Anfrage ab, scheitert der Aufruf einfach. ghostrouter leitet jeden Aufruf automatisch an das jeweils beste verfügbare Modell weiter — Circuit Breaker, Fallback-Ketten und Post-Model-Redaction sind fest integriert.

## Installation

```bash
pip install ghostrouter
```

## Schnellstart

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

Oder starte ihn als Daemon und rufe ihn per `HTTP` auf:

```bash
ghostrouter serve            # binds to localhost:8265
ghostrouter run claude "Explain recursion"
ghostrouter result <job_id> --poll
```

## Architektur

```
call → bouncer → eligibility filter → routing → adapter → redaction → result
                                          ↓
                                   circuit breaker
                                          ↓
                                   fallback chain
```

1. **Routing** — bewertet geeignete Modelle nach Vertrauensstufe, Kosten und bisheriger Latenz
2. **Eignungsprüfung** — filtert Modelle nach Intent, Ausführlichkeit und Determinismus-Anforderungen
3. **Fallback** — probiert Modelle der Reihe nach und wechselt bei Timeout, Fehler, Ablehnung oder Rate-Limit
4. **Adapter** — dünne Schicht pro Anbieter; kein geteilter Zustand zwischen Anbietern
5. **Redaction** — wirkt auf die Modellausgabe (nicht auf den Prompt); entfernt durchgesickerte Secrets, E-Mail-Adressen und Telefonnummern

## Anbieter

| Anbieter | Modelle |
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
| Ollama | Beliebiges lokales Modell |

Setze die passenden `*_API_KEY`-Umgebungsvariablen, um den jeweiligen Anbieter zu aktivieren.

## Kernfunktionen

- **Circuit Breaker** — öffnen sich automatisch bei wiederholten Fehlern und erholen sich über halboffene Probes
- **Vertrauensstufen** — Modelle sind mit Trust-Leveln markiert; Aufrufe können eine Mindeststufe verlangen
- **Post-Model-Redaction** — die Ausgabe wird vor der Rückgabe an den Aufrufer gescannt und bereinigt
- **Ausführungs-Traces** — jeder Aufruf protokolliert, welche Modelle versucht wurden, die Timings und die Ergebnisse
- **Strukturierte Schemata** — durchgängig `Pydantic v2`; strenge Validierung an jeder Grenze
- **Async-nativ** — basiert auf `httpx` + `asyncio`; läuft in `Starlette` mit geringem Overhead

## Spine-Integration (optional)

ghostrouter unterstützt [maelspine](https://github.com/adam-scott-thomas/maelspine) für import-freien Zugriff auf Registries in einer größeren Anwendung:

```python
from ghostrouter.boot import boot

core = boot()  # idempotent singleton

# Any module, anywhere:
from spine import Core
registry = Core.instance().get("model_registry")
```

## HTTP API

| Methode | Pfad | Beschreibung |
|--------|------|-------------|
| POST | `/call` | Einen Aufruf einreichen |
| GET | `/result/{job_id}` | Ergebnis abfragen |
| GET | `/health` | Health-Status und Job-Statistiken |
| GET | `/jobs` | Letzte Jobs auflisten |

## Teil des GhostLogic-Stacks

```
maelspine   — frozen capability registry
ghostrouter   — LLM orchestration gateway  ← you are here
ghostserver — evidence server (Blackbox)
```

## Lizenz

Apache-2.0 — Copyright 2026 Adam Thomas / GhostLogic LLC
