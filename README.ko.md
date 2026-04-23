# ghostrouter

[English](README.md) · [中文](README.zh-CN.md) · [日本語](README.ja.md) · **한국어** · [Русский](README.ru.md) · [Deutsch](README.de.md)

학습형 `LLM` 라우터입니다. 10개 이상의 프로바이더에 걸친 지능형 라우팅, 폴백, 서킷 브레이커, 예산 추적, 모델 후 마스킹을 제공합니다.

## 왜 필요한가

모든 프로젝트는 모델명과 `API` 클라이언트를 하드코딩합니다. 모델이 느려지거나 속도 제한에 걸리거나 요청을 거부하면 호출은 그대로 실패합니다. ghostrouter는 매 호출을 자동으로 가장 적합한 사용 가능 모델로 라우팅합니다. 서킷 브레이커, 폴백 체인, 모델 후 마스킹이 기본 내장되어 있습니다.

## 설치

```bash
pip install ghostrouter
```

## 빠른 시작

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

또는 데몬으로 실행하여 `HTTP`로 호출할 수도 있습니다.

```bash
ghostrouter serve            # binds to localhost:8265
ghostrouter run claude "Explain recursion"
ghostrouter result <job_id> --poll
```

## 아키텍처

```
call → bouncer → eligibility filter → routing → adapter → redaction → result
                                          ↓
                                   circuit breaker
                                          ↓
                                   fallback chain
```

1. **라우팅** — 신뢰 등급, 비용, 지연시간 이력을 기반으로 적격 모델의 점수를 매깁니다
2. **적격성 필터** — 인텐트, 상세도, 결정성 요구사항에 따라 모델을 걸러냅니다
3. **폴백** — 모델을 순서대로 시도하며 타임아웃, 오류, 거부, 속도 제한 시 전환합니다
4. **어댑터** — 프로바이더마다 얇은 심(shim)을 두며, 프로바이더 간에 상태를 공유하지 않습니다
5. **마스킹** — 프롬프트가 아닌 모델 출력에 적용되어 유출된 시크릿, 이메일, 전화번호를 제거합니다

## 프로바이더

| 프로바이더 | 모델 |
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
| Ollama | 모든 로컬 모델 |

각 프로바이더를 활성화하려면 해당 `*_API_KEY` 환경 변수를 설정하면 됩니다.

## 주요 기능

- **서킷 브레이커** — 반복되는 실패 시 자동으로 열리고 반개방(half-open) 탐색으로 복구합니다
- **신뢰 등급** — 모델에 신뢰 레벨을 태깅하며, 호출은 최소 등급을 요구할 수 있습니다
- **모델 후 마스킹** — 호출자에게 반환하기 전에 출력을 스캔하고 정리합니다
- **실행 트레이스** — 모든 호출에 대해 시도한 모델, 타이밍, 결과를 기록합니다
- **구조화된 스키마** — 전 구간에서 `Pydantic v2`를 사용하며 모든 경계에서 엄격히 검증합니다
- **비동기 네이티브** — `httpx` + `asyncio` 기반으로 구축되며, 낮은 오버헤드의 `Starlette`에서 동작합니다

## Spine 연동 (선택)

ghostrouter는 [maelspine](https://github.com/adam-scott-thomas/maelspine)을 지원하여 더 큰 애플리케이션 전반에서 `import` 없이 레지스트리에 접근할 수 있습니다.

```python
from ghostrouter.boot import boot

core = boot()  # idempotent singleton

# Any module, anywhere:
from spine import Core
registry = Core.instance().get("model_registry")
```

## HTTP API

| 메서드 | 경로 | 설명 |
|--------|------|-------------|
| POST | `/call` | 호출 제출 |
| GET | `/result/{job_id}` | 결과 폴링 |
| GET | `/health` | 상태 및 작업 통계 |
| GET | `/jobs` | 최근 작업 목록 |

## GhostLogic 스택의 일부

```
maelspine   — frozen capability registry
ghostrouter   — LLM orchestration gateway  ← you are here
ghostserver — evidence server (Blackbox)
```

## 라이선스

Apache-2.0 — Copyright 2026 Adam Thomas / GhostLogic LLC
