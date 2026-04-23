# ghostrouter

[English](README.md) · **中文** · [日本語](README.ja.md) · [한국어](README.ko.md) · [Русский](README.ru.md) · [Deutsch](README.de.md)

学习型 `LLM` 路由器。面向 10 多家提供商的智能路由，内置回退、熔断器、预算跟踪与模型输出脱敏。

## 为什么需要它

每个项目都会硬编码模型名与 `API` 客户端。当模型变慢、被限流或拒绝请求时，调用就直接失败。ghostrouter 会自动将每次调用路由到当前最合适的可用模型，熔断器、回退链与模型输出脱敏都已内建。

## 安装

```bash
pip install ghostrouter
```

## 快速开始

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

或者以守护进程方式运行，通过 `HTTP` 调用：

```bash
ghostrouter serve            # binds to localhost:8265
ghostrouter run claude "Explain recursion"
ghostrouter result <job_id> --poll
```

## 架构

```
call → bouncer → eligibility filter → routing → adapter → redaction → result
                                          ↓
                                   circuit breaker
                                          ↓
                                   fallback chain
```

1. **路由** — 根据信任等级、成本与历史延迟为符合条件的模型打分
2. **资格过滤** — 按意图、详尽度与确定性要求筛选模型
3. **回退** — 按顺序尝试模型；在超时、错误、拒绝或限流时切换
4. **适配器** — 每家提供商一个轻量适配层，各提供商之间不共享状态
5. **脱敏** — 作用于模型输出（而非提示词），剥离泄漏的密钥、邮箱与电话号码

## 提供商

| 提供商 | 模型 |
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
| Ollama | 任意本地模型 |

设置相应的 `*_API_KEY` 环境变量即可启用各家提供商。

## 核心特性

- **熔断器** — 连续失败时自动熔断，采用半开探测逐步恢复
- **信任等级** — 为模型打上信任标签；调用可要求最低等级
- **模型输出脱敏** — 返回调用方之前会扫描并清理输出
- **执行追踪** — 每次调用都记录尝试过的模型、耗时与结果
- **结构化模式** — 全程使用 `Pydantic v2`，在每个边界进行严格校验
- **原生异步** — 基于 `httpx` + `asyncio` 构建，以 `Starlette` 运行，开销极低

## Spine 集成（可选）

ghostrouter 支持 [maelspine](https://github.com/adam-scott-thomas/maelspine)，可在更大型的应用中无需 `import` 即可访问各注册表：

```python
from ghostrouter.boot import boot

core = boot()  # idempotent singleton

# Any module, anywhere:
from spine import Core
registry = Core.instance().get("model_registry")
```

## HTTP API

| 方法 | 路径 | 说明 |
|--------|------|-------------|
| POST | `/call` | 提交一次调用 |
| GET | `/result/{job_id}` | 轮询结果 |
| GET | `/health` | 健康状态与任务统计 |
| GET | `/jobs` | 列出近期任务 |

## 属于 GhostLogic 技术栈

```
maelspine   — frozen capability registry
ghostrouter   — LLM orchestration gateway  ← you are here
ghostserver — evidence server (Blackbox)
```

## 许可证

Apache-2.0 — Copyright 2026 Adam Thomas / GhostLogic LLC
