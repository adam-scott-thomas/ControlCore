# ghostrouter

[English](README.md) · [中文](README.zh-CN.md) · **日本語** · [한국어](README.ko.md) · [Русский](README.ru.md) · [Deutsch](README.de.md)

学習する `LLM` ルーター。10 以上のプロバイダー間でのインテリジェントルーティング、フォールバック、サーキットブレーカー、予算追跡、そしてモデル出力後のマスキングを提供します。

## なぜ必要か

どのプロジェクトもモデル名と `API` クライアントをハードコードしています。モデルが遅い、レート制限に引っかかる、リクエストを拒否されるといった状況では、呼び出しはそのまま失敗します。ghostrouter は各呼び出しを自動的に最適な利用可能モデルへルーティングします。サーキットブレーカー、フォールバックチェーン、出力後マスキングはすべて組み込み済みです。

## インストール

```bash
pip install ghostrouter
```

## クイックスタート

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

あるいはデーモンとして起動し、`HTTP` 経由で呼び出すこともできます。

```bash
ghostrouter serve            # binds to localhost:8265
ghostrouter run claude "Explain recursion"
ghostrouter result <job_id> --poll
```

## アーキテクチャ

```
call → bouncer → eligibility filter → routing → adapter → redaction → result
                                          ↓
                                   circuit breaker
                                          ↓
                                   fallback chain
```

1. **ルーティング** — 信頼ティア、コスト、レイテンシ履歴に基づき、対象モデルをスコアリングします
2. **適格性フィルタ** — インテント、詳細度、決定性の要件でモデルを絞り込みます
3. **フォールバック** — モデルを順に試し、タイムアウト、エラー、拒否、レート制限で切り替えます
4. **アダプター** — プロバイダーごとの薄いシムで、プロバイダー間で状態を共有しません
5. **マスキング** — プロンプトではなくモデル出力に適用され、漏洩したシークレット、メールアドレス、電話番号を除去します

## プロバイダー

| プロバイダー | モデル |
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
| Ollama | 任意のローカルモデル |

各プロバイダーを有効化するには、対応する `*_API_KEY` 環境変数を設定してください。

## 主な機能

- **サーキットブレーカー** — 連続した失敗で自動的にオープンし、ハーフオープンプローブで復旧します
- **信頼ティア** — モデルに信頼レベルのタグを付与し、呼び出し側は最低ティアを要求できます
- **出力後マスキング** — 呼び出し元に返す前に出力をスキャンしサニタイズします
- **実行トレース** — すべての呼び出しについて、試行したモデル、タイミング、結果を記録します
- **構造化スキーマ** — 全体で `Pydantic v2` を使用し、各境界で厳格な検証を行います
- **非同期ネイティブ** — `httpx` と `asyncio` をベースに構築し、低オーバーヘッドな `Starlette` 上で動作します

## Spine 統合（任意）

ghostrouter は [maelspine](https://github.com/adam-scott-thomas/maelspine) をサポートしており、より大規模なアプリケーション全体で `import` 不要のレジストリアクセスを実現します。

```python
from ghostrouter.boot import boot

core = boot()  # idempotent singleton

# Any module, anywhere:
from spine import Core
registry = Core.instance().get("model_registry")
```

## HTTP API

| メソッド | パス | 説明 |
|--------|------|-------------|
| POST | `/call` | 呼び出しを送信 |
| GET | `/result/{job_id}` | 結果をポーリング |
| GET | `/health` | ヘルスチェックとジョブ統計 |
| GET | `/jobs` | 最近のジョブ一覧 |

## GhostLogic スタックの一部

```
maelspine   — frozen capability registry
ghostrouter   — LLM orchestration gateway  ← you are here
ghostserver — evidence server (Blackbox)
```

## ライセンス

Apache-2.0 — Copyright 2026 Adam Thomas / GhostLogic LLC
