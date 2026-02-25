# Project Guidelines

## Overview
Cohere documentation site (Fern-powered) + Python exploration scripts. Uses `uv` for Python and Fern CLI for docs generation.

## Cohere API Reference

### Client Setup (v2 API)
```python
import cohere
from dotenv import load_dotenv
load_dotenv()
co = cohere.ClientV2(api_key=os.environ["CO_API_KEY"])
```

### Current Models
| Task | Model | Context |
|------|-------|---------|
| Text generation | `command-a-03-2025` | 256k |
| Reasoning | `command-a-reasoning-08-2025` | 256k |
| Vision | `command-a-vision-07-2025` | 128k |
| Embeddings | `embed-v4.0` | 128k |
| Rerank | `rerank-v4.0-pro` | 32k |

### Core APIs

**Chat** — text generation, RAG, tool use:
```python
response = co.chat(
    model="command-a-03-2025",
    messages=[{"role": "user", "content": "Hello"}],
    documents=[{"data": {"snippet": "..."}}],  # RAG
    tools=[...],  # Function calling
    response_format={"type": "json_object", "schema": {...}}  # Structured output
)
```

**Embed** — vectors for search/classification:
```python
response = co.embed(
    model="embed-v4.0",
    texts=["query"],
    input_type="search_query",  # search_document | search_query | classification | clustering
    embedding_types=["float"]
)
```

**Rerank** — re-order by relevance:
```python
results = co.rerank(model="rerank-v4.0-pro", query="...", documents=[...], top_n=5)
```

### Streaming
```python
for event in co.chat_stream(model="command-a-03-2025", messages=[...]):
    if event.type == "content-delta":
        print(event.delta.message.content.text, end="")
```

### Tool Use Pattern
```python
tools = [{"type": "function", "function": {"name": "...", "parameters": {...}}}]
response = co.chat(model="command-a-03-2025", messages=messages, tools=tools)
if response.message.tool_calls:
    # Execute tools, then send results back in next message
```

### RAG with Citations
Pass `documents` parameter → response includes `message.citations` with source references.

## Code Style

### Python
- Use `CO_API_KEY` env var (loaded via `python-dotenv`)
- SDK: `cohere.ClientV2()` (v2 API)
- Output JSON to `outputs/` directory
- See [helloworld.py](helloworld.py) as reference

### TSX Components ([fern/components/](fern/components/))
- File naming: kebab-case (`simple-card.tsx`)
- Component exports: PascalCase (`SimpleCard`)
- Props: TypeScript interfaces (`SimpleCardProps`)
- Icons: default exports in [fern/icons/](fern/icons/)
- Styling: Tailwind classes + custom CSS from [fern/assets/input.css](fern/assets/input.css)
- Dark mode: use `dark:` prefix variants

## Architecture

```
fern/
├── docs.yml          # Global config, custom components, typography
├── v1.yml / v2.yml   # Version-specific navigation
├── apis/             # SDK generators config
├── components/       # Custom TSX components for docs
├── icons/            # SVG icon components
├── pages/            # MDX documentation content
│   ├── v2/           # Primary API version docs
│   ├── changelog/    # YYYY-MM-DD-title.mdx format
│   └── cookbooks/    # Tutorial content
└── assets/           # Fonts, images, input.css
```

## Build and Test

```bash
# Python
uv sync
cp .env.example .env && ${EDITOR:-nano} .env
uv run python helloworld.py

# Fern docs
fern docs dev          # Local preview
fern generate          # Generate SDKs

# CSS (Tailwind)
npx tailwindcss -i fern/assets/input.css -o dist/output.css
```

## Documentation Conventions

### MDX Frontmatter (required)
```yaml
---
title: "Page Title"
slug: "docs/page-slug"
description: "SEO description"
---
```

### Common Fern Components
`<Steps>`, `<Tabs>`, `<Tab>`, `<Info>`, `<Warning>`

### Code blocks with language label
````markdown
```python PYTHON
# code here
```
````

### Multi-platform examples
Use `<Tabs>` for Cohere Platform / Bedrock / Azure / SageMaker variants.

### Changelog files
Name format: `YYYY-MM-DD-title.mdx` with `createdAt` timestamp in frontmatter.

## Deployment Options
- **Cohere Platform**: Direct API with `CO_API_KEY`
- **AWS Bedrock**: Model IDs differ (e.g., `cohere.command-r-plus-v1:0`)
- **Azure AI Foundry**: Uses `/v1/chat/completions` endpoint
- **SageMaker**: Custom endpoint deployments

## Security

- **Never commit `.env`** — use `.env.example` as template
- Store `CO_API_KEY` in `.env` (gitignored)
- `generators.yml` tokens (`${NPM_TOKEN}`, etc.) are CI/CD injected
- Documentation examples use `"COHERE_API_KEY"` placeholder for users to replace
