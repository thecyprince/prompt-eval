# prompt-eval

A lightweight CLI for evaluating Claude prompts against test suites. Define your prompt and test cases in YAML, run them, save a baseline, tweak the prompt, and see exactly what broke and what improved.

Built to demonstrate practical use of the Claude API: prompt caching, async execution, structured output, and LLM-as-judge scoring.

---

## Why this exists

When iterating on a prompt, the question is always: *did this change make things better or worse?* Running cases manually doesn't scale, and eyeballing outputs misses regressions. This tool makes prompt iteration measurable.

A secondary goal: demonstrating how to use the Claude API cost-effectively. Prompt caching alone reduces input token cost by ~90% when running many cases against the same system prompt — the tool surfaces this in every run.

---

## Quick start

```bash
pip install -e .
export ANTHROPIC_API_KEY=...

# Run a suite
python harness.py run evals/support-classifier.yml

# Save a baseline, change the prompt, see what changed
python harness.py run evals/support-classifier.yml --save results/v1.json
# ... edit the prompt in the YAML ...
python harness.py run evals/support-classifier.yml --compare results/v1.json

# Override model
python harness.py run evals/support-classifier.yml --model claude-haiku-4-5

# Print full output for failed cases
python harness.py run evals/support-classifier.yml --show-failures
```

---

## Output

```
Customer Support Ticket Classifier
model: claude-sonnet-4-6  |  cases: 12  |  scorer: json_field_match  |  prompt: a3f2b1c4

┏━━━━━━━━━━━━━━━━━┳━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━┳━━━━━━━┓
┃ Case ID         ┃   ┃ Score ┃ Cache ┃ Latency ┃ Error ┃
┡━━━━━━━━━━━━━━━━━╇━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━╇━━━━━━━┩
│ billing-001     │ ✅ │  1.00 │  💾   │   812ms │       │
│ billing-002     │ ✅ │  1.00 │  🎯   │   634ms │       │
│ technical-001   │ ✅ │  1.00 │  🎯   │   701ms │       │
│ escalate-001    │ ✅ │  1.00 │  🎯   │   589ms │       │
│ ...             │   │       │       │         │       │
└─────────────────┴───┴───────┴───────┴─────────┴───────┘

Tokens
  Uncached input :    1,240
  Cache writes   :    8,960  (1.25× cost)
  Cache reads    :   98,560  (0.1× cost)   ← 11 of 12 cases hit cache
  Output         :      840
  Cache hit rate :     91.7%
  Wall-clock     :     1240ms
```

**Cache legend:** 💾 = cache write (first call, 1.25× cost), 🎯 = cache read (~0.1× cost), — = no cache

---

## Writing a test suite

```yaml
name: "My Classifier"
model: claude-sonnet-4-6   # optional — default is sonnet-4-6
scorer: json_field_match   # json_field_match | exact | llm_judge

prompt: |
  You are a classifier. Given a message, return JSON:
  {"category": "billing|technical|general", "confidence": 0.0-1.0}

cases:
  - id: billing-001
    input: "I was charged twice"
    expected: {category: billing}   # fields that must match

  - id: open-ended-001
    input: "I need help urgently"
    expected:
      is_empathetic: true           # used by llm_judge scorer
      has_next_step: true
    tags: [edge-case]               # optional — for filtering (not yet implemented)
```

### Scorers

| Scorer | How it works | Best for |
|--------|-------------|----------|
| `json_field_match` | Parses response as JSON, checks that all `expected` fields match | Classification, extraction, any structured output |
| `exact` | Response must exactly match `expected.output` | Short, deterministic outputs |
| `llm_judge` | Uses Claude (Haiku) to grade the response against `expected` criteria | Open-ended responses, tone, style |

---

## How prompt caching works here

Every test case sends the same system prompt. The first call writes it to the cache (1.25× normal input cost). Every subsequent call reads from cache (0.1× cost).

For a 500-token system prompt running 20 test cases:
- Without caching: 20 × 500 = 10,000 input tokens at full price
- With caching: 500 at 1.25× + 19 × 500 at 0.1× = 625 + 950 = 1,575 effective tokens
- **~84% cost reduction**

The minimum cacheable prefix is 1,024 tokens (Sonnet) or 4,096 tokens (Opus). Short prompts won't cache — the tool reports `cache_creation_tokens: 0` so you know.

---

## Architecture notes

- **Async**: all test cases run concurrently via `asyncio.gather` — wall-clock time is roughly the slowest single case, not the sum
- **LLM-as-judge**: uses `claude-haiku-4-5` for scoring (fast, cheap) — its system prompt is also cached across all judge calls
- **Diff**: results are saved as JSON with a `prompt_hash` — the compare output flags when the prompt itself changed between runs
- **CI-friendly**: exits with code 1 if any cases fail

---

## Project structure

```
prompt-eval/
├── harness.py          # main CLI
├── evals/
│   ├── support-classifier.yml   # json_field_match example
│   └── tone-checker.yml         # llm_judge example
└── results/            # saved baselines (gitignored)
```
