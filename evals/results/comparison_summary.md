# Architecture comparison

Cases: 13 (26 runs per architecture). `rules_only` is a no-LLM ablation, not a shipped option.

| Metric | rules_only | single | staged |
|---|---:|---:|---:|
| Public cases passed | 12/12 | 12/12 | 12/12 |
| Extended cases passed | 12/14 | 14/14 | 14/14 |
| Avg end-to-end latency (s) | 0.0 | 16.8 | 174.2 |
| Avg active latency excl. rate-limit waits (s) | 0.0 | 5.8 | 7.5 |
| Avg LLM calls | 0.0 | 6.0 | 6.5 |
| Avg tool calls | 5.0 | 5.1 | 5.1 |
| Avg tokens (prompt+completion) | 0 | 9,415 | 8,023 |
| Runs where guardrail filled skipped tools | 26 | 3 | 0 |
| Runs where model route was relaxed and overridden | 0 | 0 | 2 |
| LLM errors (fell back to rules) | 26 | 0 | 0 |

## Failures

- **rules_only / EXT-03** (rep 0): risk flag missing (expected one of: ['existing_tool_overlap', 'overlap'])
- **rules_only / EXT-03** (rep 1): risk flag missing (expected one of: ['existing_tool_overlap', 'overlap'])
