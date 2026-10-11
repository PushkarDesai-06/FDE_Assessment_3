# Architecture Decision Memo

## Decision
Ship **Architecture A, the single agent**, with the deterministic policy engine and the guardrail merge.

## Evidence
Same 13 cases, run twice per architecture on Groq `openai/gpt-oss-120b`: 6 public, 4 more dataset requests and 3 synthetic hidden-style records (`python evals/compare_architectures.py --reps 2`).

| Metric | Single agent | Staged / 2-agent |
|---|---:|---:|
| Runs passing quality checks | 26/26 | 26/26 |
| Avg active latency (excl. rate-limit waits) | 5.8 s | 7.5 s |
| Median end-to-end latency (free-tier keys) | 16.0 s | 26.1 s |
| Avg LLM calls | 6.0 | 6.5 |
| Avg tool calls | 5.1 | 5.1 |
| Avg tokens | 9.4k | 8.0k |
| Same route on both reps | 13/13 | 13/13 |
| Notable policy/grounding failures | Skipped one evidence tool in 3/26 runs (the guardrail ran it) | Reviewer proposed a *looser* route in 2/26 runs, both with unavailable or conflicting vendor evidence (code kept the escalation) |

Both architectures reached the same final route on every case. A no-LLM `rules_only` ablation passed 24/26: it missed only the "existing tool already covers this" judgement (EXT-03), which both agents got right.

## Trade-offs
**Staged gained:** about 15% fewer tokens (the reviewer reads one compact evidence pack) and a clean boundary between gathering evidence and deciding.

**Staged lost:**
- About 30% more active latency and 0.5 more LLM calls per request.
- A second prompt, a handoff schema and another failure point.
- Twice, the reviewer treated "vendor-risk API unavailable" and "registry vs API conflict" as ordinary specialist reviews. The code guardrail restored the escalation.

The single agent's lapses were smaller: it occasionally skipped `get_vendor_profile`, which the guardrail runs anyway.

Most of the reliability comes from the shared design, not from the agent count:
- Code computes the minimum approvals, flags and missing information.
- The model can only add to them or choose a stricter route.
- Evidence is generated from tool outputs.
- `human_review_required` is always true.

## Risks / limitations
- **The test set is small** (13 cases, some written by me). Validate on a larger labelled set from real procurement history.
- **Check the model's judgement against people.** Calibrate overlap and justification decisions with procurement staff; routing PUB-01 to "check the existing SignFlow seats" is defensible but conservative.
- **Injection detection is partly pattern-based.** The model catches paraphrases, but it needs red-team cases.
- **Free-tier rate limits dominate wall-clock time** (8K tokens/minute per key). Production needs a paid tier; on Groq, gpt-oss makes one tool call per turn.
- **The policy is hand-encoded.** Policy changes need code changes and tests.

## Why this is the right MVP
The client needs a correct routing decision with grounded evidence and human control. Code already guarantees that, and one agent adds the judgement the rules can't make. The second agent matched the first on quality and added latency, LLM calls and complexity. It also produced the only cases where the model tried to relax a required escalation. The simpler system performs as well, so it is the one to ship.
