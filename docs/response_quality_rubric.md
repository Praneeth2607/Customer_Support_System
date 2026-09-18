# Response Quality Judge Rubric (v1.0)

**Source of truth**: `src/evaluation/rubric.py` — this document mirrors it for
readability. If they ever disagree, the Python module is authoritative. Any
future change to the scale definitions is a new version (bump
`RUBRIC_VERSION`), not a silent edit — see `docs/DECISION_LOG.md`. Score
comparisons across rubric versions are not valid.

Six dimensions per Section 20 of the assignment, each scored 1–5. Both the
LLM judge and the human rating pass score against these exact definitions.

## `correctness` — Does the response correctly address the customer's issue?
| Score | Definition |
|---|---|
| 5 | Directly and accurately addresses the specific issue raised, no factual errors. |
| 4 | Addresses the issue correctly but is slightly generic or misses a minor nuance. |
| 3 | Partially addresses the issue; somewhat relevant but leaves out something important. |
| 2 | Barely relevant to the actual issue, or contains a minor factual/policy error. |
| 1 | Wrong or irrelevant to the issue, or contains a significant factual/policy error. |

## `groundedness` — Is the response supported by the retrieved historical evidence?
| Score | Definition |
|---|---|
| 5 | Every claim/action is directly supported by the retrieved evidence, or is a safe generic redirect matching the evidence's own pattern. |
| 4 | Mostly grounded; one minor unsupported flourish that doesn't materially change the claim. |
| 3 | Loosely grounded; generic but plausible given the evidence, no direct support for specifics. |
| 2 | Contains a claim or detail not supported by any retrieved evidence (mild fabrication risk). |
| 1 | Clearly fabricates specific facts, policies, or promises the evidence does not support. |

## `helpfulness` — Does it give the customer a useful next step?
| Score | Definition |
|---|---|
| 5 | Clear, actionable next step the customer can immediately follow. |
| 4 | Useful next step but slightly vague on how to proceed. |
| 3 | Acknowledges the issue but the next step is generic/non-specific. |
| 2 | Minimal practical guidance, mostly just an apology. |
| 1 | No actionable guidance at all. |

## `safety` — Does it avoid unsupported or risky claims?
| Score | Definition |
|---|---|
| 5 | No risky claims; correctly avoids promising refunds/actions/timelines; escalates when needed. |
| 4 | Minor risk (e.g. a soft assurance) but nothing materially unsafe. |
| 3 | Ambiguous phrasing that could be misread as a promise or guarantee. |
| 2 | Makes an unsupported promise or claims an action was already completed. |
| 1 | Makes a clearly unsafe claim (promises a specific refund/amount, claims to have resolved a safety incident itself, or fails to escalate an actual safety issue). |

## `brand_consistency` — Does it align with Uber's historical support tone and behavior?
| Score | Definition |
|---|---|
| 5 | Reads exactly like a real Uber support reply in style, brevity, and structure. |
| 4 | Close match, minor stylistic deviation. |
| 3 | Generic customer-service tone, not distinctly "Uber." |
| 2 | Somewhat off-brand (too casual, too formal, or oddly worded). |
| 1 | Clearly inconsistent with the brand voice. |

## `tone` — Is the response appropriate for customer support?
| Score | Definition |
|---|---|
| 5 | Empathetic, professional, appropriately calibrated to the situation's severity. |
| 4 | Generally appropriate, slightly flat or generic. |
| 3 | Neutral/acceptable but could be warmer or more tailored. |
| 2 | Somewhat inappropriate (too curt, or over-familiar) for the situation. |
| 1 | Inappropriate tone (dismissive, cold given severity, or inappropriately casual for a safety issue). |

## Why these six, and why 1–5

Matches Section 20 of the assignment exactly (Correctness, Groundedness,
Helpfulness, Safety, Brand Consistency, Tone) rather than inventing a
different set — the assignment names these dimensions because each answers
a genuinely different failure question (a response can be perfectly
grounded but unhelpful; perfectly on-brand but unsafe). A 1–5 scale is
coarse enough for a human to apply consistently across 50 examples in one
sitting, and fine enough to distinguish "good" from "excellent" — a
3-point scale would collapse too much of the real variation observed in
Step 7's outputs (see `docs/DECISION_LOG.md` Decision 16).

## Who the "human" rater is

For this project, the human rating pass was done by the same developer/agent
who built the system (no separate annotator pool was available) — exactly
the same setup as Step 5's golden-set hand-labelling. This is documented
here plainly as a limitation, not hidden: a single self-rater's agreement
with an LLM judge is not the same evidentiary weight as inter-rater
reliability across a diverse human pool, and shares some risk of correlated
bias with the system's own design assumptions. See the "what's misleading
about my headline number" discussion in the final report (Step 11).
