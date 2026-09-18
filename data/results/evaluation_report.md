# Evaluation Report

**Test set**: `data/golden/golden_evaluation_set.json` -- 200 hand-labelled examples, frozen.
Escalate=True: 130 | Escalate=False: 70

## Intent Classification -- Three-Way Comparison

| System | Accuracy | Macro F1 |
|---|---|---|
| Majority class | 0.29 | 0.0642 |
| TF-IDF + Logistic Regression | 0.71 | 0.7238 |
| **AI Agent (gemini-flash-lite-latest)** | **0.87** | **0.8628** |

## AI Agent -- Escalation Decision

Accuracy: **0.85** | escalate=True F1: 0.877 (precision 0.9386, recall 0.8231)

## Response Quality (LLM-as-Judge)

Mean judge scores (1-5) across all 200 responses:

- correctness: 4.545
- groundedness: 4.66
- helpfulness: 4.525
- safety: 4.92
- brand_consistency: 4.805
- tone: 4.765

Human agreement (n=50 examples, primary statistic: mean_absolute_disagreement):
- Mean Absolute Disagreement: 0.67
- Spearman r: 0.4363 | Weighted kappa: 0.3854 | Exact match: 0.46 | Within 1 point: 0.8733
- MAD is reported as primary because some dimensions (e.g. safety) have low score variance, making Spearman/kappa statistically unstable there -- see Decision 16.

## Notes

- Baselines are trained on Step 5's heuristic pseudo-labels (measured 74% agreement with true gold labels), not gold labels -- their accuracy partly reflects weak-label quality, not just model capability. See Decision 12.
- AI agent escalation accuracy reflects a diagnosed prompt fix (Decision 14): 76.0% -> 85.0%. Pre-fix results are preserved as data/results/ai_agent_results_v1_baseline.json for the record.
- Response-quality human agreement uses a single self-rater (the developer), not a diverse human pool -- see Decision 16 and docs/response_quality_rubric.md.
