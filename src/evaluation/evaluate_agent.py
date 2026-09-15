"""
Step 7 (evaluation half): score the AI agent's results against the frozen
golden set, using the SAME metric functions as Step 6's baselines
(src/classification/baselines.py:evaluate / evaluate_escalation) so all
three systems -- majority-class, TF-IDF+LogReg, AI agent -- are compared on
identical methodology, not just the same test set.

Unlike the baselines' "escalation_supplementary" metric (which derives an
escalation guess from the predicted intent's fixed default policy), the AI
agent's should_escalate came directly from the model's own reasoning over
the retrieved evidence -- so this is the real Step 7/8 escalation
evaluation, not a supplementary approximation.

Input: data/results/ai_agent_results.json (from src/pipeline/support_agent.py)
Output: data/results/ai_agent_metrics.json, data/results/system_comparison.json
"""

import json

from src.classification.baselines import evaluate, evaluate_escalation
from src.classification.taxonomy import INTENTS

RESULTS_PATH = "data/results/ai_agent_results.json"
BASELINE_COMPARISON_PATH = "data/results/baseline_comparison.json"
METRICS_OUTPUT_PATH = "data/results/ai_agent_metrics.json"
COMPARISON_OUTPUT_PATH = "data/results/system_comparison.json"


def main():
    print("=" * 70)
    print("STEP 7: AI SUPPORT AGENT -- EVALUATION")
    print("=" * 70)

    with open(RESULTS_PATH, "r", encoding="utf-8") as f:
        results = json.load(f)
    print(f"Scoring {len(results)} AI agent results against gold labels")

    y_true = [r["gold_intent"] for r in results]
    y_pred = [r["predicted_intent"] for r in results]
    esc_true = [r["gold_should_escalate"] for r in results]
    esc_pred = [r["predicted_should_escalate"] for r in results]

    intent_metrics = evaluate(y_true, y_pred, labels=INTENTS)
    escalation_metrics = evaluate_escalation(esc_true, esc_pred)
    escalation_metrics["note"] = (
        "This IS the primary Step 7/8 escalation evaluation -- should_escalate "
        "came directly from the agent's own reasoning over retrieved evidence, "
        "not a derived default-policy lookup (contrast with the baselines' "
        "escalation_supplementary metric, which is only a rough approximation)."
    )

    heuristic_agreement = sum(
        1 for r in results if r["predicted_intent"] == r.get("gold_intent")
    )

    metrics = {
        "test_set": "data/golden/golden_evaluation_set.json",
        "test_set_size": len(results),
        "model": "gemini-flash-lite-latest",
        "intent_classification": intent_metrics,
        "escalation_decision": escalation_metrics,
    }
    with open(METRICS_OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nIntent Accuracy: {intent_metrics['accuracy']}")
    print(f"Intent Macro F1: {intent_metrics['macro_f1']}")
    print(f"\nEscalation Accuracy: {escalation_metrics['accuracy']}")
    print(f"Escalation F1 (escalate=True class): {escalation_metrics['escalate_class_f1']}")

    print("\nPer-intent F1:")
    for intent in INTENTS:
        pc = intent_metrics["per_class"][intent]
        print(f"  {intent:30s} P={pc['precision']:.3f} R={pc['recall']:.3f} "
              f"F1={pc['f1']:.3f} support={pc['support']}")

    # Combine with Step 6's baseline results for a single three-way comparison.
    try:
        with open(BASELINE_COMPARISON_PATH, "r", encoding="utf-8") as f:
            baseline_comparison = json.load(f)
        comparison = {
            "test_set": metrics["test_set"],
            "test_set_size": metrics["test_set_size"],
            "majority_class_baseline": baseline_comparison["baseline_majority_class"],
            "tfidf_logreg_baseline": baseline_comparison["baseline_tfidf_logreg"],
            "ai_agent": {
                "accuracy": intent_metrics["accuracy"],
                "macro_f1": intent_metrics["macro_f1"],
                "escalation_accuracy": escalation_metrics["accuracy"],
            },
        }
        with open(COMPARISON_OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print("\n" + "=" * 70)
        print("THREE-WAY COMPARISON (intent classification, same test set)")
        print("=" * 70)
        print(f"{'System':30s} {'Accuracy':>10s} {'Macro F1':>10s}")
        print(f"{'Majority class':30s} {comparison['majority_class_baseline']['accuracy']:>10} "
              f"{comparison['majority_class_baseline']['macro_f1']:>10}")
        print(f"{'TF-IDF + LogReg':30s} {comparison['tfidf_logreg_baseline']['accuracy']:>10} "
              f"{comparison['tfidf_logreg_baseline']['macro_f1']:>10}")
        print(f"{'AI Agent (Gemini)':30s} {comparison['ai_agent']['accuracy']:>10} "
              f"{comparison['ai_agent']['macro_f1']:>10}")
        print(f"\nWrote {COMPARISON_OUTPUT_PATH}")
    except FileNotFoundError:
        print(f"\n(Skipping three-way comparison -- {BASELINE_COMPARISON_PATH} not found. "
              f"Run src/classification/baselines.py first.)")

    print(f"Wrote {METRICS_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
