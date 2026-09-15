"""
Step 6: Baseline Intent Classifiers

Two baselines, evaluated on the SAME frozen test set as everything else in
this project (data/golden/golden_evaluation_set.json, 200 hand-labelled
examples) -- never a different sample, per the assignment's explicit rule.

Baseline 1 -- Majority Class: always predicts the single most common
training-set label. The minimum reference point; any real system must beat
this by a wide margin to be worth building.

Baseline 2 -- TF-IDF + Logistic Regression: a standard, fast, interpretable
traditional-ML text classifier.

Training data problem and how we solved it
--------------------------------------------
There is no large hand-labelled training corpus -- only the 200-example
golden set has true gold_intent labels, and it IS the test set (using it to
train would be leakage/circularity, and it's too small to split without
gutting the eval set's carefully-stratified rare-intent coverage).

So both baselines are trained on the Step 5 HEURISTIC pseudo-intent labels
(src/evaluation/sample_golden_set.py's keyword classifier) applied to the
~41.6k-conversation sampling frame, MINUS the 200 conversation_ids reserved
in data/golden/golden_conversation_ids.json (leakage control -- the same
manifest Step 7's retrieval corpus must also respect).

This is weak supervision, and it has a known, MEASURED ceiling: Step 5's
hand-labelling found the heuristic only agrees with true gold labels 74% of
the time (148/200), overwhelmingly because Eats content coincidentally
matches a ride-intent keyword. Training a classifier on these labels means
neither baseline can be expected to exceed roughly that ceiling by intent
category, and this is exactly the kind of thing that belongs in the
"what's misleading about my headline number" report section later -- a
baseline's accuracy against the gold set partly measures how well 41.6k
weak labels approximate reality, not just how good TF-IDF+LogReg is.

Outputs: data/results/baseline_majority.json, baseline_tfidf_logreg.json,
baseline_comparison.json
"""

import json
import os
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)

from src.classification.taxonomy import INTENTS, ESCALATION_POLICIES
from src.classification.leakage_utils import iter_leakage_safe_conversations
from src.evaluation.sample_golden_set import pseudo_label as heuristic_pseudo_label

GOLDEN_SET_PATH = "data/golden/golden_evaluation_set.json"
RESULTS_DIR = "data/results"
RANDOM_SEED = 42


def build_training_set():
    """Heuristic-pseudo-labelled training pool, leakage-safe (see
    src/classification/leakage_utils.py -- excludes by conversation_id AND
    exact/near-duplicate text, per Decision 12)."""
    texts, labels = [], []
    for conv in iter_leakage_safe_conversations():
        pseudo_intent, _difficulty, _matched, _subtype, _wc = heuristic_pseudo_label(conv)
        texts.append(conv["first_customer_query_clean"].strip())
        labels.append(pseudo_intent)

    return texts, labels


def load_golden_set():
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden = json.load(f)
    golden_ids = [c["golden_id"] for c in golden]
    texts = [c["customer_message"] for c in golden]
    gold_labels = [c["gold_intent"] for c in golden]
    return golden_ids, texts, gold_labels


def evaluate(y_true, y_pred, labels=INTENTS):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    per_class = {
        labels[i]: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i in range(len(labels))
    }
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    return {
        "accuracy": round(float(accuracy), 4),
        "macro_f1": round(float(macro_f1), 4),
        "per_class": per_class,
        "confusion_matrix": {"labels": labels, "matrix": cm},
    }


def escalation_from_predicted_intent(predicted_intents):
    """Supplementary metric (not a required Step 6 deliverable): what escalation
    accuracy would you get by naively applying each intent's DEFAULT policy to
    the baseline's predicted intent? Reported for context ahead of Step 7/8."""
    return [ESCALATION_POLICIES[intent]["default_escalate"] for intent in predicted_intents]


def evaluate_escalation(y_true_escalate, y_pred_escalate):
    accuracy = accuracy_score(y_true_escalate, y_pred_escalate)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true_escalate, y_pred_escalate, labels=[True, False], zero_division=0
    )
    return {
        "accuracy": round(float(accuracy), 4),
        "escalate_class_precision": round(float(precision[0]), 4),
        "escalate_class_recall": round(float(recall[0]), 4),
        "escalate_class_f1": round(float(f1[0]), 4),
        "note": (
            "Supplementary only: derived by applying each intent's fixed default "
            "escalation policy to the BASELINE's predicted intent (not a learned "
            "escalation model). A wrong intent prediction here can still yield the "
            "right escalation decision by chance, or vice versa -- do not treat this "
            "as the Step 7/8 escalation evaluation."
        ),
    }


def run_majority_baseline(train_labels, test_texts, test_gold, test_escalate):
    majority_label = Counter(train_labels).most_common(1)[0][0]
    predictions = [majority_label] * len(test_texts)
    metrics = evaluate(test_gold, predictions)
    pred_escalate = escalation_from_predicted_intent(predictions)
    metrics["escalation_supplementary"] = evaluate_escalation(test_escalate, pred_escalate)
    metrics["majority_label"] = majority_label
    metrics["predictions"] = predictions
    return metrics


def run_tfidf_logreg_baseline(train_texts, train_labels, test_texts, test_gold, test_escalate):
    vectorizer = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2), stop_words="english",
        min_df=5, max_df=0.6,
    )
    X_train = vectorizer.fit_transform(train_texts)
    X_test = vectorizer.transform(test_texts)

    clf = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED,
    )
    clf.fit(X_train, train_labels)
    predictions = clf.predict(X_test).tolist()

    metrics = evaluate(test_gold, predictions)
    pred_escalate = escalation_from_predicted_intent(predictions)
    metrics["escalation_supplementary"] = evaluate_escalation(test_escalate, pred_escalate)
    metrics["vocab_size"] = len(vectorizer.get_feature_names_out())
    metrics["predictions"] = predictions
    return metrics


def main():
    print("=" * 70)
    print("STEP 6: BASELINE INTENT CLASSIFIERS")
    print("=" * 70)

    train_texts, train_labels = build_training_set()
    print(f"Training pool (heuristic-labelled, golden set excluded): {len(train_texts):,} examples")
    train_dist = Counter(train_labels)
    print("Training label distribution (heuristic pseudo-labels):")
    for intent in INTENTS:
        print(f"  {intent:30s} {train_dist.get(intent, 0):,}")

    golden_ids, test_texts, test_gold = load_golden_set()
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        golden_full = json.load(f)
    test_escalate = [c["should_escalate"] for c in golden_full]
    print(f"\nTest set (frozen golden set): {len(test_texts)} examples")

    print("\n--- Baseline 1: Majority Class ---")
    majority_metrics = run_majority_baseline(train_labels, test_texts, test_gold, test_escalate)
    print(f"Majority label: {majority_metrics['majority_label']}")
    print(f"Accuracy: {majority_metrics['accuracy']}  Macro F1: {majority_metrics['macro_f1']}")

    print("\n--- Baseline 2: TF-IDF + Logistic Regression ---")
    tfidf_metrics = run_tfidf_logreg_baseline(train_texts, train_labels, test_texts, test_gold, test_escalate)
    print(f"Vocab size: {tfidf_metrics['vocab_size']}")
    print(f"Accuracy: {tfidf_metrics['accuracy']}  Macro F1: {tfidf_metrics['macro_f1']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    def with_predictions_by_id(metrics):
        out = dict(metrics)
        out["predictions_by_golden_id"] = dict(zip(golden_ids, metrics["predictions"]))
        del out["predictions"]
        return out

    majority_out = with_predictions_by_id(majority_metrics)
    tfidf_out = with_predictions_by_id(tfidf_metrics)

    with open(f"{RESULTS_DIR}/baseline_majority.json", "w", encoding="utf-8") as f:
        json.dump(majority_out, f, indent=2)
    with open(f"{RESULTS_DIR}/baseline_tfidf_logreg.json", "w", encoding="utf-8") as f:
        json.dump(tfidf_out, f, indent=2)

    comparison = {
        "test_set": GOLDEN_SET_PATH,
        "test_set_size": len(test_texts),
        "training_pool_size": len(train_texts),
        "training_labels_source": (
            "Heuristic pseudo-intent classifier (src/evaluation/sample_golden_set.py), "
            "NOT gold labels -- see module docstring for the known ~74% agreement ceiling."
        ),
        "baseline_majority_class": {
            "accuracy": majority_metrics["accuracy"],
            "macro_f1": majority_metrics["macro_f1"],
        },
        "baseline_tfidf_logreg": {
            "accuracy": tfidf_metrics["accuracy"],
            "macro_f1": tfidf_metrics["macro_f1"],
        },
    }
    with open(f"{RESULTS_DIR}/baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    print(f"\nWrote results to {RESULTS_DIR}/")
    print("\nPer-intent macro comparison (F1):")
    print(f"{'intent':30s} {'majority':>10s} {'tfidf+lr':>10s}")
    for intent in INTENTS:
        m = majority_metrics["per_class"][intent]["f1"]
        t = tfidf_metrics["per_class"][intent]["f1"]
        print(f"{intent:30s} {m:>10.4f} {t:>10.4f}")


if __name__ == "__main__":
    main()
