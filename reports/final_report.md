# Final Report — Hiver AI Customer Support System (Uber)

*Six-page-equivalent report. Full detail, numbers, and reasoning behind every claim here live in `README.md` (per-step findings) and `docs/DECISION_LOG.md` (17 numbered engineering decisions) — this document is the synthesis, not a duplicate.*

---

## 1. Problem Framing

The assignment's core test: turn a messy, real-world support dataset into a working AI system and *prove* it works — "the proof is worth more than the system." We selected **Uber** (ride-hailing) from the Customer Support on Twitter dataset over AmazonHelp, AppleSupport, SpotifyCares, and Delta after an empirical comparison (Decision 3): Uber's domain decomposes into a small number of cleanly separable, high-stakes-to-low-stakes intents (a safety incident and a promo-code question are both "Uber support" but demand completely different handling), its English-first, low-boilerplate corpus avoided the multilingual fragmentation (AmazonHelp) and DM-deflection wall (AppleSupport, 52%) that would have obscured real conversational content, and 56,270 outbound tweets kept the whole pipeline tractable within the 15-minute reproducibility target.

The resulting system is a **triage-and-draft pipeline**, not an autonomous resolver: given a customer message, it (1) classifies intent against a 7-category taxonomy discovered from the data itself, (2) retrieves the most similar historical Uber conversations as grounding evidence, (3) decides AUTO-HANDLE or ESCALATE with a stated reason, and (4) drafts a reply grounded in that retrieved evidence — never in the model's own general knowledge of what "Uber probably does." That last constraint (Section 17 of the assignment) shaped nearly every architectural choice below.

## 2. What "Good" Means for Uber Support

We operationalized "good" as a taxonomy-driven policy, not a single quality score (`src/classification/taxonomy.py`, `docs/intent_taxonomy.md`):

- **Safety is non-negotiable.** `driver_conduct_and_safety` carries a hard-coded, code-level mandatory-escalate override (never trust a prompt alone for a safety-critical policy) — verified in every one of 200 live runs, the override never even had to fire because the model itself escalated all 21 true safety cases it *correctly identified*. But see §7: this isn't the whole safety story.
- **Auto-handling means giving the same short, standard redirect real Uber support actually gives** — not "the AI personally resolves it." Most of Uber's real historical replies in this dataset are generic DM/help-link redirects (63% embed a help URL, 37.6% defer to DM); the system is designed to match that pattern faithfully rather than invent more specific-sounding help than the brand itself typically provides.
- **Escalation is reserved for named, specific triggers** — a large/unauthorized charge, a repeated pattern, a security concern, an explicit human request, a blocked self-service path — not vague "this needs a human" intuition. This distinction, made explicit only after a live failure diagnosis (Decision 14), was the single highest-leverage fix in the whole project (see §5).
- **Grounding is mandatory, not aspirational.** Every claim in a drafted reply should trace to retrieved historical evidence. We measure this directly via an LLM-judge `groundedness` dimension, not just assume it.
- **A brand that mostly gives generic redirects sets the honest ceiling for "helpful."** We did not try to make the AI more helpful than the real historical support team it's grounded in — that would mean fabricating specificity the brand itself doesn't actually provide (see Failure Mode 3, §6).

## 3. What We Chose Not to Build

Per the assignment's explicit scope guidance and our own judgment calls, out of scope: real Twitter/production integration, live account or trip lookups, a production dashboard, fine-tuning any model, multi-agent architectures, and a second, independently-resourced human annotator pool (Decisions throughout; see the honesty note in `docs/response_quality_rubric.md`). We also chose *not* to:

- **Build a dense-embedding retrieval index.** TF-IDF cosine similarity over the leakage-safe corpus is cheap, fully local, explainable, and sufficient at this corpus size (42k conversations) — a vector database would have added infrastructure without evidence it was the bottleneck.
- **Fine-tune anything.** The AI agent is a single structured LLM call (Gemini `gemini-flash-lite-latest`, free tier) per message, orchestrated with retrieval and a versioned prompt — the assignment explicitly discourages heavy infrastructure, and prompt-level fixes (Decision 14) already produced large, measurable gains.
- **Use the paid Anthropic API**, despite this session running inside Claude Code — an explicit cost decision by the project owner (Decision 13), documented rather than silently substituted.
- **Iterate the escalation prompt repeatedly against the golden set.** One diagnosed revision, one re-run, both versions kept for the record (Decision 14) — repeated tuning against a frozen 200-example test set would blur into overfitting the eval rather than improving the system (Section 29 of the assignment).

## 4. Results vs. Baselines

All three systems evaluated on the identical frozen 200-example golden set (`data/golden/golden_evaluation_set.json`), reproducible via `python evaluate.py`:

| System | Accuracy | Macro F1 |
|---|---|---|
| Majority class (always `out_of_scope_or_unclear`) | 29.0% | 0.0642 |
| TF-IDF + Logistic Regression | 71.0% | 0.7238 |
| **AI Agent (Gemini)** | **87.0%** | **0.8628** |

The majority baseline is the floor: it scores 0 F1 on 6 of 7 intents (never predicts them), which is exactly why macro F1, not accuracy, is the headline metric (Section 14 of the assignment) — a system that ignores minority classes should not look good. The TF-IDF+LogReg baseline is trained on **weak labels** (Step 5's heuristic pseudo-intent classifier, measured only 74.0% agreement with true gold labels — Decision 12), which caps its ceiling; its weakest class is exactly `out_of_scope_or_unclear` (F1 0.5631), the class the weak labels get wrong most often. The AI agent does not train on those labels at all and clears the baseline's ceiling by a wide margin.

**Escalation decision** (a genuinely learned decision, not a lookup — the only one of the three systems that has one): **85.0% accuracy**, up from an initial 76.0% after a single diagnosed prompt fix that cut over-escalation 68% and under-escalation 12% simultaneously (Decision 14 — full diagnosis in §6).

**Response quality** (LLM-as-judge, 6 rubric dimensions, 1–5 scale, `src/evaluation/rubric.py` v1.0, calibrated against 50 human-rated examples): mean scores correctness 4.54, groundedness 4.66, helpfulness 4.53, safety 4.92, brand_consistency 4.80, tone 4.76. Human/judge agreement: Mean Absolute Disagreement 0.67 (chosen as the primary statistic over Spearman/kappa specifically because some dimensions have too little score variance for correlation-based stats to be meaningful — Decision 16), 87.3% of paired scores within 1 point.

## 5. Top Failure Modes

Full write-up with every example, root cause, and proposed fix in `reports/failure_analysis.md`. Summary:

1. **Escalated-response phrasing ungrounded in evidence** (7.0% of escalated responses vs. 0% of auto-handled ones) — a direct, measured side effect of the Decision 14 fix: teaching the model *how* to phrase an escalation gave it a second, ungrounded source of text.
2. **Cross-customer detail leakage** — one draft response copied a customer's name from retrieved evidence into a reply for a *different* customer. Rare (1/200) but the single most severe individual error found.
3. **"Redirect reflex"** — generic redirects substituted for directly-answerable policy questions, inheriting the real historical support team's own tendency toward non-answers.
4. **Intent misclassification toward `driver_conduct_and_safety` overstates response severity, invisible to both headline metrics** — two cases have wrong intent *and* an inappropriately severe response, yet `should_escalate` still (coincidentally) matches gold in both. Neither the 87.0% intent-accuracy nor the 85.0% escalation-accuracy number flags this; only the response-quality judge did.
5. **Weak heuristic training-label noise caps the Step 6 baseline** — the same Eats-vs-ride ambiguity recurs from training data through live inference, the single most repeated root cause in the project.

## 6. What Is Misleading About My Headline Number?

**87.0% intent accuracy hides a safety-critical miss.** `driver_conduct_and_safety` has the lowest per-intent precision (0.750) and one of the lower recalls (0.857) of all 7 intents. Of 21 true safety cases, 18 were correctly identified as safety and escalated by the model's own reasoning; a further 2 were misclassified but *still escalated* by coincidence (their fallback intent happened to also trigger escalation). But **one true safety case — a message explicitly stating "Felt unsafe to give [my destination] over phone"** (`golden_0136`) — was misclassified as a routine `cancellation_issue` and **not escalated at all**. That is a 1-in-200 failure rate in aggregate, invisible inside an 87% headline number, but it is exactly the failure a real deployment cannot afford, and it is the single most important finding in this report.

**85.0% escalation accuracy came from a golden set that is not representative of real escalation volume.** The golden set was deliberately stratified to guarantee statistical power on rare intents (Decision 10), which means 65% of it is labelled `should_escalate=True` — far higher than real inbound traffic would produce, since real traffic skews toward the routine fare/cancellation disputes that make up the bulk of Step 3's conversation volume. The 85.0% figure describes performance on a benchmark deliberately enriched with hard, escalation-heavy cases; it should not be read as "85% of real Uber support traffic gets the right auto-handle/escalate call."

**71.0% and 0.7238 for the TF-IDF baseline partly measures label quality, not model quality.** Its ceiling is bounded by the 74.0% heuristic/gold agreement of the weak labels it trained on (Decision 12) — a stronger classifier trained on the same weak labels would not clear that ceiling by much.

**Groundedness scores 4.66/5 on average, but this average conceals a real bimodal pattern.** Groundedness is 4.97 for auto-handled responses and 4.43 for escalated ones — the aggregate number flattens a systematic gap that's fully explained by Failure Mode 1 (§5), not noise.

**The LLM-judge/human agreement number (MAD 0.67, exact match 46%) reflects one self-rater, not inter-rater reliability.** As with Step 5's hand-labelling, the "human" calibration rater is the developer who built the system — documented plainly in `docs/response_quality_rubric.md` — which risks correlated bias with the system's own design assumptions in a way a genuinely independent annotator pool would not.

## 7. What We Would Do With One More Week

In priority order, each targeting a specific measured gap above rather than a general "make it better":

1. **Fix the `golden_0136`-class safety miss first.** Add an explicit rule: any message containing safety-adjacent language ("unsafe," "scared," "threatened," "followed") triggers a mandatory second-pass safety check regardless of the primary intent classification — a cheap, high-leverage guardrail given the failure mode is rare but maximally severe.
2. **Add a post-generation groundedness guard**: a lightweight check (regex/NER for proper names not in the customer's own message, or a second LLM call specifically asking "is every claim in this draft traceable to the evidence below?") to catch Failure Modes 1 and 2 before a response ships, rather than only measuring them after the fact.
3. **Expand the golden set to include a real-world-proportional slice** alongside the current rare-intent-stratified one, so escalation accuracy and intent accuracy can be reported both ways — "performance on hard cases" and "expected performance on real traffic" — rather than one number doing both jobs.
4. **Recruit at least 2-3 independent human raters** for the Step 9 calibration subset, and report inter-rater agreement among them as a ceiling for how well any judge could possibly agree with "the human."
5. **Try a stronger weak-label source for the Step 6 baseline** (e.g. a fixed snapshot of the AI agent's own high-confidence predictions, held separate from the evaluation set) to see how much of the baseline's ceiling was model capacity vs. label noise — directly testable, not just hypothesized.
6. **Address the "redirect reflex"** by layering the taxonomy's own documented policy text more assertively for the subset of questions that have a real, policy-derivable answer (tolls, cash payment options), rather than defaulting to the historical corpus's own frequent non-answers.

---

*All numbers in this report are pulled directly from committed artifacts (`data/results/`, `data/judge/`, `data/golden/`) and are reproducible via `python evaluate.py`; the safety-miss and groundedness-gap findings in §6 are additionally locked in as regression tests alongside `reports/failure_analysis.md`'s claims.*
