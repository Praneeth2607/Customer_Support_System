# Step 10: Failure Mode Analysis

Five failure modes, selected for diversity across the pipeline (data quality,
intent classification, escalation logic, response generation) rather than
five variations on the same theme. Every example below is real, pulled
directly from `data/results/ai_agent_results.json`, `data/judge/
llm_judge_results.json`, and `data/golden/golden_evaluation_set.json` — none
invented. Where a pattern is systemic, the underlying counts are reported
alongside the illustrative example, not just the anecdote.

---

## 1. Escalated-response phrasing is templated from the prompt's own escalation instructions, not grounded in retrieved evidence

**Real example** — `golden_0142`:
> **Customer**: "Uber driver drove me to wrong location, makes me get out anyway and @115873 won't provide a proper refund"
> **Draft**: *"We're deeply concerned to hear about this experience. Please DM us your account details so our safety team can investigate immediately."*
> None of the three retrieved historical evidence items contain "safety team," "deeply concerned," or any comparable escalation-acknowledgment phrasing — the real historical replies for similar messages are plain "Sign in and visit us at [URL]" / "Please send us a note at [URL]."

**Expected behavior**: Every claim or phrase in the response should be traceable to the retrieved historical evidence (Section 17; the rubric's `groundedness` dimension).

**Actual behavior**: When the agent escalates, its acknowledgment phrasing ("our safety team," "we've escalated this to our support team," "we are escalating this") is consistently synthesized from the system prompt's own instruction — *"If escalating, the reply should acknowledge the issue and say a specialist will follow up"* (added in Decision 14) — not from the retrieved evidence, which almost never contains that framing because real Uber support replies are near-uniformly generic redirects regardless of internal escalation status.

**Quantified**: LLM-judge groundedness ≤2 (out of 200 responses) occurs in **8 cases — all 8 are escalated responses (7.0% of the 114 escalated responses), and 0 of the 86 auto-handled responses**. Mean groundedness is 4.43 for escalated responses vs. 4.97 for auto-handled ones. Other examples: `golden_0063` ("we've escalated this to our support team to look into the double charge right away"), `golden_0148` ("we are escalating this to our support team"), `golden_0168`, `golden_0171`, `golden_0177`.

**Why the failure happened**: Decision 14 fixed escalation *accuracy* by giving the model an explicit checklist and telling it how to phrase an escalation acknowledgment. That instruction was necessary and it worked (76.0% → 85.0% escalation accuracy) — but it also gave the model a second, competing source of phrasing (the prompt's own instruction) that isn't tied to the retrieved evidence at all. The two objectives (escalate correctly; ground every phrase in evidence) were optimized somewhat independently, and this is the resulting seam.

**Hypothesis**: This is a direct, traceable side effect of the Decision 14 fix, not an independent bug — improving one axis (escalation accuracy) measurably cost a small amount on another (groundedness of escalated replies specifically). It generalizes: any instruction that tells the model *how to phrase* something risks becoming a second, ungrounded source of text alongside the real evidence.

**Potential improvement**: Retrieve or require at least one piece of evidence that itself resolves in an escalation-like pattern (e.g., a "we've followed up via DM" or "member of our team will reach out" historical reply, both of which exist in the corpus) preferentially when the agent is escalating, so the acknowledgment phrasing has a real precedent to draw from instead of the prompt's own generic instruction. Alternatively, constrain the escalation-acknowledgment instruction to a fixed, pre-approved sentence (removing the model's freedom to phrase it) rather than describing the *intent* and letting the model generate novel wording.

---

## 2. Retrieved evidence details get copied across customers

**Real example** — `golden_0100`:
> **Customer**: "As if @Uber_Support just took £14 of my money to have the food delivered an HOUR later, lukewarm, missing a part of the order. What a joke."
> **Draft**: *"We definitely want to follow up, **Annabel**. Send us a DM with your account email address and we'll be in touch."*
> The name "Annabel" does not appear anywhere in this customer's message. It was copied from the third retrieved historical evidence item, whose real historical reply happened to address a *different* customer by that name.

**Expected behavior**: The response should never attribute a detail (a name, an account fact, a specific circumstance) from the retrieved evidence to the current customer unless that customer actually provided it (Section 17: never fabricate or misattribute specifics).

**Actual behavior**: The model treated the retrieved evidence's reply text as a stylistic template to reuse, including a personalization detail that belongs to a different historical conversation, not this one.

**Quantified**: This specific error (a copied personal detail) was caught in exactly 1 of the 200 responses on manual + LLM-judge review — genuinely rare — but it is the single lowest-groundedness score in the entire set (1/5, judge and human independently agreed), and it is a categorically different, more concerning kind of error than a generic-but-safe redirect: it produces a response that reads as confidently personalized while being factually wrong about who it's addressing.

**Why the failure happened**: The retrieval step returns full historical reply text as grounding evidence, with no mechanism distinguishing "reusable pattern/phrasing" from "specific-to-that-conversation detail" (a name, a case number, an amount). The generation prompt doesn't explicitly warn against carrying over named entities from evidence.

**Hypothesis**: Low-frequency but high-severity — this is exactly the failure class that erodes trust fastest in a real deployment (a customer seeing someone else's name in their support reply), even though it's rare in aggregate. Frequency alone (0.5% of cases) understates its importance.

**Potential improvement**: Add an explicit prompt rule — "never use a name, account detail, or specific circumstance from the retrieved evidence unless the current customer's own message stated it" — and/or a cheap post-generation regex/NER check that flags any proper name in the draft not present in the customer's own message, for automatic escalation/rejection rather than silent send.

---

## 3. "Redirect reflex": generic redirects substituted for a direct, answerable question

**Real example** — `golden_0075`:
> **Customer**: "If driver asks you in the mid of expressway toll booth to pay for toll as he doesnt have cash, does a customer fight with driver for not paying or end the trip there and book one more cab!! What to do?"
> **Draft**: *"We can definitely take a closer look! Please send us a DM of the email address connected to your account so we can further assist."*
> This is a direct policy question with a knowable answer (the historical ground truth for this exact scenario, visible in Step 5's `golden_0075` hand-labelling, was: "feel free to end the trip and request a new ride"). The draft instead asks for account details to "look into" something that isn't actually account-specific.

**Expected behavior**: When a question is genuinely informational/procedural rather than account-specific, the response should attempt an actual answer if the evidence or documented policy supports one (rubric: `correctness`, `helpfulness`).

**Actual behavior**: The model defaults to the single most common shape in the retrieved evidence (a DM/account-detail redirect) even when the question doesn't require looking at anyone's account.

**Quantified**: Two more examples from the same pattern: `golden_0116` ("is there a pay in cash option for uber eats?" → "We're always here to help! Please visit for more info" — doesn't say yes or no) and `golden_0027` ("THEY CANCELLED MY ORDER" → a reply about donut/promo demand that assumes a cause the message never stated). LLM-judge `helpfulness` ≤2 appears on 4/200 responses; `correctness` ≤2 on 5/200 — a small but real minority concentrated in exactly this pattern.

**Why the failure happened**: Because the vast majority of the retrieved historical evidence (and, by extension, most of Uber's real historical behavior in this dataset) *is* a generic "DM us your account details" redirect regardless of question type, the retrieval-grounded design has no natural signal that distinguishes "this question needs a real answer" from "this needs account lookup." The system was explicitly told to mirror the historical pattern (Section 16/17's grounding requirement) — and it's doing exactly that, even where the historical pattern itself was mediocre customer service.

**Hypothesis**: This is a structural limitation of pure retrieval-grounding on a corpus where the ground-truth "good" answer is often itself a non-answer. Grounding in mediocre historical data caps the system's helpfulness ceiling at roughly the historical support team's own ceiling.

**Potential improvement**: Layer the documented taxonomy policy text (already partially used — see `ESCALATION_POLICIES`' `policy_rationale`) more assertively for the subset of intents/questions that are answerable from policy alone (tolls, cash payment options, cancellation windows), rather than relying solely on retrieved conversation pairs; or add a lightweight "is this a yes/no or how-to question with a documented policy answer?" check before falling back to a generic redirect.

---

## 4. Intent misclassification toward `driver_conduct_and_safety` overstates response severity — even when the escalation call is coincidentally still "correct"

**Real example** — `golden_0049`:
> **Customer**: "you guys actually don't have loyal drivers. i forgot one of my accessory in uber cab. i called back & driver said - I've thrown it out"
> **Gold**: `lost_item` (Decision 11's documented boundary rule: a driver discarding/refusing to return an item escalates, but stays classified as `lost_item`)
> **Predicted**: `driver_conduct_and_safety`
> **Draft**: *"We take reports of **driver misconduct** very seriously. Please send us a direct message with your account details so our **safety team** can investigate this immediately."*

A second example, `golden_0159` ("Boycotting you guys..your drivers are the worst... unless you have a human who can talk to me"), gold intent `out_of_scope_or_unclear` (vague, no specific incident), predicted `driver_conduct_and_safety`, drafted with the same "safety team" framing for what is actually generic venting with no concrete safety incident described.

**Expected behavior**: A lost-item non-cooperation complaint should get a lost-item-appropriate escalation reply (e.g., involving a driver in a property dispute); a vague, non-specific complaint should not be answered as though a safety team is being urgently dispatched.

**Actual behavior**: Both examples produce `should_escalate = True`, which matches the gold label in both cases — so neither the intent-accuracy metric's per-class confusion nor the escalation-accuracy metric flags these as errors. Only the response-quality judge catches it, because the response *text* overstates severity even though the binary escalate/auto-handle decision happens to land correctly.

**Why the failure happened**: `driver_conduct_and_safety` is (by design, per Decision 8's precedence rule) the intent with the strongest, most distinctive keyword signal ("driver," negative sentiment, complaint-about-conduct framing) among the seven intents, making it an easy default for the classifier to reach for whenever a message expresses driver-directed frustration, even when the actual content is about a different underlying issue (a lost item, or no specific incident at all).

**Hypothesis**: This is a genuine blind spot in a metrics-only view of the system: intent accuracy (87.0%) and escalation accuracy (85.0%) both look fine on these two examples, yet the actual customer-facing text is wrong in a way that could itself cause harm (unnecessarily alarming a customer, or diverting a genuine property dispute into the safety queue instead of the right one). This is exactly the kind of thing Section 23 ("what's misleading about my headline number") needs to name explicitly — no single number in this project currently surfaces this failure class.

**Potential improvement**: When the agent's own `intent_confidence` for `driver_conduct_and_safety` is not high, or when the retrieved evidence for a `driver_conduct_and_safety` prediction is weak (as it was in both examples above — see Failure Mode 1's evidence-quality theme), route to a secondary check or a more conservative, generic escalation phrasing rather than the full safety-team framing reserved for clear-cut cases.

---

## 5. Weak heuristic training labels put a hard ceiling on the traditional-ML baseline, concentrated in Eats-vs-ride ambiguity

**Real example**: The Step 5 heuristic pseudo-intent classifier (used to weak-label the ~42k-conversation training pool for both Step 6 baselines) labels `"why was my order randomly cancelled?"` as `cancellation_issue` (matches the "cancel" keyword) — but on hand-labelling, this is `out_of_scope_or_unclear` (Eats), since "order" terminology and no ride-specific content indicate a food-delivery cancellation, not a ride one.

**Expected behavior**: Training labels should reflect the true intent distribution closely enough that a classifier trained on them generalizes to true labels.

**Actual behavior**: Measured heuristic-vs-gold agreement across the 200 hand-labelled golden examples is **74.0% (148/200)** — the pseudo-labels used to train both baselines are wrong roughly 1 in 4 times, overwhelmingly because Eats-flavored messages contain ride-intent keywords ("cancel," "charged," "forgot") by coincidence.

**Quantified**: This directly bounds what Step 6's TF-IDF+LogReg baseline can achieve (71.0% accuracy, 0.7238 macro F1) — its weakest per-intent F1 is on `out_of_scope_or_unclear` itself (0.5631), the exact class the weak labels get wrong most often. The AI agent (Step 7), which does not train on these labels at all, doesn't inherit this ceiling and scores 87.0%/0.8628 on the same test set.

**Why the failure happened**: The heuristic classifier used for weak-label generation is deliberately simple (keyword matching, documented in Step 5's methodology as a sampling aid, not a production classifier) — it has no way to distinguish "Uber" the ride service from "Uber Eats" the food-delivery service beyond a fixed keyword list, and ride-domain and food-domain complaints share a lot of generic vocabulary (cancel, charged, driver, order).

**Hypothesis**: This is a foundational, data-quality-driven failure mode rather than a model-architecture one — no amount of TF-IDF/LogReg hyperparameter tuning fixes a training-label ceiling. It's also the same root ambiguity (Eats vs. ride) that reappears independently in Failure Mode 3-adjacent territory and in several of Step 7's own live intent misses (`golden_0065`, `golden_0070`, `golden_0072` — see the escalation false-negative analysis in Decision 14's follow-up work), making Eats/ride ambiguity the single most recurring root cause across this entire project, from training data through to live inference.

**Potential improvement**: A better weak-label source (e.g., using the already-built AI agent itself, or a stronger keyword/regex classifier — trained once — as the pseudo-labeler for a *new* baseline training pass) would likely lift the traditional-ML baseline's ceiling meaningfully, since the AI agent has already demonstrated it doesn't share this specific confusion. This wasn't done in this project both to avoid circularity (using the system being evaluated to generate another system's training data) and because the traditional-ML baseline's job is to be a *simple*, interpretable reference point — not to be optimized to match the AI agent.

---

## Summary Table

| # | Failure Mode | Scope | Real Examples |
|---|---|---|---|
| 1 | Escalated-response phrasing ungrounded in evidence | 8/200 (7.0% of escalated responses) | golden_0142, golden_0063, golden_0148, golden_0168, golden_0171, golden_0177 |
| 2 | Cross-customer detail leakage (name misattribution) | 1/200, but high severity | golden_0100 |
| 3 | "Redirect reflex" on answerable questions | 4-5/200 low helpfulness/correctness | golden_0075, golden_0116, golden_0027 |
| 4 | Intent → response severity overstatement, invisible to binary metrics | 2+ identified, likely under-counted | golden_0049, golden_0159 |
| 5 | Weak training-label ceiling on baseline (Eats/ride ambiguity) | 26% of all training labels (measured) | Decision 12; golden_0065, golden_0070, golden_0072 |

All five trace back, directly or indirectly, to two root causes that recur throughout this project: **(a) Eats-vs-ride ambiguity**, present from raw data through training labels through live classification, and **(b) the tension between grounding responses in historical evidence and giving the model explicit behavioral instructions** (escalation phrasing, severity calibration), where the instruction wins out over the evidence more often than intended.
