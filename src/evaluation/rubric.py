"""
Step 9: Response-Quality Judge Rubric (v1.0)

Six dimensions, 1-5 scale each, per Section 20 of the assignment. This is
the single source of truth for both the LLM judge's prompt and the human
rating pass -- both must score against the exact same criteria, or an
agreement comparison between them is meaningless.

Version-controlled: any future change to these definitions is a new
version (bump RUBRIC_VERSION), not a silent edit -- score comparisons
across rubric versions are not valid (see docs/DECISION_LOG.md).
"""

RUBRIC_VERSION = "1.0"

DIMENSIONS = {
    "correctness": {
        "question": "Does the response correctly address the customer's issue?",
        "scale": {
            5: "Directly and accurately addresses the specific issue raised, no factual errors.",
            4: "Addresses the issue correctly but is slightly generic or misses a minor nuance.",
            3: "Partially addresses the issue; somewhat relevant but leaves out something important.",
            2: "Barely relevant to the actual issue, or contains a minor factual/policy error.",
            1: "Wrong or irrelevant to the issue, or contains a significant factual/policy error.",
        },
    },
    "groundedness": {
        "question": "Is the response supported by the retrieved historical evidence?",
        "scale": {
            5: "Every claim/action is directly supported by the retrieved evidence, or is a safe "
               "generic redirect matching the evidence's own pattern.",
            4: "Mostly grounded; one minor unsupported flourish that doesn't materially change the claim.",
            3: "Loosely grounded; generic but plausible given the evidence, no direct support for specifics.",
            2: "Contains a claim or detail not supported by any retrieved evidence (mild fabrication risk).",
            1: "Clearly fabricates specific facts, policies, or promises the evidence does not support.",
        },
    },
    "helpfulness": {
        "question": "Does it give the customer a useful next step?",
        "scale": {
            5: "Clear, actionable next step the customer can immediately follow.",
            4: "Useful next step but slightly vague on how to proceed.",
            3: "Acknowledges the issue but the next step is generic/non-specific.",
            2: "Minimal practical guidance, mostly just an apology.",
            1: "No actionable guidance at all.",
        },
    },
    "safety": {
        "question": "Does it avoid unsupported or risky claims?",
        "scale": {
            5: "No risky claims; correctly avoids promising refunds/actions/timelines; escalates when needed.",
            4: "Minor risk (e.g. a soft assurance) but nothing materially unsafe.",
            3: "Ambiguous phrasing that could be misread as a promise or guarantee.",
            2: "Makes an unsupported promise or claims an action was already completed.",
            1: "Makes a clearly unsafe claim (promises a specific refund/amount, claims to have "
               "resolved a safety incident itself, or fails to escalate an actual safety issue).",
        },
    },
    "brand_consistency": {
        "question": "Does it align with Uber's historical support tone and behavior?",
        "scale": {
            5: "Reads exactly like a real Uber support reply in style, brevity, and structure.",
            4: "Close match, minor stylistic deviation.",
            3: "Generic customer-service tone, not distinctly \"Uber.\"",
            2: "Somewhat off-brand (too casual, too formal, or oddly worded).",
            1: "Clearly inconsistent with the brand voice.",
        },
    },
    "tone": {
        "question": "Is the response appropriate for customer support?",
        "scale": {
            5: "Empathetic, professional, appropriately calibrated to the situation's severity.",
            4: "Generally appropriate, slightly flat or generic.",
            3: "Neutral/acceptable but could be warmer or more tailored.",
            2: "Somewhat inappropriate (too curt, or over-familiar) for the situation.",
            1: "Inappropriate tone (dismissive, cold given severity, or inappropriately casual "
               "for a safety issue).",
        },
    },
}

DIMENSION_ORDER = ["correctness", "groundedness", "helpfulness", "safety",
                   "brand_consistency", "tone"]


def render_rubric_text():
    """Render the rubric as prompt-ready text -- used by both the LLM judge
    prompt and the human-rating review dump, so both see identical wording."""
    lines = [f"RESPONSE QUALITY RUBRIC (v{RUBRIC_VERSION}) -- score each dimension 1-5:"]
    for dim in DIMENSION_ORDER:
        d = DIMENSIONS[dim]
        lines.append(f"\n{dim.upper()}: {d['question']}")
        for score in (5, 4, 3, 2, 1):
            lines.append(f"  {score} = {d['scale'][score]}")
    return "\n".join(lines)
