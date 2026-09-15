"""
Taxonomy Definitions & Escalation Policy Schema for Uber Customer Support

Defines:
- INTENTS: List of 6 canonical intent strings
- INTENT_DEFINITIONS: Human-readable definitions, keywords, and positive examples
- ESCALATION_POLICIES: Default escalation rule and risk category per intent
- INTENT_PRECEDENCE: Hierarchical precedence order for multi-intent resolution
"""

INTENTS = [
    "cancellation_issue",
    "fare_and_payment_dispute",
    "lost_item",
    "driver_conduct_and_safety",
    "pickup_and_route_issue",
    "account_and_promo_issue",
    "out_of_scope_or_unclear"
]

INTENT_DEFINITIONS = {
    "cancellation_issue": {
        "name": "Cancellation Issue",
        "definition": "Inquiries or disputes regarding ride cancellations initiated by either rider or driver, and disputes over cancellation fees.",
        "keywords": ["cancel", "cancelled", "cancellation fee", "driver cancelled", "charged 5", "no-show fee"],
        "examples": [
            "Driver cancelled the trip after making me wait 15 minutes and I was charged $5.",
            "Why was I charged a cancellation fee when the driver was driving in the opposite direction?",
            "Driver asked me to cancel so he wouldn't get penalized, now I got billed.",
            "How do I cancel my pending ride without incurring a penalty?"
        ]
    },
    "fare_and_payment_dispute": {
        "name": "Fare & Payment Dispute",
        "definition": "Inquiries or disputes regarding completed ride charges, unexpected or duplicate billing, toll charges, surge pricing discrepancies, cleaning fees, or receipt requests.",
        "keywords": ["charged", "overcharged", "double charge", "fare", "receipt", "refund", "toll", "surge", "cleaning fee"],
        "examples": [
            "My bank account was charged twice for the exact same trip home.",
            "The upfront fare estimate showed $18, but my final receipt came out to $38.50.",
            "I was hit with an unauthorized $50 cleaning fee for a mess I didn't make.",
            "I need an itemized tax receipt sent to my business email for my last trip."
        ]
    },
    "lost_item": {
        "name": "Lost Item",
        "definition": "Urgent inquiries or reports regarding personal property left behind in an Uber vehicle (e.g., phones, wallets, keys, luggage, bags, glasses).",
        "keywords": ["left my phone", "lost phone", "left keys", "lost wallet", "left bag", "back seat", "forgot item"],
        "examples": [
            "I left my iPhone in the back seat of my Uber 20 minutes ago. How can I contact the driver?",
            "Left my house keys and wallet in a black Camry going to the airport.",
            "My friend forgot her purse in the car and her phone was inside it so she can't log in.",
            "Driver isn't picking up the phone to return my laptop."
        ]
    },
    "driver_conduct_and_safety": {
        "name": "Driver Conduct & Safety",
        "definition": "Severe reports concerning driver misconduct, unprofessionalism, verbal abuse, discriminatory remarks, dangerous or reckless driving, vehicle safety hazards, harassment, or physical danger.",
        "keywords": ["rude", "unsafe", "reckless", "speeding", "yelled", "harassment", "scary", "dangerous", "swerving", "drunk", "threatened"],
        "examples": [
            "My driver was texting while speeding on the highway and almost collided with a truck.",
            "The driver started yelling profanities at me when I asked him to turn down the heat.",
            "Driver seemed intoxicated and was swerving between lanes.",
            "Driver refused to let me exit the vehicle at my stop."
        ]
    },
    "pickup_and_route_issue": {
        "name": "Pickup & Route Issue",
        "definition": "Operational problems during pickup or transit, including driver arriving at the wrong location, refusing to travel to destination, taking an excessive detour, or failing to move toward pickup.",
        "keywords": ["wrong pickup", "detour", "long way", "wrong route", "refused to go", "car not moving", "wrong address"],
        "examples": [
            "Driver picked me up at the wrong terminal and took a 10-mile detour around traffic.",
            "Driver refused to take me to Brooklyn after seeing my destination.",
            "The app showed the driver 2 mins away but the car never moved for 20 minutes.",
            "Driver dropped me off 4 blocks away from my requested destination in the rain."
        ]
    },
    "account_and_promo_issue": {
        "name": "Account & Promo Issue",
        "definition": "Inquiries regarding user account management, login / two-factor verification failures, promo codes or discount coupons failing to apply, Ride Pass subscriptions, or app technical bugs.",
        "keywords": ["promo code", "discount", "promo not working", "login", "account locked", "verification code", "ride pass", "update phone number"],
        "examples": [
            "My 20% off promo code failed to apply to my ride yesterday.",
            "I cannot log in because verification codes are sent to my old disconnected phone number.",
            "Why was my rider account suddenly deactivated?",
            "I paid for a Ride Pass but it is not showing up as active in my app."
        ]
    },
    "out_of_scope_or_unclear": {
        "name": "Out of Scope / Unclear",
        "definition": "Messages that are not ride-support requests our taxonomy covers, or that lack enough information to assign any of the other five intents. Covers two distinct sub-cases: (a) Uber Eats / food-delivery content, a different Uber business line with its own policies that this ride-support taxonomy does not model, and (b) generic, low-signal messages -- bare venting, a lone '@Uber_Support' mention, or a one-line 'please help me' with no stated problem.",
        "keywords": ["uber eats", "order food", "delivery", "ubereats", "need help", "worst customer service", "please help"],
        "examples": [
            "Y'all do they have Uber Eats in Tallahassee?",
            "Can I order some cereal @115877",
            "what do we do? Help!",
            "what's up with your customer service?"
        ]
    }
}

ESCALATION_POLICIES = {
    "cancellation_issue": {
        "action": "conditional_auto_handle",
        "default_escalate": False,
        "policy_rationale": "Automated guidance on the 5-minute fee waiver window and in-app dispute workflow is safe. Escalate if manual exception or dispute failure is reported."
    },
    "fare_and_payment_dispute": {
        "action": "conditional_auto_handle",
        "default_escalate": False,
        "policy_rationale": "Provide trip fare review workflow link. Escalate for disputed amounts > $50, suspected card fraud, or cleaning fee arbitration."
    },
    "lost_item": {
        "action": "auto_handle",
        "default_escalate": False,
        "policy_rationale": "Highly suitable for automation. Provide standard self-serve driver contact flow via help.uber.com and explain the $15 return fee policy."
    },
    "driver_conduct_and_safety": {
        "action": "mandatory_escalate",
        "default_escalate": True,
        "policy_rationale": "High-risk safety and liability. Must never be auto-handled. Must immediately route to specialized human safety team."
    },
    "pickup_and_route_issue": {
        "action": "conditional_auto_handle",
        "default_escalate": False,
        "policy_rationale": "Provide route adjustment review link. Escalate if passenger was abandoned in an unsafe location."
    },
    "account_and_promo_issue": {
        "action": "conditional_auto_handle",
        "default_escalate": False,
        "policy_rationale": "Provide promo eligibility rules and password/2FA recovery links. Escalate for account deactivations or security breaches."
    },
    "out_of_scope_or_unclear": {
        "action": "mandatory_escalate",
        "default_escalate": True,
        "policy_rationale": "No ride-support policy applies (Uber Eats) or there is not enough information to act (bare mention / no stated problem). Route to a human rather than guess at an action; a human can redirect Eats traffic or ask the customer a clarifying question."
    }
}

# Multi-intent precedence hierarchy (Index 0 = highest priority)
INTENT_PRECEDENCE = [
    "driver_conduct_and_safety",    # Safety always supersedes all other concerns
    "lost_item",                    # Urgent physical possession
    "cancellation_issue",           # Direct transactional cancellation dispute
    "fare_and_payment_dispute",     # General pricing / billing
    "pickup_and_route_issue",       # Navigation / route
    "account_and_promo_issue",      # Account settings / promos
    "out_of_scope_or_unclear"       # Lowest priority: only used when nothing else applies
]

def resolve_multi_intent(detected_intents: list[str]) -> str:
    """Resolves conflicting intents using the strict precedence hierarchy."""
    for intent in INTENT_PRECEDENCE:
        if intent in detected_intents:
            return intent
    return detected_intents[0] if detected_intents else "out_of_scope_or_unclear"
