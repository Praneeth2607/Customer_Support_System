# Empirical Intent Taxonomy & Escalation Policy

This document defines the official 6-intent taxonomy discovered from the Uber customer support dataset (`data/processed/uber_conversations.json`).

The taxonomy was derived through:
1. TF-IDF unigram and bigram frequency mining across 42k+ customer inquiries.
2. Unsupervised K-Means clustering (documented in [`data/audit/intent_discovery_report.json`](file:///c:/Users/study/OneDrive/Desktop/Projects/Customer_Support_System/data/audit/intent_discovery_report.json)).
3. Qualitative evaluation of operational actionability and escalation risk boundaries.

---

## 🎯 The 6 Core Intents

```text
                                  Customer Inquiry
                                         │
        ┌────────────────────────────────┼────────────────────────────────┐
        │                                │                                │
 1. cancellation_issue        2. fare_and_payment_dispute         3. lost_item
 (Cancellation fees,          (Overcharges, double billing,       (Left phone, keys,
  driver no-show cancels)      tolls, surge pricing)               wallet in vehicle)
        │                                │                                │
        ├────────────────────────────────┼────────────────────────────────┤
        │                                │                                │
 4. driver_conduct_and_safety 5. pickup_and_route_issue   6. account_and_promo_issue
 (Reckless driving, abuse,    (Wrong pickup location,     (Login lock, promo codes,
  safety violations)           detours, inefficient route) ride pass, app bugs)
```

---

### 1. `cancellation_issue`

* **Definition**: Inquiries or disputes regarding ride cancellations initiated by either rider or driver, and disputes over cancellation fees.
* **Empirical Indicators**: `cancel`, `cancelled`, `cancellation fee`, `driver cancelled`, `charged 5`, `no-show`.
* **Positive Examples**:
  1. *"Driver cancelled the trip after making me wait 15 minutes and I was charged $5."*
  2. *"Why was I charged a cancellation fee when the driver was driving in the opposite direction?"*
  3. *"Driver asked me to cancel so he wouldn't get penalized, now I got billed."*
  4. *"How do I cancel my pending ride without incurring a penalty?"*
* **Boundary Cases**:
  * *Vs. `fare_and_payment_dispute`*: If a charge is specifically identified as a *cancellation fee*, it MUST be classified as `cancellation_issue`. General fare disputes belong to `fare_and_payment_dispute`.
  * *Vs. `driver_conduct_and_safety`*: If the driver cancelled and was verbally abusive, map to `driver_conduct_and_safety` (safety takes precedence).
* **Escalation Policy**:
  * **Safe to Auto-Handle**: Explain the 5-minute cancellation window policy and provide the direct in-app link / URL to request an automated fee waiver.
  * **Escalate**: Customer already requested an automated waiver and was rejected, or disputes repetitive unauthorized driver cancellations.

---

### 2. `fare_and_payment_dispute`

* **Definition**: Inquiries or disputes regarding completed ride charges, unexpected or duplicate billing, toll charges, surge pricing discrepancies, cleaning fees, or receipt requests.
* **Empirical Indicators**: `charged`, `overcharged`, `double charge`, `fare`, `receipt`, `refund`, `toll`, `surge`, `cleaning fee`.
* **Positive Examples**:
  1. *"My bank account was charged twice for the exact same trip home."*
  2. *"The upfront fare estimate showed $18, but my final receipt came out to $38.50."*
  3. *"I was hit with an unauthorized $50 cleaning fee for a mess I didn't make."*
  4. *"I need an itemized tax receipt sent to my business email for my last trip."*
* **Boundary Cases**:
  * *Vs. `cancellation_issue`*: Cancellation fees belong to `cancellation_issue`.
  * *Vs. `account_and_promo_issue`*: If a discount or promo failed to apply, map to `account_and_promo_issue`.
* **Escalation Policy**:
  * **Safe to Auto-Handle**: Provide self-serve link to the trip receipt and the Fare Review workflow in the Uber app.
  * **Escalate**: Suspected fraudulent card usage, large disputed charges (> $50), or cleaning fee disputes requiring photo evidence verification.

---

### 3. `lost_item`

* **Definition**: Urgent inquiries or reports regarding personal property left behind in an Uber vehicle (e.g., phones, wallets, keys, luggage, bags, glasses).
* **Empirical Indicators**: `left my phone`, `lost phone`, `left keys`, `lost wallet`, `left bag`, `back seat`, `forgot item`, `contact driver`.
* **Positive Examples**:
  1. *"I left my iPhone in the back seat of my Uber 20 minutes ago. How can I contact the driver?"*
  2. *"Left my house keys and wallet in a black Camry going to the airport."*
  3. *"My friend forgot her purse in the car and her phone was inside it so she can't log in."*
  4. *"Driver isn't picking up the phone to return my laptop."*
* **Boundary Cases**:
  * *Vs. `driver_conduct_and_safety`*: If the customer explicitly accuses the driver of stealing the item or refusing return for extortion, escalate immediately under safety.
* **Escalation Policy**:
  * **Safe to Auto-Handle**: High auto-handling candidate. Provide direct instructions on how to use `help.uber.com` without a phone to contact the driver, and explain the return fee policy ($15 standard).
  * **Escalate**: Driver is unresponsive for > 24 hours, or the lost item is sensitive (passport, medication).

---

### 4. `driver_conduct_and_safety`

* **Definition**: Severe reports concerning driver misconduct, unprofessionalism, verbal abuse, discriminatory remarks, dangerous or reckless driving, vehicle safety hazards, harassment, or physical danger.
* **Empirical Indicators**: `rude`, `unsafe`, `reckless`, `speeding`, `yelled`, `harassment`, `scary`, `dangerous`, `swerving`, `drunk`, `threatened`.
* **Positive Examples**:
  1. *"My driver was texting while speeding on the highway and almost collided with a truck."*
  2. *"The driver started yelling profanities at me when I asked him to turn down the heat."*
  3. *"Driver seemed intoxicated and was swerving between lanes."*
  4. *"Driver refused to let me exit the vehicle at my stop."*
* **Boundary Cases**:
  * *Vs. `pickup_and_route_issue`*: If the driver simply took an inefficient route without aggression or safety violation, map to `pickup_and_route_issue`. If driving was reckless, map here.
* **Escalation Policy**:
  * **MANDATORY ESCALATION (100%)**: Never auto-resolve. High legal and physical safety risk. Must route immediately to Uber's Specialized Safety Response Team with high priority.

---

### 5. `pickup_and_route_issue`

* **Definition**: Operational problems during pickup or transit, including driver arriving at the wrong location, refusing to travel to destination, taking an excessive detour, or failing to move toward pickup.
* **Empirical Indicators**: `wrong pickup`, `detour`, `long way`, `wrong route`, `refused to go`, `car not moving`, `wrong address`.
* **Positive Examples**:
  1. *"Driver picked me up at the wrong terminal and took a 10-mile detour around traffic."*
  2. *"Driver refused to take me to Brooklyn after seeing my destination."*
  3. *"The app showed the driver 2 mins away but the car never moved for 20 minutes."*
  4. *"Driver dropped me off 4 blocks away from my requested destination in the rain."*
* **Boundary Cases**:
  * *Vs. `fare_and_payment_dispute`*: If customer complains about route detour *and* demands fare refund, categorize as `pickup_and_route_issue` if the core complaint describes the operational detour.
* **Escalation Policy**:
  * **Conditional Auto-Handle**: Direct rider to the Route Review tool in the app.
  * **Escalate**: Driver abandoned rider in a remote or hazardous location.

---

### 6. `account_and_promo_issue`

* **Definition**: Inquiries regarding user account management, login / two-factor verification failures, promo codes or discount coupons failing to apply, Ride Pass subscriptions, or app technical bugs.
* **Empirical Indicators**: `promo code`, `discount`, `promo not working`, `login`, `account locked`, `verification code`, `ride pass`, `update phone number`.
* **Positive Examples**:
  1. *"My 20% off promo code failed to apply to my ride yesterday."*
  2. *"I cannot log in because verification codes are sent to my old disconnected phone number."*
  3. *"Why was my rider account suddenly deactivated?"*
  4. *"I paid for a Ride Pass but it is not showing up as active in my app."*
* **Boundary Cases**:
  * *Vs. `fare_and_payment_dispute`*: Excludes disputes over standard trip charges or tolls.
* **Escalation Policy**:
  * **Safe to Auto-Handle**: Explain promo code terms / expiration rules and standard account recovery links.
  * **Escalate**: Account deactivations, suspected account takeovers, or unresolvable 2FA lockouts.

---

## ⚖️ Intent Precedence & Multi-Intent Disambiguation Hierarchy

When a customer message expresses multiple problems in one tweet (*e.g., "Driver cancelled, refused to pick me up, was rude, and charged me $5"*), the system resolves the conflict using this strict priority hierarchy:

```text
Safety / Threat (driver_conduct_and_safety)  [Rank 1 - Highest Priority]
                     ↓
Lost Personal Property (lost_item)          [Rank 2 - Time-Sensitive]
                     ↓
Cancellation Fees (cancellation_issue)      [Rank 3]
                     ↓
Fare Overcharges (fare_and_payment_dispute) [Rank 4]
                     ↓
Route / Pickup (pickup_and_route_issue)     [Rank 5]
                     ↓
Account & Promos (account_and_promo_issue)  [Rank 6 - Lowest Priority]
```

**Guiding Rule**: Safety and time-critical assets always take absolute precedence over financial and account complaints.
