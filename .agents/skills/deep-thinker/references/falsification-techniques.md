# Advanced Falsification Techniques

Use these when standard contradiction-hunting is insufficient. These techniques are designed to stress-test hypotheses beyond normal scrutiny.

---

## Table of Contents

1. [Red Teaming](#1-red-teaming)
2. [Pre-Mortem Analysis](#2-pre-mortem-analysis)
3. [Inversion Method](#3-inversion-method)
4. [Crucial Experiment Design](#4-crucial-experiment-design)
5. [Steel-Man → Destroy](#5-steel-man--destroy)
6. [Necessity-Sufficiency Audit](#6-necessity-sufficiency-audit)
7. [Temporal Stress Test](#7-temporal-stress-test)
8. [Expert Disagreement Mapping](#8-expert-disagreement-mapping)

---

## 1. Red Teaming

**Purpose**: Assign a dedicated adversarial role to attack the hypothesis.

### Process

1. Adopt the persona of someone who wants this hypothesis to fail.
2. Generate 5+ specific attacks.
3. For each attack, assess: Does the hypothesis survive? At what cost?

### Persona Options

- **The Skeptic**: "I don't believe anything without peer-reviewed evidence."
- **The Competitor**: "I have a vested interest in this being wrong."
- **The Historian**: "This has failed before in similar contexts."
- **The Detail-Obsessive**: "This seems right until you look at edge case X."
- **The Incentives-Analyst**: "Who benefits from this being believed, and how?"

### Output

```
Red Team: [Persona]
Attacks:
  1. [Attack] → [Hypothesis survives/fails] → [Cost of survival]
  2. [Attack] → [...]
  ...
```

---

## 2. Pre-Mortem Analysis

**Purpose**: Assume the hypothesis led to failure, then explain why.

### Process

1. State the hypothesis as if it were already acted upon.
2. Jump forward in time: "It is [timeframe] later. The decision based on H1 was a disaster."
3. Generate at least 3 plausible failure modes.
4. For each failure mode, assess: How likely? How detectable in advance?

### Key Question

> "What warning signs would I see 1 week / 1 month / 1 year before this fails?"

If you cannot name warning signs, the hypothesis has not been stress-tested.

---

## 3. Inversion Method

**Purpose**: Instead of proving the hypothesis true, prove its negation false.

### Process

1. State the inverse: "The opposite of H1 is..."
2. Ask: "What would I observe if the inverse were TRUE?"
3. Check: Do I observe those things?
4. If the inverse predictions are NOT observed → H1 is supported (but not proven).

### Example

- H1: "This performance issue is caused by database queries."
- Inverse: "Database queries are NOT the cause."
- Inverse prediction: "Optimizing queries will have no effect."
- Test: If query optimization improves performance → inverse falsified → H1 supported.

---

## 4. Crucial Experiment Design

**Purpose**: Find a single observation that would definitively rule out the hypothesis.

### Process

1. Ask: "What single piece of evidence, if observed, would make me abandon H1 entirely?"
2. If no such evidence exists → H1 is unfalsifiable → reject it (Popperian criterion).
3. If such evidence exists → Can I obtain it? If yes, gather it before concluding.

### The Litmus Test

A good crucial experiment must:
- Have at least 2 competing hypotheses make opposite predictions about its outcome.
- Be practically observable.
- Be unambiguous in interpretation.

---

## 5. Steel-Man → Destroy

**Purpose**: Strengthen the hypothesis as much as possible, THEN attack it.

### Process

1. State the weakest version of the hypothesis.
2. Iteratively strengthen it: "Even if [objection], it still holds because..."
3. Continue until no further strengthening is possible.
4. NOW attack the strongest version.

### Why This Works

Destroying a straw-man is useless. If the strongest version fails, the hypothesis is genuinely weak. If it survives, the confidence is earned.

---

## 6. Necessity-Sufficiency Audit

**Purpose**: Check whether the hypothesis's conditions are truly necessary and/or sufficient.

### Process

For each condition in the hypothesis:

| Question | Test | If True |
|----------|------|---------|
| Is this necessary? | Remove it. Does conclusion still hold? | If yes → condition is NOT necessary. Simplify. |
| Is this sufficient? | Is it enough ALONE? What else is needed? | If not sufficient → identify missing conditions. |
| Is it neither? | Both tests above | Condition is decorative. Remove. |

### Common Finding

Most hypotheses contain decorative conditions that feel important but are neither necessary nor sufficient.

---

## 7. Temporal Stress Test

**Purpose**: Check if the hypothesis holds across different time scales.

### Process

Apply the hypothesis to:

1. **Immediate** (hours/days): Does it hold right now?
2. **Short-term** (weeks/months): Does it hold after initial conditions change?
3. **Medium-term** (1–3 years): Does it survive structural shifts?
4. **Long-term** (5+ years): Does it hold under compounding effects?

### Pattern: The Lindy Test

> "How long has this pattern/type of hypothesis been true? The longer, the more likely it continues."

Counter-pattern: Lindy does NOT apply to rapidly changing domains (technology, fashion).

---

## 8. Expert Disagreement Mapping

**Purpose**: Understand why smart people disagree about this hypothesis.

### Process

1. Identify 2+ credible positions on the issue.
2. For each position, identify:
   - Core belief
   - Key evidence they emphasize
   - What they ignore or discount
   - Incentives/biases they may have
3. Map where the disagreement is factual (different data) vs. interpretive (same data, different conclusions) vs. values-based.

### The Crucial Question

> "If I were as smart as the smartest person who disagrees with me, what would I believe?"

If you cannot genuinely understand the opposing view, you do not understand your own.

---

## Selection Guide

| Situation | Primary Technique | Secondary Technique |
|-----------|------------------|-------------------|
| Hypothesis feels "too easy" | Red Teaming | Pre-Mortem |
| High-stakes decision | Pre-Mortem | Expert Disagreement Mapping |
| Unfalsifiable-seeming claim | Crucial Experiment Design | Inversion Method |
| Strong confirmation bias suspected | Steel-Man → Destroy | Necessity-Sufficiency Audit |
| Long-term prediction | Temporal Stress Test | Reference-Class Forecasting |
| Polarized topic | Expert Disagreement Mapping | Red Teaming |
