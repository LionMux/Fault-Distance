# Cognitive Patterns for Deep Thinking

Use these reasoning templates when approaching specific problem types. They are not rigid formulas but proven mental models.

---

## Table of Contents

1. [Causal Analysis](#1-causal-analysis)
2. [Strategic Reasoning](#2-strategic-reasoning)
3. [Evaluative Judgment](#3-evaluative-judgment)
4. [Creative Synthesis](#4-creative-synthesis)
5. [Predictive Reasoning](#5-predictive-reasoning)
6. [Decision Under Uncertainty](#6-decision-under-uncertainty)

---

## 1. Causal Analysis

**When to use**: Root-cause analysis, explaining why something happened, debugging.

### The Ladder of Causation

```
Symptom → Proximate Cause → Contributing Cause → Root Cause → Systemic Factor
   ↑            ↑                  ↑                ↑              ↑
  What       Immediate       Enabling           Fundamental    Structural
  happened?   trigger        condition          driver         pattern
```

### Process

1. **Surface symptoms**: What is observable?
2. **5 Whys**: Ask "why" recursively 5 times or until hitting a systemic factor.
3. **Counterfactual test**: If X had not occurred, would Y still have happened?
4. **Necessary vs sufficient**: Is each cause necessary? Sufficient? Neither?

### Pattern: Causal Network, Not Chain

Most problems have multiple interacting causes. Draw a network:

```
    [Factor A] ──┐
                 ├──→ [Outcome]
    [Factor B] ──┤
                 ↑
    [Factor C] ──┘
```

### Traps to Avoid

- **Single-cause fallacy**: Assuming one root cause when multiple exist.
- **Correlation = causation**: Always test with counterfactuals.
- **Reverse causality**: Does X cause Y, or does Y cause X?
- **Omitted variable**: Is there a Z causing both X and Y?

---

## 2. Strategic Reasoning

**When to use**: Planning, competitive analysis, architecture decisions, policy design.

### The Strategy Stack

```
Vision (where we want to be)
    ↓
Strategy (how we get there — 3–5 year)
    ↓
Tactics (specific moves — 1–2 year)
    ↓
Operations (day-to-day execution)
```

### Process

1. **Position assessment**: Where are we now? (honest, no rose-tinted glasses)
2. **Endgame definition**: What does winning look like?
3. **Option generation**: What 3+ paths could get us there?
4. **Second-order thinking**: For each path, ask "and then what?" twice.
5. **Competitive dynamics**: How will others react to each path?
6. **Resource fit**: Which path best matches available resources?

### Pattern: Second-Order Thinking

Always ask:
- First-order: We do X → Y happens.
- Second-order: Y happens → Z happens, others do A, market does B.
- Third-order: Z + A + B → C happens.

Most people stop at first-order. Deep thinkers reach at least second-order.

### Traps to Avoid

- **Strategy without tactics**: Grand visions without executable steps.
- **Tactics without strategy**: Busywork without direction.
- **Sunk cost fallacy**: Continuing because of past investment, not future value.
- **Winner-take-all assumption**: Not all markets have a single winner.

---

## 3. Evaluative Judgment

**When to use**: Choosing between options, quality assessment, prioritization.

### Multi-Factor Evaluation Matrix

1. **Criteria identification**: What dimensions matter? (independently list, don't reuse)
2. **Weight assignment**: How important is each? (sum = 100%)
3. **Option scoring**: Rate each option 1–10 on each criterion.
4. **Weighted sum**: Calculate weighted scores.
5. **Sensitivity check**: If weights change ±20%, does the winner change?

### Pattern: Pre-Mortem

Instead of asking "why will this succeed?", ask:

> "It is 1 year from now. This decision was a disaster. What went wrong?"

Generate at least 3 failure modes. If you cannot, you have not thought hard enough.

### Pattern: Inversion

Instead of solving for success, solve for failure:
- "How do I maximize the chance of failure?" → Then do the opposite.
- Often reveals hidden assumptions about success.

### Traps to Avoid

- **Anchoring bias**: First option or first number disproportionately influences judgment.
- **Recency bias**: Overweighting recent events.
- **False dichotomy**: Presenting two options when more exist.
- **Analysis paralysis**: Perfect evaluation is impossible. Set a decision threshold.

---

## 4. Creative Synthesis

**When to use**: Generating novel solutions, connecting disparate ideas, designing.

### The Synthesis Process

1. **Divergence**: Generate 10+ raw ideas. No filtering. Quantity over quality.
2. **Incubation**: Step back. Let unconscious processing work. (If time allows)
3. **Convergence**: Cluster ideas into themes.
4. **Combination**: Force-combine ideas from different clusters.
5. **Refinement**: Develop the most promising combinations.

### Pattern: Analogical Transfer

- "This problem is structurally similar to [distant domain]."
- "How was [analogous problem] solved?"
- "What transfers? What doesn't?"

Good analogies come from distant domains, not adjacent ones.

### Pattern: Constraint Relaxation

When stuck, systematically relax constraints:
- "What if money were no object?" → then work backward to feasible.
- "What if time were unlimited?" → then compress to timeline.
- "What if [key constraint] didn't exist?" → then reintroduce gradually.

### Traps to Avoid

- **Premature convergence**: Settling on the first viable idea.
- **Functional fixedness**: Seeing objects/functions only in their usual role.
- **Not-invented-here bias**: Rejecting ideas because they came from elsewhere.

---

## 5. Predictive Reasoning

**When to use**: Forecasting, risk assessment, scenario planning.

### Reference-Class Forecasting

1. **Find the reference class**: What category of events is this similar to?
2. **Get base rate**: What happened historically in this class?
3. **Adjust for specifics**: How is THIS case different from the average?
4. **State confidence interval**: Not a point estimate, but a range.

### Scenario Planning (2×2 Matrix)

1. **Identify 2 critical uncertainties** (most impactful and most uncertain).
2. **Create 2×2 matrix**: 4 scenarios.
3. **Name and describe each scenario vividly**.
4. **Assess probability and prepare response for each**.

### Pattern: Bayesian Updating

Start with a prior belief → observe evidence → update belief.

```
P(H|E) = P(E|H) × P(H) / P(E)

Where:
  P(H) = prior probability of hypothesis
  P(E|H) = likelihood of evidence if hypothesis is true
  P(E) = total probability of evidence
```

### Traps to Avoid

- **Overconfidence in prediction**: Humans are terrible predictors. Widen confidence intervals.
- ** cherry-picked analogies**: Selecting historical cases that confirm your view.
- **Linear extrapolation**: Most trends are non-linear.

---

## 6. Decision Under Uncertainty

**When to use**: High-stakes decisions with incomplete information.

### The Expected Value Framework

For each option:
1. List possible outcomes.
2. Assign probability to each.
3. Assign value to each.
4. Calculate: EV = Σ(P(outcome) × Value(outcome))

### The Minimax Regret Framework

1. For each option, identify the worst-case realistic scenario.
2. For each, calculate regret (difference from best possible outcome).
3. Choose the option with the minimum maximum regret.

### The Information Value Test

Before deciding, ask:
- "What information, if I had it, would change my decision?"
- "How much would I pay for that information?"
- "Can I acquire it in time?"

If information value > cost of acquiring it → Delay decision and gather.

### Traps to Avoid

- **Probability neglect**: Focusing on outcomes while ignoring their likelihood.
- **Zero-risk bias**: Preferring to reduce small risk to zero over greater risk reduction elsewhere.
- **Ambiguity aversion**: Preferring known bad over unknown possibly-good.
