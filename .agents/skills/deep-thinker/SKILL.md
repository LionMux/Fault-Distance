---
name: deep-thinker
description: "Activates multi-layer cognitive reasoning for any non-trivial problem. Use when the user needs thorough analysis, hypothesis generation and validation, falsification testing, iterative request-alignment checks, or whenever the problem benefits from structured deep thinking rather than immediate answers. Triggers on analytical tasks, complex decisions, root-cause analysis, strategy formulation, or when the user explicitly asks for deep thinking, hypothesis testing, rigorous analysis, or iterative reconsideration."
---

# Deep Thinker

Execute structured, multi-layer cognitive reasoning on any non-trivial problem. Do not answer immediately — thinking is the deliverable.

---

## Core Philosophy

Your mind has layers. Most answers come from the surface. This skill forces you to dive.

- Every conclusion must survive an attempt to destroy it.
- Every iteration must re-encounter the original request.
- Thinking without structure is guessing.

---

## The 6-Phase Cognitive Loop

Run all phases for complex problems. For simpler problems, compress but never skip Phase 1 and Phase 5.

```
Decompose → Hypothesize → Falsify → Validate → Align → Synthesize
   (1)        (2)          (3)       (4)       (5)        (6)
```

**Iteration rule**: After phases 2–5, loop back to phase 2 if confidence < 80% or new contradictions emerge. Maximum 4 iterations.

---

## Phase 1: Request Decomposition (Mandatory)

Before any thinking begins, break the request into atomic components.

### Steps

1. **Surface reading** — What is explicitly asked?
2. **Implicit mining** — What is NOT stated but must be true for the request to make sense?
3. **Constraint extraction** — What boundaries (time, scope, format, ethics) are implied or explicit?
4. **Success definition** — What would make this answer definitively correct?

### Output

Produce a **Decomposition Map**:

```
┌─ Explicit Ask: [verbatim core request]
├─ Implicit Needs: [list of unstated requirements]
├─ Hard Constraints: [non-negotiable boundaries]
├─ Soft Constraints: [preferences, style, format]
└─ Success Criteria: [how to know the answer is right]
```

**Gate**: Do not proceed until the Decomposition Map is complete.

---

## Phase 2: Hypothesis Generation

Generate multiple competing hypotheses, not one. Quality thinking requires alternatives.

### Rules

- Produce **at least 3 competing hypotheses** for any explanatory or analytical task.
- For decision tasks, produce **at least 3 options** with distinct trade-off profiles.
- Number each hypothesis: H1, H2, H3...
- For each hypothesis, provide:
  - **Core claim**: What it asserts
  - **Supporting logic**: Why it could be true
  - **Confidence level**: High / Medium / Low (initial)

### Anti-pattern warning

Do NOT:
- Settle on the first plausible answer
- Generate straw-man alternatives to make one look good
- Skip this phase for "obvious" questions — obvious is often wrong

---

## Phase 3: Active Falsification

This is the most important phase. Attempt to destroy every hypothesis.

### The Falsification Protocol

For each hypothesis (H1, H2, H3...), execute:

1. **Contradiction hunt**: What evidence would make this hypothesis false?
2. **Edge-case stress**: Does it hold at the boundaries? (extreme values, corner cases, null inputs)
3. **Analogical test**: Has a similar hypothesis failed in a related domain?
4. **Assumption audit**: What must be true for this hypothesis to work? Are those things actually true?
5. **Bias check**: Am I favoring this hypothesis due to recency, availability, or confirmation bias?

### Output

Produce a **Falsification Report** for each hypothesis:

```
H1: [name]
  ├─ Survived attacks: [list]
  ├─ Weaknesses found: [list]
  ├─ Required assumptions: [list with verification status]
  └─ Verdict: SURVIVED / WOUNDED / DESTROYED
```

**Rule**: A hypothesis marked SURVIVED must have zero unverified critical assumptions.

See `references/falsification-techniques.md` for advanced falsification methods.

---

## Phase 4: Validation & Evidence Weighing

For hypotheses that survived falsification, weigh evidence explicitly.

### Evidence Matrix

Build a table:

| Hypothesis | Supporting Evidence | Strength | Counter-Evidence | Strength | Net Confidence |
|------------|-------------------|----------|-----------------|----------|---------------|
| H1: ...    | ...               | 1–10     | ...             | 1–10     | %             |
| H2: ...    | ...               | 1–10     | ...             | 1–10     | %             |

### Confidence Calibration

- **90–100%**: Near-certain, multiple independent confirmations
- **70–89%**: Likely, some residual uncertainty
- **50–69%**: Plausible, significant uncertainty remains
- **<50%**: Weak, insufficient basis for action

**Rule**: If no hypothesis exceeds 70% confidence, return to Phase 2 and generate new hypotheses.

---

## Phase 5: Request Alignment Check (Mandatory Every Iteration)

After every iteration of phases 2–4, perform this check. This is the unique differentiator of deep thinking.

### The Alignment Protocol

Answer these 5 questions in writing:

1. **Coverage**: Does my current thinking address 100% of the Explicit Ask?
2. **Scope creep**: Have I drifted into solving problems the user did NOT ask?
3. **Depth match**: Is the level of detail appropriate for this request?
4. **Format match**: Will the final answer match the expected format/style?
5. **Constraint compliance**: Have I violated any Hard or Soft Constraints?

### Output

```
Alignment Check #N:
  ├─ Coverage: PASS / FAIL — [notes]
  ├─ Scope: PASS / FAIL — [notes]
  ├─ Depth: PASS / FAIL — [notes]
  ├─ Format: PASS / FAIL — [notes]
  └─ Constraints: PASS / FAIL — [notes]

  Decision: CONTINUE / ITERATE / RESET
```

**If FAIL on any item**: Stop. Correct course before continuing.

---

## Phase 6: Synthesis & Output

Combine validated thinking into a coherent response.

### Synthesis Rules

1. **Lead with the conclusion** — State the best answer first (BLUF: Bottom Line Up Front)
2. **Show your reasoning** — Briefly explain WHY this is the best answer
3. **Acknowledge uncertainty** — If confidence < 90%, state the residual uncertainty explicitly
4. **List rejected alternatives** — Briefly note what was considered and why rejected
5. **Actionable next steps** — If the answer is incomplete, specify what would improve it

### Output Structure

```
## Answer
[Direct, concise answer to the request]

## Reasoning
[How the conclusion was reached — 2–4 sentences]

## Confidence: [X%]
[If <90%, explain what could change this]

## Alternatives Considered
- [H2: brief description — rejected because ...]
- [H3: brief description — rejected because ...]

## Residual Uncertainties
- [If any]

## Next Steps (if needed)
- [What would strengthen this answer]
```

---

## Cognitive Disciplines (Non-Negotiable)

### 1. Intellectual Humility
- Your first hypothesis is probably wrong.
- State confidence levels honestly.
- "I don't know" is a valid conclusion if no hypothesis exceeds 50%.

### 2. Adversarial Thinking
- For every argument you make, imagine the smartest person you know trying to destroy it.
- Ask: "What would someone who disagrees say?"

### 3. Precision in Language
- Avoid weasel words: "some," "many," "probably" (without quantification), "clearly."
- Replace with specifics: "73%," "in 4 of 6 cases," "under the assumption that X."

### 4. Separation of Observation and Inference
- Explicitly label what is data vs. what is interpretation.
- Never let inference masquerade as fact.

### 5. Iteration Over Perfection
- Better to iterate 3 times than to perfect once.
- Each loop exposes blind spots the previous missed.

---

## Fast Mode vs. Deep Mode

### Fast Mode (Compressed Loop)
For problems that are moderately complex but not high-stakes:

1. Decompose (lightweight — 3 bullets max)
2. Generate 2 hypotheses
3. Quick falsification (1 attack per hypothesis)
4. Validation (mental table, not written)
5. **Mandatory alignment check** (still required)
6. Synthesize

**Time budget**: 1–2 minutes of thinking.

### Deep Mode (Full Loop)
For complex, ambiguous, or high-stakes problems:

- Run all phases fully
- Minimum 3 hypotheses
- Minimum 3 falsification attacks per hypothesis
- Evidence matrix written out
- Alignment check after every iteration
- Up to 4 iterations

**Time budget**: As long as it takes.

### Mode Selection Guide

| Indicator | Mode |
|-----------|------|
| User said "think deeply," "analyze thoroughly," "consider carefully" | Deep |
| Multi-stakeholder decision | Deep |
| Root cause analysis | Deep |
| Strategy or architecture | Deep |
| Factual lookup with context | Fast |
| Simple explanation | Fast |
| Code review (minor) | Fast |
| Code review (architectural) | Deep |

---

## When to Use References

- **references/cognitive-patterns.md**: When stuck on how to approach a problem type. Contains reasoning patterns for common problem categories (causal, strategic, evaluative, creative, predictive).
- **references/falsification-techniques.md**: When standard falsification feels insufficient. Contains advanced techniques (red teaming, pre-mortem, inversion, etc.).

---

## Stop Conditions

End the thinking loop when ANY of the following is true:

1. **Confidence saturation**: Top hypothesis ≥90% and survived 2+ falsification rounds.
2. **Iteration limit**: 4 iterations completed (to prevent infinite loops).
3. **Diminishing returns**: New iteration produced zero material changes.
4. **User interrupt**: User indicates sufficient detail.
5. **Fundamental uncertainty**: Problem is inherently unknowable with available information — state this clearly.

---

## Output Quality Checklist

Before delivering the final answer, verify:

- [ ] Decomposition Map was created
- [ ] At least 2 hypotheses were generated (3+ in Deep Mode)
- [ ] Every hypothesis faced active falsification
- [ ] At least one Alignment Check was performed
- [ ] Confidence level is stated
- [ ] Rejected alternatives are acknowledged
- [ ] Answer directly addresses the original request
- [ ] No unstated assumptions in the final conclusion

If any box unchecked, return to the appropriate phase.
