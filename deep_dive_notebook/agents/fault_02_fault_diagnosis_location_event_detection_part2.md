# Fault-focused ML (Diagnosis / Location / Event Detection) — Part 2 (ranked papers + training templates)

## A) Paper list (Part 2)
This file continues the fault-focused curation. **Same integrity constraint as Part 1:** I will not invent “full citations” without reliable bibliographic verification in this environment.

If you can provide DOI/arXiv IDs (or allow a lookup step), I will:
- produce a true ranked list of ~20–25 total papers,
- ensure each item includes: authors, year, title, and venue/arXiv.

For now, the remainder of this file provides **additional deep-dive outline sections** that cover other frequent protection tasks: impedance/admittance learning, semi-supervised fault detection, and hybrid physics-informed objectives.

---

## B) Deep-dive outline 7 — Impedance/admittance learning for fault classification & location

### B1) Typical input
From measured electrical quantities (often SCADA + inferred steady-state variables):
- apparent impedance \(Z_{meas}\)
- admittance \(Y_{meas}\)
- sequence components (positive/negative sequence)

Fault changes network equivalent:
\[
Z_{eq}(\theta) \neq Z_{eq}(\theta_0)
\]
where \(\theta\) includes topology and operating-point parameters.

### B2) Model families
- MLP / 1D CNN over vectorized \( [\Re(Z), \Im(Z), ...] \)
- Siamese networks to compare signatures with a library
- Metric learning:
  \[
  d(f(x_a), f(x_b))
  \]
  for “same fault type/location” similarity.

### B3) Loss functions
- Metric learning (triplet loss):
\[
\mathcal{L}_{tri} = \max(0, d(a,p)-d(a,n)+m)
\]
- Probabilistic location (classification over distance bins):
\[
\mathcal{L} = -\sum_b y_b \log p_b
\]

### B4) Physical evaluation
- confusion cost: misclassifying near/adjacent segments
- calibration: reliability of “fault probability”
- invariance checks: scaling/transform invariance if impedance magnitudes vary

---

## C) Deep-dive outline 8 — Semi-supervised / self-supervised for rare faults

### C1) Problem
Fault events are rare compared with normal operation → limited labeled faults.

### C2) Self-supervised pretraining
Contrastive objective on waveform/representation:
\[
\mathcal{L}_{NCE} = -\log \frac{\exp(\mathrm{sim}(z_i,z_i^+)/\tau)}{\sum_j \exp(\mathrm{sim}(z_i,z_j)/\tau)}
\]
Augmentations:
- time shift within safe margin
- noise injection consistent with sensor SNR
- masking (SpecAugment-like on time-frequency images)

### C3) Fine-tuning
- Freeze encoder partially, then CE/Huber losses for supervised tasks.

### C4) Physical metrics
- improvement in detection under unseen operating points
- robustness vs noise and parameter uncertainty

---

## D) Deep-dive outline 9 — Physics-informed / hybrid models for protection-compatible inference

### D1) Hybrid design pattern
Use physics to constrain outputs:
- ML predicts parameters \(\hat \theta\) (fault impedance, location segment)
- physics layer computes expected observables \(\hat x_{phys}\)
- loss enforces consistency:
\[
\mathcal{L}_{cons} = \|x_{meas} - \hat x_{phys}(\hat\theta)\|_2^2
\]

### D2) Combined objective
\[
\mathcal{L} = \lambda_{sup}\mathcal{L}_{sup} + \lambda_{cons}\mathcal{L}_{cons} + \lambda_{reg}\|\hat\theta\|^2
\]

### D3) Protection relevance
- fast inference (physics layer closed form or small differentiable solver)
- uncertainty estimation for threshold coordination

---

## E) Deep-dive outline 10 — Protection decision thresholds from calibrated ML outputs

### E1) Map probabilities to relay decisions
If ML outputs probability of fault \(p(\text{fault}\mid x)\):
- trip if \(p \ge \gamma\)

Choose \(\gamma\) to meet protection constraints:
- bound false trip rate
- ensure detection latency within coordination window

### E2) Evaluation
- ROC curve with constraint-based operating points
- expected miscoordination cost (domain-specific)
- compute “effective zone correctness”:
  - fraction of trips whose predicted location is within correct relay zone reach.

---

## F) What to do next (so this file can become a real “ranked papers” inventory)
Please provide DOI/arXiv IDs or a seed list you trust. Then I will:
- rewrite the paper lists in Part 1 + Part 2 into a complete, ranked set of ~20–25 items,
- ensure each citation is fully accurate.