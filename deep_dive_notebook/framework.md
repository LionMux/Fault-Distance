# Variant A Study Notebook Framework (Master Workflow + Per-Paper Deep-Dive Template)

This document defines the **shared structure** used for Variant A “deep-dive” notes across all curated papers (fault diagnosis/localization/event detection and protection-related ML, plus the forecasting/stability/security tracks from other agents). It enforces:

- **Mathematical rigor** (explicit notation, assumptions, objectives, and derivations where relevant)
- **Intuitive explanations** (plain-language meaning of every modeling choice)
- **Reproducible training logic** (data → features → model \(f_\theta\) → loss → optimizer update → regularization → validation)
- **Physically grounded evaluation** (mapping ML metrics to grid/physical impact)

---

## 0) Notation conventions (used throughout)

- Scalars: \(a, b, \alpha, \beta\)
- Vectors: \(\mathbf{x}, \mathbf{y}, \mathbf{h}\)
- Matrices: \(\mathbf{X}, \mathbf{W}\)
- Sets: \(\mathcal{S}\)
- Random variables: \(X, Y\)
- Distributions: \(p(\cdot), q(\cdot)\)
- Time index: \(t\in\{1,\dots,T\}\)
- Sample index: \(i\in\{1,\dots,N\}\)
- Parameters: \(\theta\)
- Model: \(f_\theta(\cdot)\)
- Loss: \(\mathcal{L}(\cdot)\)
- Regularization strength/hyperparameter: \(\lambda\)

---

## 1) Master Outline (workflow chapter)

> Use this outline as the top-level chapter for Variant A. Each paper note should link back to the workflow steps below.

### 1.1 Problem formalization (grid task → ML learning objective)
1. **Task definition**
   - Input modality: (PMU/SCADA waveform, events, topology, relay signals, logs, etc.)
   - Output type:
     - classification (fault/no-fault, event type)
     - detection (alarm vs no-alarm)
     - regression (fault location coordinates, time, distance, impedance)
     - ranking/severity estimation (risk score)
   - Output constraints (e.g., physically feasible location on a graph, monotonicity, causality)
2. **Learning formulation**
   - Supervised \((\mathbf{x}, y)\), self-supervised, weakly supervised, or semi-supervised
   - Define target variable(s) precisely (e.g., \(y\) for event time, \(y\in\mathbb{R}\) for distance)
3. **Assumptions**
   - Noise model assumptions (measurement noise, missingness)
   - Sampling/temporal alignment assumptions (windowing choices)
   - Distribution shift considerations (different fault types, operating points)

**Template sentences**
- “We denote the input as \(\mathbf{x}\in\mathbb{R}^d\) and the target as \(y\). The goal is to learn \(f_\theta\) such that …”

---

### 1.2 Data pipeline design (from raw measurements to training tensors)
1. **Acquisition & synchronization**
2. **Event segmentation / windowing**
3. **Feature extraction (raw-to-features map \(g(\cdot)\))**
   - time-domain features: magnitudes/phasors, transient energy, slopes
   - frequency-domain features: spectra, harmonic content
   - graph features: node/edge attributes, topology embeddings
4. **Preprocessing**
   - normalization/scaling
   - handling missing data
   - class imbalance handling plan
5. **Train/val/test splits**
   - avoid leakage: by fault episode, by feeder, by operating regime

**Template**
- “Define \(\mathbf{x}=g(\text{raw})\). We normalize using statistics computed on training only.”

---

### 1.3 Model design (architecture family → inductive bias)
1. Select inductive bias:
   - CNN/TCN for local temporal patterns
   - RNN/Transformer for long-range dependencies
   - Graph neural networks for topology-aware reasoning
   - Physics-informed / constraint-based components
2. Specify the forward model:
   - \( \hat{y} = f_\theta(\mathbf{x}) \)
3. If sequence:
   - \( \hat{y}_t=f_\theta(\mathbf{x}_{1:t}) \) (causal) or window-based \( \hat{y}=f_\theta(\mathbf{x}_{t:t+W}) \)

**Rigor requirement**
- Every claim about “what the model captures” must connect to the architecture’s mathematical operation (e.g., receptive field, message passing, attention weights).

---

### 1.4 Training objective (loss, regularization, optimization)
1. Define the base loss \(\mathcal{L}_{\text{base}}\)
   - cross-entropy, focal loss, MSE/Huber, negative log-likelihood, ranking loss, etc.
2. Define regularization \(\mathcal{R}(\theta)\)
   - weight decay, dropout (interpretation), spectral norm, label smoothing, constraint penalties
3. Total objective:
\[
\mathcal{L}_{\text{total}}(\theta)=\mathbb{E}_{(x,y)\sim \mathcal{D}}\left[\mathcal{L}_{\text{base}}(f_\theta(x),y)\right]+\lambda \mathcal{R}(\theta)
\]
4. Optimizer update step (show the update rule at least conceptually)
   - SGD/Adam update: \(\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{total}}\)

**Intuition requirement**
- After each loss component, add a short intuitive meaning: “This term penalizes … because …”

---

### 1.5 Validation & selection (metrics → decisions)
1. Define metrics aligned with task:
   - detection: precision/recall, F1, ROC-AUC, false alarm rate
   - regression: MAE/RMSE, calibration error
   - classification: accuracy, macro-F1
2. Choose stopping/selection criterion.
3. Robustness tests:
   - noise injection, domain shift, different operating points
   - ablations: remove features / remove topology bias / remove regularization

---

### 1.6 From ML outputs to physical/grid actions (impact mapping)
1. Convert predicted outputs to operational signals:
   - trip decision, alarm issuance, protection setting adjustment
2. Define the impact measure:
   - time-to-detection vs damage growth
   - spatial error → wrong switch isolation zone
   - risk score → probability of unsafe operation
3. Provide a mapping from ML metrics (e.g., MAE, F1) to these impact quantities (section 5).

---

## 2) Paper Inventory Schema (for a consistent index)

> Create one entry per paper. This schema is intended for a structured spreadsheet/JSON-like notebook section.

### 2.1 Inventory fields (minimum)
- **Paper ID**: `P###`
- **Citation**: full reference (authors, year, venue)
- **Archive**: IEEE Xplore / arXiv / ScienceDirect / Springer / other
- **Domain**: {Fault diagnosis, Fault location, Event detection, Protection, Forecasting, Stability, Security}
- **Task type**:
  - Detection / Classification / Regression / Sequence forecasting / Graph reasoning / Other
- **Signals / Inputs**:
  - PMU channels? SCADA variables? relay outputs? topology data?
- **Output**:
  - label definition, units (e.g., distance in km, bus index, event class)
- **Model family**:
  - CNN / TCN / RNN/LSTM / Transformer / GNN / Hybrid / Physics-informed / Other
- **Feature pipeline**:
  - raw-to-features map \(g(\cdot)\): what transforms are used?
- **Supervision**:
  - fully supervised / weak / semi / self-supervised
- **Training setup**:
  - dataset size \(N\), window length \(W\), batch size (if stated)
- **Loss**:
  - \(\mathcal{L}_{\text{base}}\) and regularization strategy
- **Key results (with metrics)**:
  - top-1 metrics + baselines + improvement magnitude
- **Ablations**:
  - which components were removed/modified?
- **Robustness / Generalization**:
  - tests under noise, topology changes, different regimes
- **Physical plausibility checks**:
  - constraint enforcement, feasibility, monotonicity, etc.
- **Failure modes**:
  - where the model breaks and why (as described/argued)
- **Reproducibility notes**:
  - code availability, hyperparameters, seeds, training length

### 2.2 Inventory schema (suggested YAML-like block)
```yaml
paper_id: P001
citation: "Author et al., Year, Venue"
archive: "IEEE Xplore"
domain: "Fault diagnosis"
task_type: ["Detection","Classification"]
inputs:
  - modality: "PMU"
    channels: ["V", "I", "freq"]   # example
outputs:
  - name: "event_type"
    type: "classification"
    classes: ["SLG","LL","3ph"]   # example
feature_pipeline:
  - transform: "windowing"
    window_length: W
  - transform: "phasor magnitude"
model_family: ["Transformer","MLP head"]
supervision: "fully_supervised"
training:
  dataset_size: N
  loss_base: "focal_loss"
  regularization: "weight_decay"
key_results:
  metric_primary: "macro-F1"
  value: 0.87
  baseline_improvement: "+0.12 vs best baseline"
ablation_summary:
  - "removing topology embedding reduces macro-F1 to 0.79"
failure_modes:
  - "high-noise regime causes false alarms around load switching"
physical_plausibility:
  - "prediction obeys feasible bus subset constraint"
```

---

## 3) Per-Paper Deep-Dive Section Template (copy/paste)

> Use this template verbatim for each paper. Replace bracketed placeholders. Keep the **math** and **intuition** sections together.

---

### 3.1 Summary block (1–2 paragraphs)
- **What problem does the paper solve?**
- **What is the key modeling contribution?**
- **What is the empirical performance and under what conditions?**

---

### 3.2 Task & notation (precise definitions)
1. **Inputs**: define \(\mathbf{x}\) and how it is constructed:
   - raw measurements → preprocessing → windowing → features
2. **Targets**: define \(y\) (and if sequence: \(y_t\))
3. **Prediction rule**:
\[
\hat{y} = f_\theta(\mathbf{x})
\]
4. **Assumptions/constraints**:
   - e.g., causality, physical feasibility, monotonic relationship, graph constraint

**Intuition (mandatory)**
- Explain what information in \(\mathbf{x}\) should be relevant and why.

---

### 3.3 Data pipeline: raw-to-features map \(g(\cdot)\)
Describe the full pipeline:
- acquisition/sync
- segmentation/windowing
- feature engineering or learned feature extraction
- normalization/scaling and leakage prevention
- train/val/test split logic

**Mathematical placeholder**
\[
\mathbf{x} = g(\text{raw signals}; \phi_g)
\]
If they use learned features, define the internal parameters too.

---

### 3.4 Model architecture: \(f_\theta\) (equations + explanation)
Provide:
- architecture diagram description
- the forward pass equations (as close to the paper as possible)
- parameter roles (what parts learn what)

**Example sub-structure**
- If graph-based:
  - message passing:
\[
\mathbf{h}_v^{(k+1)} = \sigma\left(\sum_{u\in \mathcal{N}(v)} \alpha_{uv}^{(k)} \mathbf{W}^{(k)} \mathbf{h}_u^{(k)}\right)
\]
- If temporal:
  - causal convolution receptive field / attention computation
- Output head:
  - \( \hat{y}=\text{Head}(\mathbf{h}) \)

**Intuition (mandatory)**
- For each major module, explain: “This module is designed to capture … because …”

---

### 3.5 Training-derivation sub-template (must include the full chain)
Use the following explicit sequence:

#### (A) Data
- Training set: \(\mathcal{D}=\{(\mathbf{x}_i,y_i)\}_{i=1}^N\)
- Batching: \(\{(\mathbf{x}_i,y_i)\}_{i\in\mathcal{B}}\)

#### (B) Features
\[
\mathbf{x}_i=g(\text{raw}_i)
\]

#### (C) Model forward
\[
\hat{y}_i=f_\theta(\mathbf{x}_i)
\]

#### (D) Loss
Define base loss:
- Classification:
\[
\mathcal{L}_{\text{base}}=\; -\sum_{c} y_{i,c}\log \hat{y}_{i,c}
\]
- Regression:
\[
\mathcal{L}_{\text{base}}=\|\hat{y}_i-y_i\|_2^2 \quad \text{or}\quad \text{Huber}(\hat{y}_i,y_i)
\]

Write the total loss expression used in training:
\[
\mathcal{L}_{\text{total}}(\theta)=\frac{1}{|\mathcal{B}|}\sum_{i\in\mathcal{B}}\mathcal{L}_{\text{base}}(f_\theta(\mathbf{x}_i),y_i)+\lambda \mathcal{R}(\theta)
\]

**Intuition**
- Explain what the loss penalizes (errors you care about) and how that maps to physical meaning.

#### (E) Optimizer update (show the update rule)
For SGD-like update:
\[
\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}_{\text{total}}(\theta)
\]
If Adam:
- describe that the first/second moments are maintained, but still link it back to gradient of \(\mathcal{L}_{\text{total}}\).

#### (F) Regularization
Describe each regularizer:
- weight decay: \(\mathcal{R}(\theta)=\|\theta\|_2^2\)
- dropout (conceptual effect)
- label smoothing / constraint penalty
- data augmentation and its role

Add an “effect on generalization” sentence:
- “This regularization reduces … by …”

#### (G) Validation & model selection
- validation metric(s)
- early stopping criterion
- hyperparameter tuning approach

---

### 3.6 Results: metrics, baselines, and ablations (with interpretation)
Include:
- primary metrics and values
- baselines and why they’re fair
- ablation results and what each removed component implies physically

**Intuition (mandatory)**
- Translate “improved metric” → “better physical decision”:
  - fewer false trips
  - smaller location error
  - earlier alarm onset

---

### 3.7 Robustness & failure analysis (where the model may mislead)
Required:
- performance under noise
- out-of-distribution operating points
- topology changes or unseen fault types
- mismatch between training/testing event definitions

Then:
- list 2–5 failure modes (specific, not generic)
- provide a hypothesis: “the model fails because …”

---

### 3.8 Physical impact mapping (explicit)
Use the guidelines in section 5 to convert ML outputs to protection/grid impact:
- decision layer (alarm/trip/location)
- impact costs (risk of unsafe operation vs unnecessary intervention)
- final statement: “Given metric \(M\), we expect impact \(I\) to be …”

---

### 3.9 Key takeaways (bullet list)
- contribution
- best-performing setting
- conditions under which it works
- open limitations

---

## 4) Consistent Notation Glossary (fill once, reference everywhere)

Use this section to maintain consistent meaning across papers.

| Symbol | Meaning |
|---|---|
| \(N\) | number of training samples/episodes |
| \(T\) | number of time steps in a window |
| \(W\) | window length (samples) |
| \(\mathbf{x}\) | feature vector/tensor input to the model |
| \(g(\cdot)\) | raw-to-features transformation |
| \(\phi_g\) | parameters of feature transform (if learned) |
| \(f_\theta(\cdot)\) | prediction model |
| \(\theta\) | model parameters |
| \(\hat{y}\) | predicted output |
| \(y\) | ground truth output |
| \(\mathcal{L}_{\text{base}}\) | primary loss term |
| \(\mathcal{R}(\theta)\) | regularization function |
| \(\lambda\) | regularization strength |
| \(\eta\) | learning rate |
| \(\mathcal{D}\) | data distribution |

If the paper uses specialized symbols (e.g., phasor angles, graph adjacency \(A\), attention weights), add them here for that paper.

---

## 5) Guidelines: mapping ML metrics to physical grid impact

The goal is to avoid “metric-only” evaluation. Every curated paper note should include **at least one** mapping from ML performance to grid-relevant quantities.

### 5.1 Build a decision model (what action does the ML output trigger?)
1. **Protection logic layer**
   - detection threshold: \(\hat{p}(\text{fault})\ge \tau\)
   - trip logic: time-to-trip or relay setting update
2. **Localization/action layer**
   - choose isolation zone based on predicted location \(\hat{\ell}\)

Define this explicitly:
- alarm indicator \(a(\hat{y})\)
- trip decision \(t(\hat{y})\)
- selected asset set \(\Omega(\hat{y})\)

### 5.2 Choose impact costs (risk vs unnecessary actions)
Define two kinds of costs:

- **Missed detection cost** \(C_{\text{miss}}\)
  - corresponds to delayed/incorrect protection clearing
  - physically: fault current persists longer → thermal/mechanical stress increases
- **False alarm/intervention cost** \(C_{\text{false}}\)
  - unnecessary trips → service interruption and operational cost

Then define an operational expected cost:
\[
\mathbb{E}[C] = C_{\text{miss}}\cdot P(\text{miss}) + C_{\text{false}}\cdot P(\text{false alarm})
\]

### 5.3 Translate common ML metrics into these probabilities

#### For detection/classification
- Precision, recall relate to:
  - \(P(\text{false alarm})\)
  - \(P(\text{miss})\)
- Example mapping (conceptual):
  - If recall \(R=\frac{\text{TP}}{\text{TP}+\text{FN}}\), then
\[
P(\text{miss}) \approx 1-R
\]
- Use threshold selection (ROC/PR curve) to minimize expected cost for a chosen policy:
\[
\tau^\*=\arg\min_{\tau}\; \mathbb{E}[C(\tau)]
\]

#### For regression (fault location, event time)
- MAE/RMSE map to expected spatial/time error:
\[
\mathbb{E}[\|\hat{\ell}-\ell\|] \approx \text{MAE}
\]
Then translate to:
- probability that predicted location falls outside correct isolation region
- expected fraction of incorrectly isolated buses/lines

If the paper gives confidence intervals or calibrated uncertainty, use them to compute “probability of correct region membership”.

### 5.4 Connect error to physical consequence (minimal but defensible)
At minimum, provide one of:
- **time impact**: earlier detection reduces sustained stress (use monotonic argument)
- **spatial impact**: localization error leads to wrong zone selection
- **risk score impact**: calibration quality affects probability of unsafe operation

Example statements (adapt to the task):
- “When localization error increases by \(\Delta d\), the chance of selecting the wrong isolation zone increases; therefore MAE is directly related to operational risk.”
- “Better recall at fixed false-alarm rate implies fewer missed faults, which reduces the duration of fault currents and associated thermal damage.”

### 5.5 Report threshold/operating point explicitly
For any metric that depends on a threshold (precision/recall, F1), specify:
- the chosen threshold \(\tau\)
- how it was selected (validation policy)
- sensitivity (performance change if \(\tau\) shifts)

---

## 6) Checklist before finalizing each paper note

- [ ] Defined \(x\), \(y\), and \(\hat{y}=f_\theta(x)\) precisely
- [ ] Explicit pipeline \(g(\cdot)\) from raw data to features
- [ ] Training derivation includes: data → features → model → loss → optimizer update → regularization → validation
- [ ] Mathematical expressions for loss/objective and at least one key architectural equation
- [ ] Intuition after every major equation/choice
- [ ] Metrics mapped to physical/grid impact (detection policy or action layer)

---

*End of Variant A framework.*