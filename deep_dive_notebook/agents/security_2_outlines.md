# Agent: security — Deep-dive outlines (anomaly objectives, thresholding, ROC/PR, robustness)

This file provides **3–5** notebook-ready deep-dive outlines to formalize an ML pipeline for **grid cybersecurity anomaly detection** and **resilient operation**. It is written to be compatible with a typical “detector → threshold → decision policy” workflow.

---

## Outline 1 — Reconstruction-loss objective for cyber-physical anomaly detection (AE/VAE)

### Problem formalization
Given nominal (clean) measurements \(x \sim p_0(x)\) and attacked measurements \(x \sim p_1(x)\), learn a model of nominality with parameters \(\theta\):
- **Encoder** \(q_\theta(z\mid x)\), **decoder** \(p_\theta(x\mid z)\)
- Or deterministic **encoder-decoder** \(z = g_\theta(x)\), \(\hat{x} = h_\theta(z)\)

### Deterministic autoencoder (AE)
Train on nominal-only:
\[
\min_\theta \ \mathbb{E}_{x\sim p_0}\left[\|x-\hat{x}_\theta\|_2^2\right]
\]
Anomaly score:
\[
s_{\text{AE}}(x)=\|x-\hat{x}_\theta\|_2^2
\]
Decision rule at time \(t\):
\[
\hat{y}_t = \mathbb{1}\{s_{\text{AE}}(x_t)>\tau\}
\]

### Variational autoencoder (VAE)
ELBO-based nominal likelihood surrogate:
\[
\mathcal{L}_{\text{VAE}}(\theta)=\mathbb{E}_{q_\theta(z\mid x)}[-\log p_\theta(x\mid z)]+\mathrm{KL}(q_\theta(z\mid x)\|p(z))
\]
Anomaly score using negative ELBO / reconstruction + KL:
\[
s_{\text{VAE}}(x)= -\log p_\theta(x)\approx \mathcal{L}_{\text{VAE}}(\theta)
\]

### Grid-specific nuance
- Normalize by measurement noise covariance \(R\) (so errors in noisy SCADA signals don’t dominate):
\[
s(x) = (x-\hat{x})^\top R^{-1}(x-\hat{x})
\]
- If \(x_t\) includes physical state estimates/residuals, reconstruction loss becomes aligned with *residual energy* used in classical BDD.

---

## Outline 2 — Contrastive objectives for learning attack-robust representations (InfoNCE / CPC)

### Why contrastive for security
Stealthy attacks may preserve marginal distributions of individual sensors. Contrastive learning instead targets **temporal/coherent structure**:
- Positive pairs: same time window under nominal perturbations / augmentations
- Negative pairs: mismatched time windows or different operating regimes

### Setup (time-series)
Let \(u_t\) be an embedding of a window \(x_{t-k:t}\):
\[
u_t = f_\theta(x_{t-k:t})
\]
Define a “paired” or “future” embedding \(v_t\) (e.g., \(v_t=f_\theta(x_{t-k+1:t+1})\) or via a separate head):
\[
v_t = g_\theta(x_{t-k+1:t+1})
\]

### InfoNCE loss
For a batch \(\{(u_i,v_i)\}\):
\[
\mathcal{L}_{\text{NCE}} = -\sum_i \log \frac{\exp(\mathrm{sim}(u_i,v_i)/\kappa)}{\sum_{j}\exp(\mathrm{sim}(u_i,v_j)/\kappa)}
\]
where \(\mathrm{sim}\) is cosine similarity and \(\kappa\) a temperature.

### Anomaly score from representation mismatch
At inference, compute:
\[
s_{\text{CL}}(x_{t-k:t}) = 1 - \max_{v\in \mathcal{N}(t)} \mathrm{sim}(u_t, v)
\]
or use a learned contrastive classifier’s logit/margin. Attacks that break physical consistency yield lower agreement \(\Rightarrow\) higher score.

### Grid-specific augmentations (augmentation operators)
- masking random sensors \(x^{(S)}\) within plausible fault-free ranges
- jitter based on PMU time-stamp / noise model
- topology-aware channel dropping (simulate missing sensors)
- regime augmentation: scale by operating-point-dependent normalization

---

## Outline 3 — Thresholding & evaluation: ROC/PR with imbalanced grid attack datasets

### Score distribution under nominal
Assume scores \(s(x)\) follow nominal distribution \(F_0\). Choose threshold \(\tau\) for target false alarm rate \(\alpha\):
\[
\Pr_{x\sim p_0}(s(x)>\tau)=\alpha
\]
Practical method:
- fit \(F_0\) empirically on a validation set (nominal only)
- set \(\tau\) as the \((1-\alpha)\)-quantile

### Thresholding under class imbalance
Attacks are rare, thus ROC can appear optimistic. Prefer **PR-AUC**:
\[
\text{Precision}=\frac{TP}{TP+FP}, \quad \text{Recall}=\frac{TP}{TP+FN}
\]
Use **cost-sensitive** metrics relevant to operators:
- cost of false alarm \(C_{FA}\)
- cost of missed attack \(C_{FN}\)

Bayes decision rule using calibrated \(p(\text{attack}\mid x)\):
\[
\text{decide attack if } \frac{p(\text{attack}\mid x)}{p(\text{no}\mid x)} > \frac{C_{FA}}{C_{FN}}
\]

### Grid context complication: operating points
Scores depend on load level, topology, and contingencies. Use conditional thresholding:
- \( \tau = \tau(\text{operating point } \omega)\)
- e.g., separate thresholds per regime cluster \(\omega\in\{\text{peak},\text{off-peak}\}\)

### ROC/PR computation details
- episode/window-level labeling: if any measurement in window is attacked, mark entire window positive
- use block bootstrap / time-aware splitting to avoid temporal leakage

---

## Outline 4 — Robustness & uncertainty estimation for resilient operation

### Objective: avoid catastrophic actions
A resilient controller should not only detect attacks but also:
- quantify uncertainty
- degrade gracefully (e.g., switch to conservative estimator)

### Uncertainty types
Let \(p_\theta(y\mid x)\) be detector output.
1) **Epistemic uncertainty** (model uncertainty): decreases with more/better data  
2) **Aleatoric uncertainty** (noise/variability): inherent in measurements

### MC-dropout / ensembles
For \(M\) stochastic forward passes:
\[
\hat{p}_m = p_\theta(y\mid x, \xi_m)
\]
Predictive mean:
\[
\bar{p} = \frac{1}{M}\sum_{m=1}^M \hat{p}_m
\]
Uncertainty via predictive entropy:
\[
H(\bar{p}) = -\sum_c \bar{p}_c \log \bar{p}_c
\]
Or mutual information (epistemic proxy):
\[
I(y,\theta\mid x) = H(\bar{p}) - \frac{1}{M}\sum_{m=1}^M H(\hat{p}_m)
\]

### Decision with uncertainty gating
Use a 2-stage policy:
1) compute attack probability \(p(\text{attack}\mid x)\)
2) only trigger high-cost actions if uncertainty low enough:
\[
\text{trigger resilience mode if } p(\text{attack}\mid x)>\tau_p \ \land\ H(\bar{p})<\tau_H
\]
Else remain in “monitoring” mode.

### Robustness to adversarial/stealthy attacks
Robustness goal: detector should resist attackers that attempt to keep scores low.
Training approaches to summarize in the notebook:
- adversarial training (optimize worst-case perturbation in measurement space)
- distributionally robust training (minimize risk under plausible attack distributions)
- contrastive objectives that enforce physical consistency constraints

---

## Outline 5 — Detection-to-control link: resilient operation pipeline (detector + switched estimator)

### Two-mode estimator architecture
- nominal estimator \( \hat{s}_{\text{nom}} \) for state \(s\)
- resilient estimator \( \hat{s}_{\text{res}} \) (robust filter / constrained estimator)

Detector output \(\hat{y}_t\) selects:
\[
\hat{s}_t =
\begin{cases}
\hat{s}_{\text{nom}}(x_{1:t}) & \hat{y}_t=\text{no-attack}\\
\hat{s}_{\text{res}}(x_{1:t}) & \hat{y}_t=\text{attack}
\end{cases}
\]

### Formal objective for end-to-end resilience
Minimize control/estimation loss under both regimes:
\[
\min \ \mathbb{E}\left[\mathcal{C}(\hat{s}_t,s_t)\right]
\]
with constraint on false alarms:
\[
\Pr(\hat{y}=\text{attack}\mid \text{nominal}) \le \alpha
\]
This links anomaly detection metrics directly to operational losses.

### Evaluation
Report both:
- detector ROC/PR **and** downstream resilience metrics:
  - estimation error under attack
  - constraint violations (voltage limits, thermal limits)
  - recovery time after attack ends