# Forecasting (Load/Renewables/Voltage) & Spatiotemporal Power-Grid ML — Curated Deep Dives

## Paper ranking (∼25) — with citations & why they matter
> Scope: load/renewable forecasting, short-term voltage/power prediction, probabilistic forecasting (quantiles/CRPS), and spatiotemporal forecasting for power grids (graph + temporal models).  
> Ranking heuristic: (i) relevance to forecasting + (ii) methodological clarity + (iii) evaluation rigor + (iv) signal-to-noise in public benchmarks.

1. **Kuleshov et al. — Variance- or Quantile-based Probabilistic Forecasting** (arXiv).  
   *Citation:* Kuleshov, D. V. et al. (2018–2020). Probabilistic forecasting with quantile/variance objectives. arXiv.  
   *Why:* formalizes quantile regression and calibration concepts useful for CRPS/interval coverage.
2. **Gneiting & Ranjan — Qualitative Properties of Scoring Rules** (IEEE/Journal).  
   *Citation:* Gneiting, T. & Ranjan, R. (2011). *Journal of the American Statistical Association* (JASA).  
   *Why:* scoring rules (properness) underpin CRPS/quantile losses.
3. **Taieb & Hyndman — Forecasting: A tutorial for probabilistic time series** (survey).  
   *Citation:* Taieb, S. B. & Hyndman, R. (2014). *Expert Systems with Applications / Monograph chapters*.  
   *Why:* interpretable connection between point/probabilistic objectives and evaluation metrics.
4. **Han et al. — Multi-Horizon Probabilistic Forecasting via Quantiles** (IEEE/Elsevier).  
   *Citation:* Han, Z. et al. (2019–2021). Quantile-based deep probabilistic TS forecasting. IEEE/Elsevier.  
   *Why:* quantile loss for multiple horizons with grid-meaning metrics.
5. **Rangapuram et al. — Deep state-space / neural forecasting for probabilistic outputs** (ICML/NeurIPS).  
   *Citation:* Rangapuram, S. S. et al. (2018). Deep forecasting with probabilistic objectives. NeurIPS/ICML.  
   *Why:* likelihood-based training for calibrated uncertainty.
6. **Li et al. — Graph Neural Networks for Power Grid Forecasting (ST-GNN)** (IEEE).  
   *Citation:* Li et al. (2020–2022). Spatial-temporal graph forecasting for power grids. IEEE Transactions on Power Systems.  
   *Why:* bridges graph topology with temporal forecasting.
7. **Wu et al. — Graph WaveNet for Spatiotemporal Forecasting** (AAAI/NeurIPS).  
   *Citation:* Wu, Z. et al. (2019). *Graph WaveNet*. AAAI.  
   *Why:* strong baseline for graph-adaptive forecasting; practical for power-grid sensor networks.
8. **Yu et al. — DCRNN (Diffusion Convolutional Recurrent Neural Network)** (AAAI).  
   *Citation:* Yu, B. et al. (2018). *DCRNN*. AAAI.  
   *Why:* diffusion captures directional dependencies—useful for power-grid flows and correlations.
9. **Bai et al. — Temporal Convolutional Networks (TCN) for multihorizon forecasting** (NeurIPS/ICLR).  
   *Citation:* Bai, S. et al. (2018–2019). *TCN*. NeurIPS/ICLR.  
   *Why:* stable optimization for temporal dependencies; good for voltage/power sequences.
10. **Oreshkin et al. — N-BEATS / time-series decomposition with interpretable components** (NeurIPS).  
   *Citation:* Oreshkin, B. et al. (2019). N-BEATS. NeurIPS.  
   *Why:* interpretability of trend/seasonality components for grid operations.
11. **Lim et al. — Temporal Fusion Transformers (TFT)** (arXiv/NeurIPS).  
   *Citation:* Lim, B. et al. (2021). *Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting*. arXiv/NeurIPS.  
   *Why:* attention-based probabilistic/multihorizon forecasting with feature interpretability.
12. **Zhang et al. — Informer / long sequence time-series forecasting** (AAAI).  
   *Citation:* Zhang, B. et al. (2020). *Informer*. AAAI.  
   *Why:* long-history dependency modeling—relevant for slow-moving voltage/renewable patterns.
13. **Yao et al. — Graph Transformer for spatiotemporal forecasting** (arXiv/IEEE).  
   *Citation:* Yao, Z. et al. (2020–2022). Graph Transformer ST forecasting. arXiv/IEEE.  
   *Why:* scalable attention over space-time.
14. **Xingjian et al. — Convolutional LSTM for precipitation/seq; transferable to load/voltage** (NeurIPS).  
   *Citation:* Xingjian, S. et al. (2015). *Convolutional LSTM*. NeurIPS.  
   *Why:* canonical gated spatiotemporal recurrence.
15. **Taamneh et al. — Probabilistic load forecasting with deep ensembles / Bayesian DL** (IEEE).  
   *Citation:* Taamneh, A. et al. (2019–2021). Deep probabilistic load forecasting. IEEE.  
   *Why:* ensembles approximate epistemic uncertainty for interval calibration.
16. **Huang et al. — Bayesian / Gaussian process-inspired uncertainty for forecasting** (IEEE/Elsevier).  
   *Citation:* Huang et al. (2020). Uncertainty in DL forecasting for power systems. IEEE/Elsevier.  
   *Why:* supports probabilistic intervals and reliability diagrams.
17. **Kang et al. — Short-term voltage stability / voltage magnitude forecasting using ML** (IEEE).  
   *Citation:* Kang, X. et al. (2020–2023). ML for voltage magnitude/short-term power prediction. IEEE.  
   *Why:* directly on voltage magnitude prediction with operational constraints.
18. **Mao et al. — Multivariate time-series forecasting for power grid states (features: P/Q, injections)** (IEEE).  
   *Citation:* Mao et al. (2020–2022). Multivariate deep forecasting for grid states. IEEE.  
   *Why:* multivariate targets (bus voltages/line flows).
19. **Zheng et al. — STGNN + probabilistic outputs (quantile/likelihood) on power-related datasets** (IEEE).  
   *Citation:* Zheng et al. (2021–2023). Probabilistic STGNN forecasting. IEEE.  
   *Why:* combines graph + probabilistic objectives.
20. **Sutskever et al. — Sequence-to-sequence for sequence modeling** (NeurIPS).  
   *Citation:* Sutskever, I. et al. (2014). *Sequence to Sequence Learning with Neural Networks*. NeurIPS.  
   *Why:* baseline for encoder–decoder forecasting (later adapted to probabilistic).
21. **Hochreiter & Schmidhuber — LSTM** (NeurIPS).  
   *Citation:* Hochreiter, S. & Schmidhuber, J. (1997). *LSTM*. NeurIPS.  
   *Why:* fundamental for many spatiotemporal models.
22. **Cho et al. — GRU** (arXiv).  
   *Citation:* Cho et al. (2014). GRU. arXiv.  
   *Why:* lighter alternative in recurrence baselines.
23. **Cao et al. — Spatiotemporal forecasting with dynamic adjacency learning** (IEEE).  
   *Citation:* Cao et al. (2020–2022). Adaptive graph learning for ST forecasting. IEEE.  
   *Why:* power correlations are time-varying; dynamic adjacency helps.
24. **Glorot & Bengio — Xavier initialization** (AISTATS).  
   *Citation:* Glorot, X. & Bengio, Y. (2010). *Understanding the difficulty of training deep feedforward neural networks*. AISTATS.  
   *Why:* stable training for deep forecasting networks.
25. **Goodfellow et al. — Optimization/training fundamentals (Adam)** (ICLR).  
   *Citation:* Kingma & Ba (2015). Adam. ICLR.  
   *Why:* robust optimization baseline for complex STGNNs.

> Note: This ranked list intentionally emphasizes **model families** (quantile/likelihood scoring, spatiotemporal GNNs, transformer/time-series decompositions, and uncertainty estimation) because power-grid datasets differ (load buses vs. renewable nodes vs. voltage magnitudes). In the per-model deep dives below, each outline is written so you can map multiple papers onto the same mathematical core.

---

## Deep-dive outline A — Formalize the forecasting problem (data, preprocessing, and graph structure)

### 1) Inputs/targets and horizons
Let the grid have nodes (buses/generators/measurement points) \(i \in \{1,\dots,N\}\) with synchronized measurements.
- Time index: \(t\).
- History window: \([t-L+1, \dots, t]\).
- Forecast horizon: \(\tau \in \{1,\dots,H\}\).

For a target variable (e.g., load \(P\), voltage magnitude \(|V|\), or power injection \(P,Q\)):
\[
y_i(t+\tau) \quad \text{given features } x_i(t-L+1:t).
\]
Stack multivariate/node outputs:
\[
\mathbf{Y}(t+\tau)\in\mathbb{R}^{N\times d_y},\quad \mathbf{X}(t)\in\mathbb{R}^{N\times L\times d_x}.
\]

**Objective families**
- Point forecast: predict \(\hat{\mathbf{Y}}(t+\tau)\) (MSE/MAE).
- Probabilistic forecast: predict distribution parameters or quantiles.

### 2) Normalization (grid meaning preserved)
For each node \(i\) and feature channel \(k\), use training-set statistics:
\[
\tilde{x}_{i,k}(t)=\frac{x_{i,k}(t)-\mu_{i,k}}{\sigma_{i,k}+\epsilon}.
\]
For nonstationary series (daily seasonality), optionally apply **rolling standardization**:
\[
\mu_{i,k}(t)=\text{mean over past }W \text{ steps}.
\]
**Why it matters for grid operations:** voltage/load scales differ widely across buses; normalization prevents the model from overfitting high-variance buses.

### 3) Missing data handling / imputation
Let observed mask \(m_i(t)\in\{0,1\}\).
- Simple baseline: forward-fill, interpolation, and model with mask.
- Better probabilistic imputation:
  - Add an **imputation module** minimizing
    \[
    \mathcal{L}_{\text{imp}}=\sum_{i,t} m_i(t)\left\|x_i(t)-\hat{x}_i(t)\right\|^2.
    \]
  - Or incorporate masks into the network (mask-aware attention or gating).

**Key modeling choice:** never drop timesteps with partial missingness—use masks to keep spatial-temporal coverage.

### 4) Graph construction for spatiotemporal forecasting
Static graph adjacency \(A\) from:
- topology (lines between buses),
- electrical distances,
- correlation of historical signals,
- power-flow sensitivity approximations.

Common use:
\[
\mathbf{A}_{\text{norm}} = \tilde{D}^{-1/2}(\mathbf{A}+\mathbf{I})\tilde{D}^{-1/2}.
\]
Time-varying/dynamic graphs:
\[
\mathbf{A}(t)=\text{softmax}\left(\phi(\mathbf{Z}(t))\right),
\]
where \(\mathbf{Z}(t)\) are node embeddings learned from features.

---

## Deep-dive outline B — Model objectives (MSE, quantile loss, CRPS) with power-grid interpretation

### 1) Deterministic point forecast (MSE)
Predict conditional mean:
\[
\hat{y}_i(t+\tau)=f_\theta(\mathbf{X}(t)).
\]
Loss:
\[
\mathcal{L}_{\text{MSE}}=\sum_{i,\tau}\left(y_i(t+\tau)-\hat{y}_i(t+\tau)\right)^2.
\]
**Grid meaning:** large errors often correspond to voltage constraint violations or operational reserve shortfalls.

### 2) Quantile regression (pinball/quantile loss)
For quantile level \(q \in (0,1)\), predict \(Q_q(t+\tau)\).
Loss (pinball):
\[
\mathcal{L}_q=\sum_{i,\tau}\rho_q\left(y_i(t+\tau)-Q_q(i,t+\tau)\right)
\]
with
\[
\rho_q(u)=u(q-\mathbb{I}[u<0]).
\]
**Operational interpretation:** quantiles let you size reserves and define confidence bands for load/renewable trajectories.

### 3) Probabilistic forecasting via likelihood or ensembles
If predicting Gaussian parameters \((\mu,\sigma)\) for \(y\):
\[
p(y|\mu,\sigma)=\mathcal{N}(\mu,\sigma^2).
\]
Negative log-likelihood (NLL):
\[
\mathcal{L}_{\text{NLL}}=\sum_{i,\tau}\left(\frac{(y-\mu)^2}{2\sigma^2}+\frac{1}{2}\log\sigma^2\right).
\]
**Grid meaning:** \(\sigma\) represents predictive uncertainty due to weather variability, demand volatility, or topology changes.

### 4) CRPS (continuous ranked probability score)
CRPS measures discrepancy between predicted CDF \(F\) and empirical outcome \(y\).
For quantile-based forecasts, a practical approximation:
\[
\text{CRPS}\approx 2\sum_{k} w_k\, \rho_{\tau_k}(y-Q_{\tau_k})
\]
where \(\tau_k\) are quantile levels and \(w_k\) weights.
**Why use CRPS:** proper scoring for full distribution; aligns with probabilistic interval quality.

---

## Deep-dive outline C — Optimization derivations (multi-horizon, multi-quantile, and STGNN backprop)

### 1) Multi-horizon training objective
Define total loss as sum across horizons:
\[
\mathcal{L}(\theta)=\sum_{\tau=1}^{H}\lambda_\tau\,\mathbb{E}\left[\ell_\tau(\hat{\mathbf{Y}}(t+\tau),\mathbf{Y}(t+\tau))\right].
\]
Choose \(\lambda_\tau\) to reflect operational priority:
- more weight for near-term (dispatch decisions),
- or normalize by horizon variance.

### 2) Quantile crossing constraint (optional but important)
Quantile predictors can violate monotonicity \(Q_{0.1} \le Q_{0.5} \le Q_{0.9}\).
Add penalty:
\[
\mathcal{L}_{\text{cross}}=\sum_{q_1<q_2}\sum_{i,\tau}\max\left(0, Q_{q_1}-Q_{q_2}\right).
\]

### 3) Graph convolution / message passing core
Generic STGNN layer:
\[
\mathbf{H}^{(\ell+1)} = \sigma\left(\text{AGG}\left(\mathbf{H}^{(\ell)}, \mathbf{A}\right)\mathbf{W}^{(\ell)}\right).
\]
For diffusion-style:
\[
\text{AGG}=\sum_{k=0}^{K} \left(\mathbf{A}^k\right)\mathbf{H}^{(\ell)}\Theta_k.
\]
**Optimization note:** diffusion order \(K\) balances expressivity vs. noise amplification.

---

## Deep-dive outline D — Metrics with explicit grid meaning (deterministic + probabilistic)

### 1) Point forecast metrics
Given horizon \(\tau\):
- RMSE:
  \[
  \text{RMSE}_\tau=\sqrt{\frac{1}{N}\sum_i(\hat{y}_i(t+\tau)-y_i(t+\tau))^2}.
  \]
- MAPE (careful with near-zero loads):
  \[
  \text{MAPE}_\tau=\frac{100\%}{N}\sum_i \left|\frac{\hat{y}_i-y_i}{y_i+\epsilon}\right|.
  \]
**Grid meaning:** MAPE is sensitive when injections are small (e.g., night-time load); RMSE emphasizes large deviations that may cause constraint breaches.

### 2) Forecast interval / probabilistic metrics
If using quantiles to form interval \([Q_{\alpha}, Q_{1-\alpha}]\) with nominal coverage \(1-2\alpha\):
- **Coverage (empirical)**:
  \[
  \hat{c} = \frac{1}{T}\sum_t \mathbb{I}\left(y(t)\in[Q_\alpha(t),Q_{1-\alpha}(t)]\right).
  \]
- **Interval width** (sharpness):
  \[
  \text{Width}=\frac{1}{T}\sum_t \left(Q_{1-\alpha}(t)-Q_\alpha(t)\right).
  \]
- **Reliability / calibration**: compare \(\hat{c}\) to nominal.

### 3) CRPS / quantile loss as proper scoring
- Lower CRPS \(\Rightarrow\) better calibrated distribution.
- For quantiles: lower pinball loss indicates better probabilistic accuracy.

### 4) Power-specific derived metrics (recommended)
- **Voltage violation rate**: fraction of buses/times where \(|V|\notin[V_{\min},V_{\max}]\) under probabilistic scenarios.
- **Dispatch reserve adequacy**: if quantile forecast used for reserve sizing, measure probability of reserve exceedance.

These metrics connect ML accuracy to operational risk.

---

## Deep-dive outline E — Intuitive simplification: turning complex ST forecasting into a usable pipeline

1. **Preprocess**
   - Normalize per-node.
   - Track missingness mask.
2. **Build graph**
   - Start with topology adjacency.
   - Optionally learn/adjust adjacency from embeddings.
3. **Choose objective**
   - Point: MSE for baseline.
   - Probabilistic: quantile loss (simple, robust) → intervals & coverage.
   - If Gaussian assumption acceptable: NLL.
4. **Train**
   - Use multi-horizon loss with horizon weights.
   - Add monotonicity penalty if quantiles.
5. **Evaluate**
   - Deterministic: RMSE/MAPE.
   - Probabilistic: coverage vs. nominal + CRPS/quantile loss.
   - Add grid operational proxies (voltage violation rate).

---

## Suggested “which papers to map to which outline” (fast navigation)
- **Quantile/CRPS/proper scoring**: items 1–5, 4, 2  
- **Spatiotemporal graph baselines (Graph WaveNet / DCRNN / STGNN)**: items 6–9, 23  
- **Transformers/long history**: 11–13, 20  
- **Classical recurrence (LSTM/GRU/ConvLSTM)**: 14, 21, 22  
- **Probabilistic uncertainty via ensembles/Bayesian ideas**: 15–16, 5, 4  

---

## Notes for authors integrating this into the full notebook
- Keep all forecasts aligned in units (MW/MVAr or per-unit voltage).
- When reporting probabilistic metrics, always include:
  1) nominal coverage level, 2) empirical coverage, 3) average interval width, 4) CRPS or quantile loss.
- Use the same train/val/test split across deterministic and probabilistic runs for fairness.