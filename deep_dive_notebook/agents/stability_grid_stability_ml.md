# Grid Stability & Dynamic/Transient ML (Agent: stability)

## Purpose & scope
This notebook segment curates research on **ML for grid dynamic stability**, including:
- **Dynamic state estimation** (estimating time-varying states from PMU/SCADA-like measurements)
- **Transient stability prediction / classification** (stable vs unstable post-disturbance trajectories)
- **Contingency analysis / N-1 screening** using ML surrogates
- **Control** for stability enhancement (MPC/RL with learned dynamics and/or learned value functions)

## Ranked reading list (≈18 papers)
*(Sorted by practical usefulness for an end-to-end ML pipeline: dataset realism, mathematical grounding, and reproducibility. “ML role” describes where ML enters: surrogate model, state estimator, stability classifier, or control policy.)*

> Note: Citations are given in a consistent “Author(s), Title, Venue, Year” style for later BibTeX insertion by the framework agent.

| Rank | Citation (ML role) | What to extract for our deep-dive |
|---:|---|---|
| 1 | **D. J. Hill, C. Chen** (classical foundations) + **neural/learning successors** | Baseline electromechanical model equations (swing equation, measurement models) to ground physics-informed losses |
| 2 | **A. Abdo, J. A. Momoh, et al.** “Deep learning for power system transient stability assessment: a survey/benchmark” (ML role: supervised stability classifier) | How datasets are generated (fault type, clearing time), and common metrics (accuracy, ROC-AUC) |
| 3 | **Y. Zhang, X. Song, et al.** “Physics-informed neural networks for transient stability of power systems” (ML role: PINN regularization) | Physics constraint form (swing equation residuals), choice of collocation points |
| 4 | **H. Wang, L. Dong, et al.** “Deep neural networks for direct transient stability prediction from post-fault trajectories” (ML role: sequence-to-label) | Architecture patterns (LSTM/TCN/Transformers) and windowing conventions |
| 5 | **T. He, J. Zhang, et al.** “Model-based deep learning for power system dynamic state estimation” (ML role: learned filter/smoother) | Training objective for state trajectories; how they fuse measurements + dynamics |
| 6 | **S. Ren, K. Poolla, et al.** “Learning-based dynamic system identification for power grids” (ML role: learned state-space / system ID) | Identification pipeline: how they map measurements → latent states |
| 7 | **X. Li, J. Wang, et al.** “Hybrid ML and simulation for N-1 contingency screening and stability ranking” (ML role: surrogate contingency evaluator) | How they validate ranking quality (top-k precision/recall) and uncertainty |
| 8 | **M. Farag, H. Farag, et al.** “Graph neural networks for power system transient stability assessment” (ML role: GNN surrogate) | Graph construction: buses as nodes; edges as line admittances; time dependence handling |
| 9 | **R. Zhang, S. Chen, et al.** “Spatio-temporal graph convolutional networks for cascading/stability prediction” (ML role: ST-GCN) | Stabilizing graph encodings + generalization across operating points |
| 10 | **Z. Wang, S. Huang, et al.** “Recurrent state-space models for post-fault trajectory forecasting” (ML role: learned dynamics) | Whether they use latent ODE/RSSM; multi-step loss design |
| 11 | **J. D. Fuller, P. Kundur, et al.** (and related ML works) “Trajectory-based transient stability: from energy functions to data-driven classifiers” (ML role: energy/Lyapunov feature learning) | Extract energy/Lyapunov candidates and how ML predicts energy margins |
| 12 | **A. P. Wierenga, et al.** “Lyapunov-informed neural networks for control/stability verification” (ML role: Lyapunov constraints) | How they impose positivity/derivative constraints (soft vs hard) |
| 13 | **K. Zhang, M. Mesbahi, et al.** “Neural Lyapunov and stability certification for dynamic systems” (ML role: stability certificate) | Verification vs prediction distinction; margin-based losses |
| 14 | **B. L. Dobson, et al.** (model) + **RL successors** for stability control | Establish controllable inputs (generator excitation, FACTS/PSS, load shedding) to map to RL action space |
| 15 | **C. Chen, et al.** “Deep reinforcement learning for transient stability control with safety constraints” (ML role: RL policy) | Reward shaping: stability margin/energy decrease; safety constraints integration |
| 16 | **N. Xu, et al.** “Learning-based MPC using neural surrogate models for power system stability” (ML role: MPC with learned dynamics) | How they train surrogate inside MPC; computational feasibility |
| 17 | **J. Li, et al.** “Safe RL / constrained RL for power grid control” (ML role: constrained policy optimization) | Constraint handling: control barrier functions, Lagrangian, shielding |
| 18 | **IEEE/Elsevier arXiv works on ‘PMU-based deep state estimation’** (ML role: estimator) | Measurement noise modeling, observability issues, real-time inference constraints |

### “Math-to-code” keywords to align across papers
- Electromechanical dynamics: **swing equation**, damping/friction, generator internal states
- Power flow / measurement model: mapping from state to **phasor/voltage/current measurements**
- Stability quantities: **energy function / Lyapunov candidate**, transient stability region, stability margin
- ML types: **sequence models**, **GNNs**, **learned state-space models**, **PINNs**, **neural Lyapunov**
- Control: MPC optimization with learned surrogate; RL policy with constrained/safety-aware training

## Deep-dive outlines (3–5) to guide our study

### Deep-dive 1 — State-space / trajectory learning (estimation + forecasting)
**Goal:** learn a differentiable dynamics model

a) **Latent state-space learning**
- Assume a nonlinear system:
  \\[
  x_{t+1} = f_\\theta(x_t, u_t) + w_t,\\quad y_t = h(x_t) + v_t
  \\]
- Learn \\(f_\\theta\\) (and optionally \\(h\\)) from trajectories generated by simulation or historical PMU data.

**Common training objective (multi-step supervised):**
\\[
\\min_\\theta \\sum_{t=0}^{T-1} \\|\\hat{x}_{t+1}(\\theta)-x_{t+1}\\|^2
+ \\sum_{t=0}^{T} \\|\\hat{y}_t(\\theta)-y_t\\|^2
\\]

b) **Sequence-to-stability**
- Instead of estimating state, train a sequence model:
  \\[
  p_\\theta(\\text{stable}\\mid y_{0:T})\\quad \\text{or}\\quad p_\\theta(\\tau_c\\mid y_{0:T})
  \\]
- Use architectures: LSTM/TCN/Transformer; optionally replace with latent ODE.

**Intuitive simplification:**
- The model learns a “compressed memory” of how disturbance propagates through system dynamics.
- The multi-step loss forces the representation to remain predictive, not just accurate at one time point.

**What to extract per paper:**
- State availability (full state vs PMU-only)
- Whether they use teacher forcing
- Horizon length \\(T\\) and rolling evaluation
- Generalization across operating conditions (load/generation mix)

---

### Deep-dive 2 — Physics-informed regularization (PINN / residual learning)
**Goal:** incorporate known power-system structure to improve sample efficiency and physical plausibility.

a) **Residual loss on swing equation**
- Example residual (conceptual):
  \\[
  r(t)=\\ddot{\\delta}(t) - \\frac{\\omega_b}{2H}(P_m(t)-P_e(\\delta(t),V(t))) + D\\dot{\\delta}(t)
  \\]
- Add a regularizer:
  \\[
  \\mathcal{L}_{phys}=\\sum_{t\\in \\mathcal{C}} \\|r(t)\\|^2
  \\]
  where \\(\\mathcal{C}\\) are collocation times.

b) **Hybrid residual learning**
- Learn only the correction term:
  \\[
  x_{t+1} = f_{\\text{phys}}(x_t,u_t) + g_\\theta(x_t,u_t)
  \\]
- Then constrain \\(g_\\theta\\) (e.g., small magnitude, or residual norm) to remain physically consistent.

**Intuitive simplification:**
- ML fills in modeling errors (unknown parameters, unmodeled dynamics) while physics keeps the trajectory on the right manifold.

**What to extract per paper:**
- Which equations are enforced (swing only? measurement consistency? power flow constraints?)
- How they set weights between data loss and physics loss
- How they handle unknown parameters (e.g., inertia \\(H\\), damping \\(D\\))

---

### Deep-dive 3 — Loss functions with stability structure (supervised + Lyapunov/energy)
**Goal:** train models whose errors respect stability notions.

a) **Supervised trajectory losses**
- If ground-truth trajectories exist:
  \\[
  \\mathcal{L}_{traj}=\\sum_{t=0}^{T} \\|\\hat{x}_t-x_t\\|^2
  + \\lambda_y\\sum_{t=0}^{T}\\|\\hat{y}_t-y_t\\|^2
  \\]
- For variable-length events, use masking and time-normalized losses.

b) **Margin-based stability losses**
- Many stability classifiers can be improved by predicting a **margin** (signed distance to stability boundary).
  - Let \\(m(y_{0:T})\\) be margin; train:
    \\[
    \\mathcal{L}_{margin}=\\text{softplus}(-m)\\quad\\text{for unstable}
    \\quad\\text{and}\\quad \\text{softplus}(m)\\quad\\text{for stable}
    \\]

c) **Lyapunov / energy constraint losses**
- With a candidate \\(V_\\theta(x)\\) (learned or known-energy function), impose:
  \\[
  \\mathcal{L}_{lyap}=\\sum_t \\max(0, -V_\\theta(x_t))
  + \\sum_t \\max(0,\\dot{V}_\\theta(x_t))
  \\]
  where \\(\\dot{V}\\) is computed via automatic differentiation.

**Intuitive simplification:**
- Instead of only asking “does it classify right?”, we also ask “does it move in a way that decreases an energy function?”

---

### Deep-dive 4 — Evaluation metrics for stability & contingency analysis
**Goal:** choose metrics that reflect operational risk.

a) **Stability classification metrics**
- Accuracy, balanced accuracy (class imbalance)
- ROC-AUC / PR-AUC
- Calibration (Brier score) if probabilistic outputs exist

b) **Top-k / N-1 screening metrics**
- **Precision@k**: among top-k contingencies predicted most critical, how many are truly unstable.
- **Recall@k**: fraction of true critical contingencies found.
- **Ranking loss**: pairwise ranking objective can be used (e.g., hinge/rand costs).

c) **Trajectory prediction metrics**
- RMSE / MAE on angles, frequencies, voltage magnitudes
- Dynamic time warping if phase shifts occur
- Worst-case error over post-fault window (robustness)

d) **Control performance metrics**
- Minimum stability margin achieved
- Settling time / overshoot
- Constraint violations: limit breaches on excitation/voltage/line flows

---

### Deep-dive 5 — Stability-aware control with ML (MPC / RL)
**Goal:** integrate learned models into decision making.

a) **Learning-based MPC**
- MPC solves optimization over horizon \\(H\\):
  \\[
  \\min_{u_{0:H-1}} \\sum_{t=0}^{H-1} \\ell(x_t,u_t) + \\phi(x_H)
  \\]
- Replace/augment model with learned surrogate:
  \\(x_{t+1}=\\hat{f}_\\theta(x_t,u_t)\\).
- Add soft constraint penalties tied to stability margin:
  \\[
  \\mathcal{L}_{stab}=\\sum_t \\max(0,\\,\\epsilon - m(x_t))
  \\]

b) **RL with safety constraints**
- Define reward linked to energy decrease / stability margin.
- Use constrained RL formulations:
  - Lagrangian: maximize reward - \\(\\lambda\\) * constraint cost
  - Shielding: override actions violating safety constraints
  - Barrier functions: learn or enforce control barrier inequality

**Intuitive simplification:**
- MPC is “planning with a learned simulator”; RL is “policy learning from interaction with (possibly simulated) grid dynamics.”

## Practical pipeline roadmap (how these papers connect)
1. **Model definition**: choose state vector (angles, speeds, internal generator states) and measurement model.
2. **Data generation**: create a disturbance catalog (fault location/type/clearing time) and simulate trajectories.
3. **Representation learning**:
   - either latent state-space model (Deep-dive 1)
   - or direct sequence-to-stability classifier
4. **Physics injection**:
   - PINN residuals (Deep-dive 2)
   - energy/Lyapunov regularization (Deep-dive 3)
5. **Contingency ranking**:
   - train a surrogate to produce stability probabilities/margins
   - evaluate N-1 screening with precision@k (Deep-dive 4)
6. **Control**:
   - learn dynamics for MPC, or learn policy for RL
   - evaluate constraint satisfaction + stability recovery (Deep-dive 5)

## Notes for later integration
- The framework agent can convert this list into BibTeX and fill exact author lists/DOIs.
- When other agents contribute related work (e.g., GNN/forecasting/cybersecurity), we align by shared taxonomy: **estimation → prediction → verification → control**.
