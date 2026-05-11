# Agent: security — Grid Cybersecurity ML & Anomaly Detection (Ranked Papers)

## Scope & keywords
**Grid cybersecurity** and **cyber-physical attack detection** using ML/Deep learning, focusing on:
- anomaly detection in SCADA/AMI/PMU streams
- load/measurement tampering detection
- resilient operation / attack-resilient learning

### Typical problem setting (used across papers)
Let the system produce measurements \(x_t \in \mathbb{R}^d\) (SCADA points, PMU features, load profiles, etc.). Under nominal operation: \(x_t \sim p_0\). Under an attack: \(x_t \sim p_1\). The ML goal is to estimate a detector \(f_\theta(x_{1:T})\) that yields:
- **event-level decision**: \( \hat{y} \in \{\text{no-attack}, \text{attack}\}\)
- often **uncertainty**: \(p(\text{attack}\mid x)\) or calibrated confidence

---

## Ranked citation list (≈18 papers)
> Ranking heuristic: (i) direct relevance to grid cyber-physical attacks + anomaly detection, (ii) dataset/realism, (iii) methodological novelty (reconstruction/contrastive/sequential/uncertainty), (iv) citation reputation (IEEE/T-PWRS/Elsevier/ Springer / arXiv).

### 1) "Deep Learning for Cybersecurity in Smart Grid: A Survey"
- **Citation**: (Survey; replace with exact authors/title/year/venue in final integration)
- **Why high**: Gives taxonomy of ML detectors, including autoencoders/RNNs and data sources. Useful to anchor the design space.
- **Use in notebook**: map each method to our anomaly objective + thresholding/ROC/PR protocol.

### 2) Stealthy data injection & detection using learning (FDIA / residual-based detection)
- **Citation**: (Select a canonical IEEE paper on FDIA and residual-based detection + ML augmentation)
- **Why**: Stealthy attacks challenge naive thresholding; ML can learn residual distributions conditioned on operating points.
- **Use**: formalize *attack labels weakly supervised vs fully supervised*, and evaluate ROC/PR under class imbalance.

### 3) False Data Injection Attack Detection via Deep Learning on SCADA/PMU Residuals
- **Citation**: IEEE/Elsevier paper on DL-based FDIA detection using residuals or feature engineering from state estimation.
- **Why**: Bridges classical BDD / residual tests with learned scoring functions.
- **Use**: anomaly score \(s(x)\) derived from reconstruction error, with calibration.

### 4) PMU-based cyber-physical attack detection using LSTM / GRU + anomaly scores
- **Citation**: IEEE paper on sequential ML for PMU streaming.
- **Why**: PMU features have temporal dynamics; sequential models better capture correlations.
- **Use**: reconstruction/contrastive objectives on \(x_{t-k:t}\).

### 5) Autoencoder/Variational Autoencoder for smart grid anomaly detection (data/measurement attacks)
- **Citation**: IEEE/ Springer paper on AE/VAE anomaly detection for grid cyber or system anomalies.
- **Why**: Clear reconstruction-loss objective and thresholding to detector.
- **Use**: derive \(s(x)=\|x-\hat{x}\|^2\) or ELBO-based scoring; discuss ROC/PR.

### 6) Contrastive learning for anomaly detection in cyber-physical systems
- **Citation**: arXiv/IEEE paper on contrastive anomaly detection (CPC/SimCLR-style) adapted to security/industrial data.
- **Why**: Contrastive losses directly learn invariances under nominal behavior; anomalies break latent alignment.
- **Use**: formalize objective with positive/negative pairs and scoring via embedding distance.

### 7) Bayesian / uncertainty-aware anomaly detection in smart grids
- **Citation**: IEEE/Elsevier on Bayesian deep learning, MC-dropout, ensembles for anomaly confidence.
- **Why**: Resilient operation needs *actionable uncertainty* (avoid false alarms & missed attacks).
- **Use**: robust decision rule with predictive entropy / mutual information.

### 8) Robust learning under adversarial perturbations (security-aware robustness)
- **Citation**: IEEE paper on adversarial ML robustness for power systems measurements.
- **Why**: ML detectors can themselves be fooled; robust training improves resilience.
- **Use**: incorporate adversarial training / distributionally robust discussion at high level.

### 9) Graph neural networks (GNN) for attack detection using grid topology
- **Citation**: IEEE paper applying GNN/GCN to cyber-physical anomaly detection.
- **Why**: Encodes physical connectivity; helps generalization across buses/operating points.
- **Use**: modeling \(x_t\) on graph nodes and learning anomaly scores per subgraph.

### 10) Federated learning for privacy-preserving cyber-attack detection
- **Citation**: IEEE/Elsevier on FL for anomaly detection in smart grid environments.
- **Why**: Grid cybersecurity requires data privacy and distributed sensing.
- **Use**: mention distribution shift across utilities; uncertainty under FL.

### 11) GAN-based anomaly detection / rare-event augmentation for attack detection
- **Citation**: IEEE/Elsevier paper using GAN or synthetic data for cyber-attack detection.
- **Why**: Attack labels are rare; generating plausible attacked samples can help.
- **Use**: careful discussion of synthetic-data bias and evaluation.

### 12) Explainable ML for FDIA detection (feature attribution / saliency)
- **Citation**: IEEE paper on XAI for power system cyber-attack detectors.
- **Why**: Operators need interpretability to isolate compromised sensors.
- **Use**: integrate with our detection score (post-hoc attribution).

### 13) Ensemble learning for intrusion/anomaly detection in industrial control systems (ICS)
- **Citation**: IEEE paper (often ICS general, but applicable to grid-like telemetry).
- **Why**: Ensembles often improve calibrated probabilities for ROC/PR.
- **Use**: thresholding + reliability diagrams.

### 14) Threat modeling + detection for AMI / load profile tampering
- **Citation**: IEEE/Elsevier paper on load shape manipulation and detection using ML.
- **Why**: Directly matches **load profile tampering detection**.
- **Use**: treat attacked load as time series; apply reconstruction/forecast residuals.

### 15) ML for tampered load / energy theft detection via sequence modeling
- **Citation**: Springer/Elsevier paper on LSTM/Transformer-based theft detection.
- **Why**: Temporal consumption patterns can reveal subtle tampering.
- **Use**: combine forecasting residuals with anomaly scores.

### 16) Resilient state estimation under cyber attacks (learning-guided robust estimation)
- **Citation**: IEEE paper integrating ML with robust estimation (e.g., robust filters / resilient estimation).
- **Why**: Links detection to resilient operation (not just detection).
- **Use**: decision-dependent pipeline: if detected, switch estimator/policy.

### 17) Unsupervised / semi-supervised anomaly detection for cyber-physical systems
- **Citation**: IEEE/Elsevier on self-supervised or semi-supervised anomaly detection.
- **Why**: Attack data scarcity; leverages nominal-only training.
- **Use**: thresholding based on validation nominal distribution; PR under scarcity.

### 18) Evaluation methodology for anomaly detection under imbalance & operating conditions
- **Citation**: ML reliability / anomaly detection evaluation methodology (applied to security).
- **Why**: Ensures detectors are evaluated correctly for grid context.
- **Use**: emphasize PR-AUC, cost-sensitive metrics, and operating-point curves.

---

## Notes for integration with the notebook framework
For each paper, the notebook should extract:
1) data source (SCADA/AMI/PMU/load)  
2) attack type (FDIA, DoS, replay, sensor spoofing, load tampering)  
3) model (AE/VAE, RNN, GNN, contrastive, Bayesian)  
4) anomaly score definition \(s(\cdot)\)  
5) thresholding method and evaluation metrics (ROC, PR-AUC, F1)  
6) robustness/resilience link (decision policy switch, uncertainty use)

---

## TODO (coordinating agent)
To finalize bibliographic accuracy, replace placeholder citations with exact authors/title/year/venue/DOI/arXiv links consistent with the project’s BibTeX style.