---
name: ml-research
description: Machine learning research and experimentation workflows. Use when working on ML projects, thesis work, model prototyping, dataset analysis, or benchmarking — especially in PyTorch, scikit-learn, or Jupyter environments.
---

# ML Research

Guidelines for machine learning experiments, from data exploration to model evaluation.

## Workflow

1. **Exploratory Data Analysis**
   - Use `jupyter` for EDA: plot distributions, check missing values, detect outliers.
   - Document findings in markdown cells or short notes.

2. **Literature Review**
   - Use `exa` to search for recent papers (2024+).
   - Preferred sources: arxiv.org, ieeexplore.ieee.org.
   - Summarize 3–5 key papers and their relevance to the project.

3. **Modeling**
   - Build multiple baselines before complex architectures.
   - Track metrics across experiments.
   - Document hyperparameters and results.

4. **Code Quality**
   - Use type hints for all public functions.
   - Keep notebooks under version control discipline (clean outputs before sharing).
   - Maintain `data/`, `models/`, `notebooks/`, `src/`, `tests/` structure.

## Metrics to Track

- Classification: Accuracy, Precision, Recall, F1
- Regression: MAE, RMSE, R²
- Efficiency: training time, inference time, model size
