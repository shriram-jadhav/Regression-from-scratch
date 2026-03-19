# Salary Prediction using Machine Learning

> A custom gradient-descent Linear Regression model built from scratch to predict salary from years of experience — with Z-score normalization, loss tracking, and visualization.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-scientific-blue?style=flat-square&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-visualization-orange?style=flat-square)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project predicts an employee's salary based on their years of experience using a **custom Linear Regression model** implemented from scratch with gradient descent — no Scikit-learn for modeling. The dataset contains 30 samples spanning 1.1 to 10.5 years of experience, with salaries ranging from $37,731 to $122,391.

---

## Dataset

| Property       | Value                                |
| -------------- | ------------------------------------ |
| File           | `Salary_Data.csv`                    |
| Samples        | 30                                   |
| Feature        | `YearsExperience` (1.1 – 10.5 years) |
| Target         | `Salary` ($37,731 – $122,391)        |
| Missing Values | None                                 |

---

## ML Pipeline

| Stage          | Description                                                            |
| -------------- | ---------------------------------------------------------------------- |
| Data Loading   | Load CSV with Pandas, extract `YearsExperience` and `Salary`           |
| Normalization  | Z-score normalization (custom `norm()` using mean & variance)          |
| Visualization  | Line plot to confirm linear relationship before modeling               |
| Model Training | Custom `SimpleLR` class using gradient descent (lr=0.1, max_iter=2000) |
| Loss Tracking  | MSE loss logged each iteration; training stops when loss change < 1e-6 |
| Evaluation     | R² Score and MSE computed on training data                             |

---

## Model: Custom Linear Regression (`SimpleLR`)

Built entirely with NumPy — no ML libraries for the core model.

```python
class SimpleLR:
    def __init__(self, lr=0.1, max_iter=2000, threshold=1e-6):
        ...
    def predict(self, X):
        return self.weight * X + self.bias
    def fit(self, X, Y):
        # Gradient descent with early stopping
        ...
    def plot(self, X, Y):
        # Plot actual vs predicted
        ...
```

---

## Results

| Metric               | Value                                      |
| -------------------- | ------------------------------------------ |
| **R² Score**         | **0.9570**                                 |
| **MSE**              | **31,270,951**                             |
| **RMSE**             | **~$5,592**                                |
| Training Convergence | ~170 iterations (early stopping triggered) |

The model explains **95.7%** of salary variance using only years of experience, converging smoothly with a steadily decreasing loss curve.

---

## Project Structure

```
.
├── regression.ipynb      # Main notebook (data loading → training → evaluation)
├── Salary_Data.csv       # Dataset (30 samples)
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## Tech Stack

- **Language**: Python 3.10
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib
- **Environment**: Conda, Jupyter Notebook
- **Modeling**: Custom implementation (no Scikit-learn for the model)

---

## Getting Started

### 1. Create and activate environment

```bash
conda create -n ml_env python=3.10
conda activate ml_env
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the notebook

```bash
jupyter notebook
```

Open `regression.ipynb` and run all cells in order.

---

## Roadmap

- [ ] Add train/test split for proper generalization evaluation
- [ ] Try advanced models (Random Forest, XGBoost)
- [ ] Hyperparameter tuning (learning rate, iterations)
- [ ] Deploy as a Streamlit web app for interactive predictions
- [ ] Incorporate larger, real-world salary datasets with more features

---

## Author

**Shriram Jadhav**
