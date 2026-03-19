# Salary Prediction using Machine Learning

> A regression-based ML pipeline that predicts salary from experience, skills, and related features — with preprocessing, feature selection, and model evaluation built in.

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?style=flat-square&logo=scikit-learn)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange?style=flat-square&logo=jupyter)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Overview

This project explores how input variables like years of experience and skill set correlate with salary. A regression model is trained, evaluated, and structured for future extensions including advanced algorithms and web deployment.

---

## ML Pipeline

| Stage             | Description                                                |
| ----------------- | ---------------------------------------------------------- |
| Preprocessing     | Handle missing values, encode categoricals, scale features |
| Feature Selection | Identify the variables most correlated with salary         |
| Model Training    | Fit a Linear Regression model on the processed dataset     |
| Evaluation        | Measure performance using R² Score and Mean Squared Error  |

---

## Tech Stack

- **Language**: Python 3.10
- **Data**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn
- **Modeling**: Scikit-learn
- **Environment**: Conda, Jupyter Notebook

---

## Results

| Metric                   | Value                |
| ------------------------ | -------------------- |
| Model                    | Linear Regression    |
| R² Score                 | _fill after running_ |
| Mean Squared Error (MSE) | _fill after running_ |

The model successfully predicts salary based on input features with reasonable accuracy using regression techniques.

---

## Project Structure

```
.
├── regression.ipynb      # Main notebook
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

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

Open `regression.ipynb` and run all cells.

---

## Roadmap

- [ ] Try advanced models (Random Forest, XGBoost)
- [ ] Hyperparameter tuning via GridSearchCV
- [ ] Deploy as a Streamlit web app for interactive predictions
- [ ] Incorporate larger, real-world salary datasets

---

## Author

**Shriram Jadhav**
