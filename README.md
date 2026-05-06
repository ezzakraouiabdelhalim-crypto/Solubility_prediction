# 🧪 Aqueous Solubility Prediction

Predicting aqueous solubility (logS) from molecular structure using an ensemble of tuned tree-based models.

## Results

| Model | RMSE | R² |
|-------|------|-----|
| LightGBM | 1.256 | 0.684 |
| XGBoost | 1.259 | 0.682 |
| Random Forest | 1.289 | 0.667 |
| **Stacking (LGBM+XGB+RF)** | **1.234** | **0.694** |

## Features

- **Dataset:** ~9,000 molecules from the Curated Solubility Dataset
- **Molecular features:** 200+ RDKit descriptors + 2048-bit Morgan fingerprints
- **Data cleaning:** SMILES standardization, fragment removal, uncharging, atom filtering
- **Scaffold split** for realistic evaluation (no data leakage)
- **Hyperparameter tuning** with Optuna
- **Applicability domain** check using Tanimoto similarity
- **Confidence scoring** — composite system combining AD score, BertzCT complexity, and miscibility detection
- **SHAP analysis** for model interpretability
- **Streamlit app** for interactive predictions

## Project Structure

```
├── solubility_clean.ipynb    # Main notebook (full pipeline)
├── app.py                    # Streamlit web app
├── stack_model.pkl           # Trained stacking model
├── scaler.pkl                # Feature scaler
├── desc_columns.pkl          # Descriptor column names
├── train_fps.pkl             # Training set fingerprints (for AD)
└── README.md
```

## Quick Start

```bash
# Install dependencies
pip install pandas numpy scikit-learn xgboost lightgbm rdkit optuna shap streamlit joblib

# Run the Streamlit app
streamlit run app.py
```

## How It Works

1. **Input:** SMILES string (molecular structure)
2. **Feature extraction:** RDKit descriptors + Morgan fingerprints
3. **Prediction:** Stacking ensemble (LGBM + XGB + RF → Ridge meta-learner)
4. **Output:** Predicted logS + AD check + confidence score

## Confidence System

The model provides two layers of confidence:

- **Applicability Domain:** Tanimoto similarity to nearest training molecule
  - \> 0.4: In domain
  - \> 0.25: Low confidence
  - ≤ 0.25: Out of domain

- **Composite Confidence:** Combines AD score, molecular complexity (BertzCT), atom count, and miscibility risk (small polar molecules flagged as unreliable)

## Example Predictions

| Molecule | Predicted logS | Real logS | Error |
|----------|---------------|-----------|-------|
| Cholesterol | -7.01 | -7.0 | 0.01 |
| Diazepam | -3.86 | -3.84 | 0.02 |
| DDT | -7.13 | -7.3 | 0.17 |
| Ibuprofen | -3.58 | -3.7 | 0.12 |
| Hexane | -3.95 | -3.85 | 0.10 |

## Data
Download the dataset [here](https://raw.githubusercontent.com/PatWalters/solubility/master/solubility-data.csv) and place it in the project folder.

The model excels on drug-like molecules (avg error ~0.3 logS) and is less reliable on very simple/miscible compounds.

## Tools

Python, pandas, scikit-learn, XGBoost, LightGBM, RDKit, Optuna, SHAP, Streamlit
