# Clayey Soil ANN Prediction

This Streamlit application provides a research-oriented ANN workflow for clayey-soil engineering-property prediction.

## Main workflow

1. Upload a development Excel dataset.
2. Select **any column as the target**; the target is automatically excluded from the input list.
3. Select any remaining columns as input features.
4. Split the usable dataset into **80% development/training and 20% untouched holdout** using a fixed random state.
5. Within the 80% development set, evaluate multiple ANN architectures using **5-fold cross-validation**.
6. Select the best architecture primarily by the lowest mean CV RMSE, with R² and model simplicity used as tie-breakers.
7. Retrain the selected architecture on the complete 80% development set.
8. Evaluate the untouched 20% holdout.
9. Optionally evaluate a completely separate external unseen Excel dataset.
10. Inspect detailed ANN training diagnostics, residuals, architecture comparisons and overfitting indicators.
11. Generate SHAP global feature importance.
12. Use Gemini only for academic textual interpretation; ANN remains the numerical prediction model.

## Training diagnostics included

- Training loss vs iteration
- Internal validation score vs iteration
- Actual vs predicted for development data
- Residual vs predicted
- Residual distribution
- Automatic ANN architecture/network diagram
- Architecture comparison by CV RMSE and R²
- Fold-by-fold CV results
- Overfitting/underfitting diagnostic based on training vs CV R² gap
- 20% holdout actual-vs-predicted and residual plots
- External unseen-test actual-vs-predicted and residual plots
- CSV downloads for CV, predictions and SHAP results

## ANN configuration

- Imputation: median
- Scaling: StandardScaler
- Activation: ReLU
- Solver/optimizer: Adam
- Early stopping: enabled
- Internal validation fraction: 15% of each ANN fitting portion
- Candidate architectures: `(8,)`, `(16,)`, `(32,)`, `(8,4)`, `(16,8)`, `(32,16)`, `(64,32)`

## Important research interpretation

The 20% holdout is kept outside architecture selection. This provides a cleaner estimate of generalization than selecting the model using the same data later reported as an independent test.

The optional external unseen dataset is an additional evaluation and is not used during training or optimization.

## Gemini

Set `GEMINI_API_KEY` in Streamlit Secrets or as an environment variable. Do not place API keys directly in `app.py`.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
