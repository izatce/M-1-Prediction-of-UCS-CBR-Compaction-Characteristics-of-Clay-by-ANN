import io
import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import shap
from google import genai
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    mean_absolute_percentage_error
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Clayey Soil ANN Prediction", page_icon="🌍", layout="wide")

TARGET_DEFAULT = "UCS (kPa)"
CANDIDATE_ARCHITECTURES = [(8,), (16,), (32,), (8, 4), (16, 8), (32, 16), (64, 32)]


def load_excel(file):
    df = pd.read_excel(file)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def metric_dict(y, p):
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    return {
        "R²": r2_score(y, p),
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
        "MAPE (%)": mean_absolute_percentage_error(y, p) * 100,
    }


def build_ann(architecture, max_iter, seed):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ann", MLPRegressor(
            hidden_layer_sizes=architecture,
            activation="relu",
            solver="adam",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=int(max_iter),
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=50,
            random_state=int(seed),
        )),
    ])


def evaluate_architecture(X, y, architecture, max_iter, seed, return_folds=False):
    kf = KFold(n_splits=5, shuffle=True, random_state=int(seed))
    rows = []
    for fold, (tr, va) in enumerate(kf.split(X), 1):
        model = build_ann(architecture, max_iter, seed + fold)
        model.fit(X.iloc[tr], y.iloc[tr])
        p = model.predict(X.iloc[va])
        m = metric_dict(y.iloc[va], p)
        rows.append({
            "Architecture": str(architecture),
            "Fold": fold,
            **m,
            "Iterations": model.named_steps["ann"].n_iter_,
            "Final Loss": model.named_steps["ann"].loss_,
        })
    fold_df = pd.DataFrame(rows)
    summary = {
        "Architecture": str(architecture),
        "Hidden Layers": len(architecture),
        "Total Neurons": sum(architecture),
        "Mean R²": fold_df["R²"].mean(),
        "SD R²": fold_df["R²"].std(ddof=1),
        "Mean MAE": fold_df["MAE"].mean(),
        "SD MAE": fold_df["MAE"].std(ddof=1),
        "Mean RMSE": fold_df["RMSE"].mean(),
        "SD RMSE": fold_df["RMSE"].std(ddof=1),
        "Mean MAPE (%)": fold_df["MAPE (%)"].mean(),
        "SD MAPE (%)": fold_df["MAPE (%)"].std(ddof=1),
        "Mean Iterations": fold_df["Iterations"].mean(),
    }
    return (summary, fold_df) if return_folds else summary


def png_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def architecture_diagram(input_features, architecture, target):
    # Keep visual node count manageable while still showing the real architecture.
    max_nodes = 10
    layers = [min(len(input_features), max_nodes), *[min(n, max_nodes) for n in architecture], 1]
    labels = ["Inputs", *[f"Hidden {i}" for i in range(1, len(architecture) + 1)], "Output"]
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = np.linspace(0.08, 0.92, len(layers))
    coords = []
    for li, (x, n) in enumerate(zip(x_positions, layers)):
        ys = np.linspace(0.12, 0.88, n) if n > 1 else np.array([0.5])
        coords.append([(x, y) for y in ys])
        for y in ys:
            ax.scatter(x, y, s=280, facecolors="white", edgecolors="black", zorder=3)
        ax.text(x, 0.98, labels[li], ha="center", va="bottom", fontsize=11, fontweight="bold")
    # connections
    for a, b in zip(coords[:-1], coords[1:]):
        for x1, y1 in a:
            for x2, y2 in b:
                ax.plot([x1, x2], [y1, y2], linewidth=0.35, alpha=0.25, zorder=1)
    ax.text(x_positions[0], 0.03, f"{len(input_features)} input features", ha="center", fontsize=9)
    for i, n in enumerate(architecture, 1):
        ax.text(x_positions[i], 0.03, f"{n} neurons", ha="center", fontsize=9)
    ax.text(x_positions[-1], 0.03, target, ha="center", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    ax.axis("off")
    ax.set_title(f"Selected ANN Architecture: {len(input_features)} → {' → '.join(map(str, architecture))} → 1")
    return fig


def get_gemini_key():
    try:
        if "GEMINI_API_KEY" in st.secrets:
            return st.secrets["GEMINI_API_KEY"]
    except Exception:
        pass
    return os.getenv("GEMINI_API_KEY")


# Session state
for key, value in {
    "trained": None,
    "optimization": None,
    "fold_results": None,
    "holdout_result": None,
    "external_test_result": None,
    "shap_importance": None,
    "gemini_text": None,
    "file_signature": None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

st.title("🌍 Clayey Soil Engineering Prediction Using ANN")
st.caption("Automatic architecture optimization • 80/20 holdout • 5-fold CV • independent unseen testing • training diagnostics • SHAP")

st.sidebar.header("⚙️ ANN Configuration")
st.sidebar.success("🤖 Automatic Architecture Selection")
st.sidebar.write("Hidden-layer architecture is selected automatically from the candidate set below.")
st.sidebar.code("\n".join(map(str, CANDIDATE_ARCHITECTURES)))
max_iter = st.sidebar.number_input("Maximum Training Iterations", min_value=500, max_value=5000, value=2000, step=100)
seed = st.sidebar.number_input("Random State", min_value=0, max_value=9999, value=42, step=1)
st.sidebar.markdown("**Main validation design:** 80% development/training + 20% untouched holdout testing. 5-fold CV is performed only within the 80% development set for architecture selection.")

# Seven clear stages
TABS = st.tabs([
    "📘 1. Data & Selection",
    "🧠 2. Train & Optimize",
    "📈 3. Training Diagnostics",
    "📊 4. CV Results",
    "🧪 5. 20% Holdout Test",
    "🎯 6. Manual Prediction",
    "📕 7. External Unseen Test",
    "🔍 8. XAI & Gemini",
])
t1, t2, t3, t4, t5, t6, t7, t8 = TABS

with t1:
    st.header("Stage 1: Upload Data and Select Inputs / Target")
    train_file = st.file_uploader("📘 Upload Development Dataset", type=["xlsx", "xls"], key="train_file")
    if train_file:
        try:
            train_df = load_excel(train_file)
            signature = f"{train_file.name}_{train_file.size}"
            if st.session_state.file_signature != signature:
                st.session_state.file_signature = signature
                st.session_state.trained = None
                st.session_state.optimization = None
                st.session_state.fold_results = None
                st.session_state.holdout_result = None
                st.session_state.external_test_result = None
                st.session_state.shap_importance = None
                st.session_state.gemini_text = None
                st.session_state.pop("manual_prediction", None)
                st.session_state.pop("manual_inputs", None)
                cols = list(train_df.columns)
                default_target = TARGET_DEFAULT if TARGET_DEFAULT in cols else cols[-1]
                st.session_state.target_widget = default_target
                st.session_state.features_widget = [c for c in cols if c not in [default_target, "S.No."]]
            st.session_state.train_df = train_df

            st.success(f"Loaded {len(train_df)} rows and {len(train_df.columns)} columns.")
            st.dataframe(train_df.head(10), use_container_width=True)

            cols = list(train_df.columns)
            target = st.selectbox("🎯 Select Output / Target", cols, key="target_widget")
            available_features = [c for c in cols if c != target]
            # The target is selected first, so it is automatically removed from the input options.
            # The multiselect owns its own session-state key; do not assign to that key after widget creation.
            features = st.multiselect("🧩 Select Input Features", available_features, key="features_widget")

            if target in features:
                st.error("The selected target is excluded automatically from the input list. Please refresh the feature selection if necessary.")
            if len(features) < 2:
                st.warning("Select at least two input features.")
            else:
                st.info(f"**Target:** {target}  |  **Inputs:** {len(features)} features  |  **Samples:** {len(train_df)}")
        except Exception as e:
            st.error(f"Could not read training file: {e}")

ready = (
    "train_df" in st.session_state
    and "target_widget" in st.session_state
    and "features_widget" in st.session_state
    and len(st.session_state.features_widget) >= 2
    and st.session_state.target_widget not in st.session_state.features_widget
)

if ready:
    selected = st.session_state.features_widget + [st.session_state.target_widget]
    raw = st.session_state.train_df[selected].copy()
    for c in raw.columns:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    raw = raw.dropna(subset=[st.session_state.target_widget]).reset_index(drop=True)
    X_all = raw[st.session_state.features_widget]
    y_all = raw[st.session_state.target_widget]

with t2:
    st.header("Stage 2: Automatic ANN Architecture Optimization")
    if not ready:
        st.warning("Upload the development dataset and select at least two input features.")
    else:
        st.write(f"Usable samples after removing missing target values: **{len(X_all)}**")
        st.markdown("**Scientific workflow:** 80% development data → 5-fold CV architecture selection → final model trained on the 80% development set → 20% untouched holdout evaluation.")
        if st.button("🚀 Automatically Optimize and Train ANN", type="primary"):
            try:
                Xdev, Xhold, ydev, yhold = train_test_split(
                    X_all, y_all, test_size=0.20, random_state=int(seed)
                )
                rows, all_folds = [], []
                bar = st.progress(0)
                status = st.empty()
                for i, arch in enumerate(CANDIDATE_ARCHITECTURES):
                    status.write(f"5-fold CV: evaluating architecture {arch} ...")
                    summary, folds = evaluate_architecture(
                        Xdev.reset_index(drop=True), ydev.reset_index(drop=True),
                        arch, int(max_iter), int(seed), return_folds=True
                    )
                    rows.append(summary)
                    all_folds.append(folds)
                    bar.progress(int((i + 1) / len(CANDIDATE_ARCHITECTURES) * 100))

                opt = pd.DataFrame(rows).sort_values(
                    ["Mean RMSE", "Mean R²", "Total Neurons"],
                    ascending=[True, False, True]
                ).reset_index(drop=True)
                opt.insert(0, "Rank", range(1, len(opt) + 1))
                fold_df = pd.concat(all_folds, ignore_index=True)
                lookup = {str(a): a for a in CANDIDATE_ARCHITECTURES}
                best_arch = lookup[opt.loc[0, "Architecture"]]

                status.write(f"Final training of selected architecture {best_arch} on the complete 80% development set ...")
                final_model = build_ann(best_arch, int(max_iter), int(seed))
                final_model.fit(Xdev, ydev)

                # Holdout is untouched until now.
                hold_pred = final_model.predict(Xhold)
                hold_metrics = metric_dict(yhold, hold_pred)
                hold_table = Xhold.copy()
                hold_table["Actual"] = yhold.values
                hold_table["Predicted"] = hold_pred
                hold_table["Residual (Actual - Predicted)"] = yhold.values - hold_pred

                st.session_state.optimization = opt
                st.session_state.fold_results = fold_df
                st.session_state.holdout_result = {
                    "metrics": hold_metrics,
                    "actual": yhold.to_numpy(),
                    "predicted": hold_pred,
                    "table": hold_table,
                }
                st.session_state.trained = {
                    "model": final_model,
                    "architecture": best_arch,
                    "features": list(st.session_state.features_widget),
                    "target": st.session_state.target_widget,
                    "Xdev": Xdev.copy(),
                    "ydev": ydev.copy(),
                    "Xall": X_all.copy(),
                    "yall": y_all.copy(),
                    "Xhold": Xhold.copy(),
                    "yhold": yhold.copy(),
                    "training_samples": len(Xdev),
                    "holdout_samples": len(Xhold),
                }
                st.session_state.external_test_result = None
                st.session_state.shap_importance = None
                st.session_state.gemini_text = None
                bar.empty(); status.empty()
                st.success(f"Training completed. Best architecture: {best_arch}. The 20% holdout was evaluated without being used for model selection.")
            except Exception as e:
                st.error(f"Training error: {e}")

        if st.session_state.trained:
            r = st.session_state.trained
            st.metric("Automatically Selected Architecture", str(r["architecture"]))
            st.write(f"Development training samples: **{r['training_samples']}** | Holdout samples: **{r['holdout_samples']}**")

with t3:
    st.header("Stage 3: Detailed ANN Training Diagnostics")
    if st.session_state.trained is None:
        st.info("Train the ANN model first.")
    else:
        trained = st.session_state.trained
        ann = trained["model"].named_steps["ann"]
        Xdev, ydev = trained["Xdev"], trained["ydev"]
        train_pred = trained["model"].predict(Xdev)
        train_m = metric_dict(ydev, train_pred)

        st.subheader("🧠 Selected Network Summary")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Architecture", str(trained["architecture"]))
        c2.metric("Hidden Layers", len(trained["architecture"]))
        c3.metric("Total Hidden Neurons", sum(trained["architecture"]))
        c4.metric("Iterations", ann.n_iter_)
        c5.metric("Final Loss", f"{ann.loss_:.6f}")
        st.write(f"**Activation:** {ann.activation} | **Optimizer:** {ann.solver} | **Learning rate:** {ann.learning_rate_init} | **Early stopping:** {ann.early_stopping}")

        st.subheader("🏗️ Automatic ANN Architecture Diagram")
        fig = architecture_diagram(trained["features"], trained["architecture"], trained["target"])
        st.pyplot(fig, use_container_width=True)
        st.download_button("⬇️ Download Architecture Diagram", png_bytes(fig), "ann_architecture.png", "image/png")

        st.subheader("📉 Learning Curve: Training Loss")
        if ann.loss_curve_:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(np.arange(1, len(ann.loss_curve_) + 1), ann.loss_curve_)
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Loss")
            ax.set_title("ANN Training Loss vs Iteration")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
            st.download_button("⬇️ Download Loss Curve", png_bytes(fig), "ann_loss_curve.png", "image/png")

        st.subheader("📈 Internal Validation Score")
        if getattr(ann, "validation_scores_", None) is not None:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(np.arange(1, len(ann.validation_scores_) + 1), ann.validation_scores_)
            ax.set_xlabel("Training Iteration")
            ax.set_ylabel("Internal Validation R²")
            ax.set_title("ANN Internal Validation Score vs Iteration")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        else:
            st.info("Internal validation scores are not available for this fitted model.")

        st.subheader("🎯 Development Training: Actual vs Predicted")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Training R²", f"{train_m['R²']:.4f}")
        c2.metric("Training RMSE", f"{train_m['RMSE']:.4f}")
        c3.metric("Training MAE", f"{train_m['MAE']:.4f}")
        c4.metric("Training MAPE", f"{train_m['MAPE (%)']:.2f}%")
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(ydev, train_pred, alpha=0.75)
        lo, hi = min(ydev.min(), train_pred.min()), max(ydev.max(), train_pred.max())
        ax.plot([lo, hi], [lo, hi], linestyle="--")
        ax.set_xlabel(f"Actual {trained['target']}")
        ax.set_ylabel(f"Predicted {trained['target']}")
        ax.set_title("Development Training: Actual vs Predicted")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        st.subheader("📊 Residual Diagnostics")
        residual = ydev.to_numpy() - train_pred
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(train_pred, residual, alpha=0.75)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual (Actual - Predicted)")
            ax.set_title("Residual vs Predicted")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(residual, bins=20, alpha=0.75)
            ax.axvline(0, linestyle="--")
            ax.set_xlabel("Residual")
            ax.set_ylabel("Frequency")
            ax.set_title("Residual Distribution")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.subheader("🏆 Architecture Comparison")
        opt = st.session_state.optimization
        fig, ax = plt.subplots(figsize=(9, 5))
        plot = opt.sort_values("Mean RMSE", ascending=True)
        ax.barh(plot["Architecture"], plot["Mean RMSE"])
        ax.set_xlabel("Mean 5-Fold CV RMSE (lower is better)")
        ax.set_ylabel("ANN Architecture")
        ax.set_title("Automatic Architecture Selection")
        ax.grid(axis="x", alpha=0.3)
        st.pyplot(fig)

        st.subheader("⚠️ Overfitting / Underfitting Diagnostic")
        best_cv_r2 = float(opt.iloc[0]["Mean R²"])
        gap = train_m["R²"] - best_cv_r2
        if gap > 0.15:
            st.warning(f"Training R² exceeds mean CV R² by {gap:.3f}. This can indicate overfitting; confirm using the untouched holdout and external unseen test.")
        elif gap < -0.10:
            st.info(f"Training R² is {abs(gap):.3f} below mean CV R². This is unusual and should be investigated with the holdout results and data distribution.")
        else:
            st.success(f"Training/CV R² gap = {gap:.3f}. No large training-to-CV gap is evident from this diagnostic alone.")

        st.subheader("🔎 Step-by-Step ANN Learning Process")
        st.markdown(f"""
1. **Data preparation:** {len(Xdev)} development samples and {len(trained['features'])} selected input features are used.
2. **20% holdout separation:** the holdout set is kept untouched during architecture selection.
3. **Missing-value treatment:** median imputation is fitted inside the pipeline using only each training fold.
4. **Standardization:** input features are standardized inside the pipeline, preventing scale differences from dominating learning.
5. **Forward propagation:** standardized inputs pass through the automatically selected hidden layers using **ReLU** activation.
6. **Prediction:** the output neuron produces the predicted target value.
7. **Loss calculation:** the ANN compares predicted and measured target values.
8. **Backpropagation + Adam:** weights are updated iteratively to minimize the loss.
9. **Internal early stopping:** 15% of each fold's training portion is used internally to monitor validation performance.
10. **5-fold CV:** candidate architectures are compared using RMSE, MAE, R² and MAPE.
11. **Architecture selection:** the lowest mean CV RMSE is the primary criterion; R² and model simplicity break ties.
12. **Final development training:** the selected architecture is retrained on all 80% development data.
13. **Final evaluation:** the untouched 20% holdout and, optionally, a separate external unseen dataset are evaluated only after model selection.
""")

with t4:
    st.header("Stage 4: Detailed 5-Fold Cross-Validation Results")
    if st.session_state.optimization is None:
        st.info("Train the model first.")
    else:
        opt = st.session_state.optimization.copy()
        fold_df = st.session_state.fold_results.copy()
        st.subheader("Architecture-Level Performance")
        st.dataframe(opt.style.format({
            "Mean R²": "{:.4f}", "SD R²": "{:.4f}",
            "Mean MAE": "{:.4f}", "SD MAE": "{:.4f}",
            "Mean RMSE": "{:.4f}", "SD RMSE": "{:.4f}",
            "Mean MAPE (%)": "{:.2f}", "SD MAPE (%)": "{:.2f}",
            "Mean Iterations": "{:.1f}",
        }), use_container_width=True)

        best = opt.iloc[0]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best Mean R²", f"{best['Mean R²']:.4f}")
        c2.metric("Best Mean RMSE", f"{best['Mean RMSE']:.4f}")
        c3.metric("Best Mean MAE", f"{best['Mean MAE']:.4f}")
        c4.metric("Best Mean MAPE", f"{best['Mean MAPE (%)']:.2f}%")

        st.subheader("📌 Fold-by-Fold Results for Selected Architecture")
        best_arch = best["Architecture"]
        selected_folds = fold_df[fold_df["Architecture"] == best_arch].copy()
        st.dataframe(selected_folds, use_container_width=True)

        st.subheader("📊 CV Performance by Fold")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(selected_folds["Fold"], selected_folds["RMSE"], marker="o", label="RMSE")
        ax.set_xlabel("Fold")
        ax.set_ylabel("RMSE")
        ax.set_title(f"Selected Architecture {best_arch}: Fold-by-Fold RMSE")
        ax.grid(alpha=0.3)
        st.pyplot(fig)

        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(opt["Architecture"], opt["Mean R²"])
            ax.set_ylabel("Mean CV R²")
            ax.set_title("Architecture Comparison: Mean R²")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.bar(opt["Architecture"], opt["Mean RMSE"])
            ax.set_ylabel("Mean CV RMSE")
            ax.set_title("Architecture Comparison: Mean RMSE")
            ax.tick_params(axis="x", rotation=45)
            ax.grid(axis="y", alpha=0.3)
            st.pyplot(fig)

        st.download_button("⬇️ Download All CV Results (CSV)", fold_df.to_csv(index=False).encode("utf-8"), "ann_all_cv_fold_results.csv", "text/csv")
        st.download_button("⬇️ Download Architecture Summary (CSV)", opt.to_csv(index=False).encode("utf-8"), "ann_architecture_summary.csv", "text/csv")

with t5:
    st.header("Stage 5: 20% Untouched Holdout Testing")
    if st.session_state.holdout_result is None:
        st.info("Train the ANN first. The 20% holdout is generated automatically and is not used during architecture selection.")
    else:
        result = st.session_state.holdout_result
        m = result["metrics"]
        st.success("The holdout set was not used for architecture selection or CV optimization.")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Holdout R²", f"{m['R²']:.4f}")
        c2.metric("Holdout RMSE", f"{m['RMSE']:.4f}")
        c3.metric("Holdout MAE", f"{m['MAE']:.4f}")
        c4.metric("Holdout MAPE", f"{m['MAPE (%)']:.2f}%")

        actual, pred = result["actual"], result["predicted"]
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(actual, pred, alpha=0.75)
            lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
            ax.plot([lo, hi], [lo, hi], linestyle="--")
            ax.set_xlabel(f"Actual {st.session_state.trained['target']}")
            ax.set_ylabel(f"Predicted {st.session_state.trained['target']}")
            ax.set_title("20% Holdout: Actual vs Predicted")
            ax.grid(alpha=0.3)
            st.pyplot(fig)
        with c2:
            residual = actual - pred
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.scatter(pred, residual, alpha=0.75)
            ax.axhline(0, linestyle="--")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Residual")
            ax.set_title("20% Holdout: Residual vs Predicted")
            ax.grid(alpha=0.3)
            st.pyplot(fig)

        st.dataframe(result["table"], use_container_width=True)
        st.download_button("⬇️ Download Holdout Predictions", result["table"].to_csv(index=False).encode("utf-8"), "ann_holdout_predictions.csv", "text/csv")

with t6:
    st.header("Stage 6: Manual Sample Prediction")
    if st.session_state.trained is None:
        st.info("Train the ANN model first. Manual prediction becomes available after the final ANN has been trained.")
    else:
        trained = st.session_state.trained
        st.write(
            "Enter one new soil sample below. **Every input is restricted to the minimum–maximum range observed in the training (80% development) data.** "
            "The trained ANN generates the numerical prediction; this step does not retrain or modify the model."
        )

        # Training-range table for transparency and research reporting.
        ranges = pd.DataFrame({
            "Input Feature": trained["features"],
            "Training Minimum": [trained["Xdev"][c].min() for c in trained["features"]],
            "Training Maximum": [trained["Xdev"][c].max() for c in trained["features"]],
            "Training Mean": [trained["Xdev"][c].mean() for c in trained["features"]],
        })
        st.subheader("📏 Allowed Training Data Range")
        st.dataframe(
            ranges.style.format({
                "Training Minimum": "{:.6g}",
                "Training Maximum": "{:.6g}",
                "Training Mean": "{:.6g}",
            }),
            use_container_width=True,
        )

        st.subheader("🧪 Enter Manual Sample")
        manual_values = {}
        input_cols = st.columns(2)
        for i, feature in enumerate(trained["features"]):
            series = pd.to_numeric(trained["Xdev"][feature], errors="coerce").dropna()
            if series.empty:
                st.error(f"No valid training values are available for input feature: {feature}")
                continue

            min_val = float(series.min())
            max_val = float(series.max())
            default_val = float(series.median())
            if min_val == max_val:
                step = 0.01
            else:
                step = max((max_val - min_val) / 1000.0, 1e-6)

            with input_cols[i % 2]:
                manual_values[feature] = st.number_input(
                    f"{feature}",
                    min_value=min_val,
                    max_value=max_val,
                    value=min(max(default_val, min_val), max_val),
                    step=step,
                    format="%.6f",
                    key=f"manual_input_{i}_{feature}",
                    help=f"Allowed range from training data: {min_val:.6g} to {max_val:.6g}",
                )

        st.caption("⚠️ Values outside the training-data range cannot be entered. This reduces extrapolation risk.")

        if st.button("🔮 Predict Manual Sample", type="primary", key="manual_predict_button"):
            try:
                manual_df = pd.DataFrame([manual_values], columns=trained["features"])
                prediction = float(trained["model"].predict(manual_df)[0])
                st.session_state.manual_prediction = prediction
                st.session_state.manual_inputs = manual_values.copy()
            except Exception as e:
                st.error(f"Manual prediction error: {e}")

        if "manual_prediction" in st.session_state and st.session_state.trained is not None:
            st.success("Manual sample prediction completed using the trained ANN.")
            result_col, info_col = st.columns([1, 2])
            with result_col:
                st.metric(
                    label=f"Predicted {trained['target']}",
                    value=f"{st.session_state.manual_prediction:.6g}",
                )
            with info_col:
                st.write("**Prediction method:** Final automatically selected ANN")
                st.write(f"**Architecture:** `{trained['architecture']}`")
                st.write("**Input constraint:** Within 80% development/training-data range")

            manual_table = pd.DataFrame([st.session_state.manual_inputs])
            manual_table[f"Predicted {trained['target']}"] = st.session_state.manual_prediction
            st.dataframe(manual_table, use_container_width=True)
            st.download_button(
                "⬇️ Download Manual Prediction (CSV)",
                manual_table.to_csv(index=False).encode("utf-8"),
                "ann_manual_prediction.csv",
                "text/csv",
                key="download_manual_prediction",
            )

with t7:
    st.header("Stage 7: Independent External Unseen Dataset")
    if st.session_state.trained is None:
        st.warning("Train the ANN model first.")
    else:
        st.write("This is an optional second, completely external test. It must contain the selected input columns and the selected target column.")
        test_file = st.file_uploader("📕 Upload External Unseen Dataset", type=["xlsx", "xls"], key="test_file")
        if test_file:
            try:
                test_df = load_excel(test_file)
                trained = st.session_state.trained
                required = trained["features"] + [trained["target"]]
                missing = [c for c in required if c not in test_df.columns]
                if missing:
                    st.error("Missing required columns: " + ", ".join(missing))
                else:
                    st.success(f"External unseen dataset contains {len(test_df)} rows.")
                    st.dataframe(test_df.head(10), use_container_width=True)
                    if st.button("🔬 Test ANN on External Unseen Data", type="primary"):
                        work = test_df[required].copy()
                        for c in required:
                            work[c] = pd.to_numeric(work[c], errors="coerce")
                        valid = work[trained["target"]].notna()
                        work = work.loc[valid].copy()
                        if len(work) == 0:
                            st.error("No rows contain a valid target value.")
                        else:
                            pred = trained["model"].predict(work[trained["features"]])
                            ytest = work[trained["target"]]
                            m = metric_dict(ytest, pred)
                            table = test_df.loc[work.index].copy()
                            table[f"Actual {trained['target']}"] = ytest.values
                            table[f"Predicted {trained['target']}"] = pred
                            table["Residual (Actual - Predicted)"] = ytest.values - pred
                            st.session_state.external_test_result = {
                                "metrics": m,
                                "actual": ytest.to_numpy(),
                                "predicted": pred,
                                "table": table,
                            }
                            st.success("External unseen testing completed.")
            except Exception as e:
                st.error(f"Testing error: {e}")

        if st.session_state.external_test_result:
            result = st.session_state.external_test_result
            m = result["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("External R²", f"{m['R²']:.4f}")
            c2.metric("External RMSE", f"{m['RMSE']:.4f}")
            c3.metric("External MAE", f"{m['MAE']:.4f}")
            c4.metric("External MAPE", f"{m['MAPE (%)']:.2f}%")
            actual, pred = result["actual"], result["predicted"]
            c1, c2 = st.columns(2)
            with c1:
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(actual, pred, alpha=0.75)
                lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())
                ax.plot([lo, hi], [lo, hi], linestyle="--")
                ax.set_xlabel(f"Actual {st.session_state.trained['target']}")
                ax.set_ylabel(f"Predicted {st.session_state.trained['target']}")
                ax.set_title("External Unseen Test: Actual vs Predicted")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            with c2:
                residual = actual - pred
                fig, ax = plt.subplots(figsize=(7, 5))
                ax.scatter(pred, residual, alpha=0.75)
                ax.axhline(0, linestyle="--")
                ax.set_xlabel("Predicted")
                ax.set_ylabel("Residual")
                ax.set_title("External Unseen Test: Residual vs Predicted")
                ax.grid(alpha=0.3)
                st.pyplot(fig)
            st.dataframe(result["table"], use_container_width=True)
            st.download_button("⬇️ Download External Test Predictions", result["table"].to_csv(index=False).encode("utf-8"), "ann_external_test_predictions.csv", "text/csv")

with t8:
    st.header("Stage 8: Explainable AI (SHAP) + Gemini Interpretation")
    if st.session_state.trained is None:
        st.info("Train the ANN model first.")
    else:
        trained = st.session_state.trained
        st.subheader("🔍 SHAP Global Feature Importance")
        if st.button("Generate SHAP Feature Importance"):
            try:
                with st.spinner("Generating SHAP analysis..."):
                    Xexp = trained["Xdev"]
                    background = shap.sample(Xexp, min(50, len(Xexp)), random_state=int(seed))
                    explain_data = Xexp.sample(min(100, len(Xexp)), random_state=int(seed))
                    explainer = shap.Explainer(trained["model"].predict, background)
                    values = explainer(explain_data)
                    imp = pd.DataFrame({
                        "Feature": trained["features"],
                        "Mean |SHAP Value|": np.abs(values.values).mean(axis=0),
                    }).sort_values("Mean |SHAP Value|", ascending=False)
                    st.session_state.shap_importance = imp
            except Exception as e:
                st.error(f"SHAP error: {e}")
        if st.session_state.shap_importance is not None:
            imp = st.session_state.shap_importance
            st.dataframe(imp, use_container_width=True)
            fig, ax = plt.subplots(figsize=(8, 5))
            plot = imp.sort_values("Mean |SHAP Value|")
            ax.barh(plot["Feature"], plot["Mean |SHAP Value|"])
            ax.set_xlabel("Mean Absolute SHAP Value")
            ax.set_title(f"Global Feature Importance for {trained['target']}")
            st.pyplot(fig)
            st.download_button("⬇️ Download SHAP Importance", imp.to_csv(index=False).encode("utf-8"), "ann_shap_importance.csv", "text/csv")

        st.subheader("🤖 Gemini Academic Interpretation")
        test_source = st.session_state.external_test_result or st.session_state.holdout_result
        if test_source is None:
            st.info("Complete the 20% holdout or external unseen test first.")
        else:
            m = test_source["metrics"]
            source_name = "external unseen testing" if st.session_state.external_test_result else "20% holdout testing"
            st.write(f"Interpretation source: **{source_name}** | R² = **{m['R²']:.4f}** | RMSE = **{m['RMSE']:.4f}** | MAE = **{m['MAE']:.4f}**")
            if st.button("🤖 Generate Gemini Interpretation"):
                key = get_gemini_key()
                if not key:
                    st.error("Gemini API key not found. Add GEMINI_API_KEY to Streamlit Secrets or environment variables.")
                else:
                    try:
                        features = ", ".join(trained["features"])
                        prompt = f"""You are assisting with academic geotechnical engineering research.\nTarget property: {trained['target']}.\nInput features: {features}.\nSelected ANN architecture: {trained['architecture']}.\nArchitecture selection used 5-fold cross-validation within the 80% development set. A separate 20% holdout was not used during model selection.\nTesting metrics from {source_name}: R2={m['R²']:.4f}, RMSE={m['RMSE']:.4f}, MAE={m['MAE']:.4f}, MAPE={m['MAPE (%)']:.2f}%.\nProvide a concise academic interpretation of model performance, limitations, and engineering meaning. Do not generate or alter numerical predictions. State clearly that the ANN generated the numerical predictions and Gemini only provides textual interpretation."""
                        client = genai.Client(api_key=key)
                        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
                        st.session_state.gemini_text = response.text
                    except Exception as e:
                        st.error(f"Gemini error: {e}")
            if st.session_state.gemini_text:
                st.markdown(st.session_state.gemini_text)

st.markdown("---")
st.caption("ANN research workflow: data selection → 80/20 split → 5-fold architecture optimization → final development training → holdout testing → manual prediction → external testing → diagnostics → SHAP → interpretation")
