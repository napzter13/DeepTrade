#!/usr/bin/env python3
"""
feature_selection.py

Performs feature selection analyses on the same data used by fitter.py / nn_model.py.
Methods used:
  1) Correlation-based selection
  2) PCA
  3) Recursive Feature Elimination (RFE)
  4) SHAP

Flattening time-series features into a single vector per sample, **after** using
the same input_preprocessing steps (scaling, handling NaNs/Infs, etc.) that fitter.py does.

We pick y = Y[:, 0] as our target for these analyses by default.

USAGE:

  python feature_selection.py \
    --csv training_data/training_data.csv \
    --out_dir feature_selection_results \
    --corr_threshold 0.95

Dependencies:
  - scikit-learn
  - numpy, pandas, matplotlib, seaborn
  - shap
  - botlib.input_preprocessing

Outputs:
  - correlation_heatmap.png
  - pca_explained_variance.png
  - rfe_ranking.txt
  - shap_summary_bar.png
  - shap_summary_dot.png
"""

import os
import csv
import json
import argparse
import logging
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from botlib.input_preprocessing import ModelScaler, prepare_for_model_inputs

def drop_highly_correlated_features(df, threshold=0.95):
    """
    Given a DataFrame 'df' of features (no target column),
    returns a list of columns to drop where correlation > threshold.

    We use a greedy approach: once a feature is identified
    to be correlated above threshold with any feature already
    in 'to_drop', we skip it.
    """
    corr_matrix = df.corr().abs()
    upper_triangle = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )
    to_drop = [
        column
        for column in upper_triangle.columns
        if any(upper_triangle[column] > threshold)
    ]
    return to_drop

def flatten_timeseries_features(
    X_5m, X_15m, X_1h,
    X_gt, X_sa,
    X_ta, X_ctx
):
    """
    Flatten each row's time-series data into a 1D vector,
    then concatenate all feature groups together.

    Output shape: (N, total_features)
    """
    N = X_5m.shape[0]
    # Flatten
    X_5m_flat  = X_5m.reshape(N, -1)   # e.g. (N, 241*9)
    X_15m_flat = X_15m.reshape(N, -1)
    X_1h_flat  = X_1h.reshape(N, -1)
    X_gt_flat  = X_gt.reshape(N, -1)   # e.g. (N, 24*1)
    # X_sa, X_ta, X_ctx are already 2D with shapes (N, 12), (N, 63), (N, 11)

    # Concatenate horizontally
    X_concat = np.concatenate(
        [X_5m_flat, X_15m_flat, X_1h_flat,
         X_gt_flat, X_sa, X_ta, X_ctx],
        axis=1
    )
    return X_concat

def run_correlation_analysis(df, out_path, threshold=0.95):
    """
    Correlation analysis: produce heatmap and drop correlated features.
    """
    corr = df.corr()
    plt.figure(figsize=(12,10))
    sns.heatmap(corr, cmap='coolwarm', vmax=1.0, vmin=-1.0)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    heatmap_file = os.path.join(out_path, "correlation_heatmap.png")
    plt.savefig(heatmap_file, dpi=150)
    plt.close()

    # Drop features with correlation above threshold
    to_drop = drop_highly_correlated_features(df, threshold=threshold)
    return to_drop

def run_pca(X, feature_names, out_path, n_components=20):
    """
    Fit PCA and visualize variance explained by the first n_components.
    """
    pca = PCA(n_components=n_components)
    pca.fit(X)
    explained_variance = pca.explained_variance_ratio_
    cum_explained = np.cumsum(explained_variance)

    plt.figure(figsize=(8,5))
    plt.bar(range(1, n_components+1), explained_variance, alpha=0.5,
            align='center', label='individual variance')
    plt.step(range(1, n_components+1), cum_explained, where='mid',
             label='cumulative variance')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('PCA Explained Variance')
    plt.legend(loc='best')
    plt.tight_layout()

    pca_plot_file = os.path.join(out_path, "pca_explained_variance.png")
    plt.savefig(pca_plot_file, dpi=150)
    plt.close()

    return pca

def run_rfe(X, y, feature_names, out_path, n_features_to_select=20):
    """
    Perform Recursive Feature Elimination with a RandomForestRegressor
    to pick the top n features.
    """
    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rfe = RFE(estimator=rf, n_features_to_select=n_features_to_select, step=1)
    rfe.fit(X, y)

    ranks = rfe.ranking_
    support = rfe.support_

    # Summarize in a text file
    selected_features = [
        (feature_names[i], ranks[i])
        for i in range(len(feature_names))
    ]
    # sort by rank
    selected_features.sort(key=lambda x: x[1])

    out_txt = os.path.join(out_path, "rfe_ranking.txt")
    with open(out_txt, "w") as f:
        f.write("Feature\tRank\tSelected\n")
        for i, (fname, rank) in enumerate(selected_features):
            sel_flag = "*" if rank == 1 else ""
            f.write(f"{fname}\t{rank}\t{sel_flag}\n")

    # The final chosen features
    chosen_feat_names = [
        feature_names[i]
        for i in range(len(feature_names))
        if support[i]
    ]
    logging.info(f"[RFE] Top {n_features_to_select} features: {chosen_feat_names}")

    return rfe

def run_shap(X, y, feature_names, out_path, sample_size=1000):
    """
    Compute SHAP values on a RandomForestRegressor and produce a summary plot.
    Because SHAP can be expensive, we sample the dataset if it's large.
    """
    if X.shape[0] > sample_size:
        idx = np.random.choice(X.shape[0], sample_size, replace=False)
        X_sample = X[idx]
        y_sample = y[idx]
    else:
        X_sample = X
        y_sample = y

    rf = RandomForestRegressor(n_estimators=50, random_state=42)
    rf.fit(X_sample, y_sample)

    explainer = shap.Explainer(rf, X_sample)
    shap_values = explainer(X_sample)

    # SHAP summary plot (bar)
    shap.summary_plot(
        shap_values,
        features=X_sample,
        feature_names=feature_names,
        show=False,
        plot_type='bar'
    )
    plt.title("SHAP Feature Importance (RandomForest) - Bar")
    shap_barplot_file = os.path.join(out_path, "shap_summary_bar.png")
    plt.savefig(shap_barplot_file, dpi=150, bbox_inches='tight')
    plt.close()

    # SHAP summary plot (dot)
    shap.summary_plot(
        shap_values,
        features=X_sample,
        feature_names=feature_names,
        show=False,
        plot_type='dot'
    )
    plt.title("SHAP Feature Importance (RandomForest) - Dot")
    shap_dot_file = os.path.join(out_path, "shap_summary_dot.png")
    plt.savefig(shap_dot_file, dpi=150, bbox_inches='tight')
    plt.close()

def load_and_preprocess_data(training_csv, num_future_steps=10):
    """
    Loads raw data from training_csv, then uses the same ModelScaler & prepare_for_model_inputs
    logic used in fitter.py to handle NaNs/Infs and scaling.
    Finally returns:
      X_concat, y, feature_names, timestamps
    where X_concat is a 2D array ready for correlation, PCA, RFE, etc.
    We pick y = Y[:,0] by default.
    """
    if not os.path.exists(training_csv):
        raise FileNotFoundError(f"CSV file not found: {training_csv}")

    all_5m   = []
    all_15m  = []
    all_1h   = []
    all_gt   = []
    all_sa   = []
    all_ta   = []
    all_ctx  = []
    all_Y    = []
    all_ts   = []

    with open(training_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Check columns
        missing_cols = []
        for i in range(1, num_future_steps+1):
            c = f"y_{i}"
            if c not in reader.fieldnames:
                missing_cols.append(c)
        if missing_cols:
            raise ValueError(f"CSV missing columns: {missing_cols}")

        for row in reader:
            try:
                timestamp_str = row["timestamp"]
                arr_5m_str  = row["arr_5m"]
                arr_15m_str = row["arr_15m"]
                arr_1h_str  = row["arr_1h"]
                arr_gt_str  = row["arr_google_trend"]
                arr_sa_str  = row["arr_santiment"]
                arr_ta_str  = row["arr_ta_63"]
                arr_ctx_str = row["arr_ctx_11"]

                arr_5m_list   = json.loads(arr_5m_str)[0]
                arr_15m_list  = json.loads(arr_15m_str)[0]
                arr_1h_list   = json.loads(arr_1h_str)[0]
                arr_gt_list   = json.loads(arr_gt_str)[0]
                arr_sa_list   = json.loads(arr_sa_str)[0]
                arr_ta_list   = json.loads(arr_ta_str)[0]
                arr_ctx_list  = json.loads(arr_ctx_str)[0]

                # dimension checks
                if (len(arr_5m_list)  !=241 or
                    len(arr_15m_list) !=241 or
                    len(arr_1h_list)  !=241 or
                    len(arr_gt_list)  !=24  or
                    len(arr_sa_list)  !=12  or
                    len(arr_ta_list)  !=63  or
                    len(arr_ctx_list) !=11):
                    continue

                y_vec = []
                for i in range(1, num_future_steps+1):
                    y_val = float(row[f"y_{i}"])
                    y_vec.append(y_val)

                all_5m.append(arr_5m_list)
                all_15m.append(arr_15m_list)
                all_1h.append(arr_1h_list)
                all_gt.append(arr_gt_list)
                all_sa.append(arr_sa_list)
                all_ta.append(arr_ta_list)
                all_ctx.append(arr_ctx_list)
                all_Y.append(y_vec)
                all_ts.append(timestamp_str)

            except Exception as e:
                # Skip malformed row
                continue

    if len(all_5m) == 0:
        raise ValueError("No valid rows loaded from CSV.")

    # Convert to NumPy
    X_5m  = np.array(all_5m, dtype=np.float32)   # (N, 241, 9)
    X_15m = np.array(all_15m, dtype=np.float32)  # (N, 241, 9)
    X_1h  = np.array(all_1h, dtype=np.float32)   # (N, 241, 9)
    X_gt  = np.array(all_gt, dtype=np.float32)   # (N, 24, 1)
    X_sa  = np.array(all_sa, dtype=np.float32)   # (N, 12)
    X_ta  = np.array(all_ta, dtype=np.float32)   # (N, 63)
    X_ctx = np.array(all_ctx, dtype=np.float32)  # (N, 11)
    Y     = np.array(all_Y,  dtype=np.float32)   # (N, num_future_steps)

    # ----------------------------------------------------------------------
    # 1) Scale + handle NaNs via the same logic as in fitter.py
    #    We create and fit a new ModelScaler. (We do not necessarily load from .pkl here.)
    # ----------------------------------------------------------------------
    model_scaler = ModelScaler()
    model_scaler.fit_all(
        X_5m, X_15m, X_1h,
        X_gt, X_sa,
        X_ta, X_ctx
    )
    X_5m, X_15m, X_1h, X_gt, X_sa, X_ta, X_ctx = prepare_for_model_inputs(
        X_5m, X_15m, X_1h,
        X_gt, X_sa,
        X_ta, X_ctx,
        model_scaler
    )

    # ----------------------------------------------------------------------
    # 2) Flatten and build final X_concat
    # ----------------------------------------------------------------------
    X_concat = flatten_timeseries_features(
        X_5m, X_15m, X_1h,
        X_gt, X_sa,
        X_ta, X_ctx
    )
    # We'll pick y = Y[:,0] for single-target feature selection
    y = Y[:, 0]

    # ----------------------------------------------------------------------
    # 3) Create some feature names for reference in correlation, RFE, etc.
    # ----------------------------------------------------------------------
    # Each block: 241 * 9 = 2169 for 5m/15m/1h, 24*1=24 for GT, 12 for SA, 63 for TA, 11 for ctx
    n_5m  = X_5m.shape[1] * X_5m.shape[2] if X_5m.ndim==3 else X_5m.shape[1]
    n_15m = X_15m.shape[1]*X_15m.shape[2] if X_15m.ndim==3 else X_15m.shape[1]
    n_1h  = X_1h.shape[1] * X_1h.shape[2] if X_1h.ndim==3 else X_1h.shape[1]
    n_gt  = X_gt.shape[1] * X_gt.shape[2] if X_gt.ndim==3 else X_gt.shape[1]
    n_sa  = X_sa.shape[1]
    n_ta  = X_ta.shape[1]
    n_ctx = X_ctx.shape[1]

    feature_names_5m  = [f"5m_f{i}"  for i in range(n_5m)]
    feature_names_15m = [f"15m_f{i}" for i in range(n_15m)]
    feature_names_1h  = [f"1h_f{i}"  for i in range(n_1h)]
    feature_names_gt  = [f"gt_f{i}"  for i in range(n_gt)]
    feature_names_sa  = [f"sa_f{i}"  for i in range(n_sa)]
    feature_names_ta  = [f"ta_f{i}"  for i in range(n_ta)]
    feature_names_ctx = [f"ctx_f{i}" for i in range(n_ctx)]

    feature_names = (
        feature_names_5m + feature_names_15m + feature_names_1h +
        feature_names_gt + feature_names_sa + feature_names_ta + feature_names_ctx
    )

    return X_concat, y, feature_names, np.array(all_ts)

def main():
    parser = argparse.ArgumentParser(description="Feature Selection Analysis")
    parser.add_argument("--csv", type=str, default="training_data/training_data.csv",
                        help="Path to training CSV (same as used by fitter).")
    parser.add_argument("--out_dir", type=str, default="feature_selection_results",
                        help="Output directory for plots/results.")
    parser.add_argument("--corr_threshold", type=float, default=0.95,
                        help="Correlation threshold for dropping features.")
    parser.add_argument("--test_size", type=float, default=0.2,
                        help="Test split for RFE/SHAP evaluations.")
    parser.add_argument("--num_future_steps", type=int, default=10,
                        help="Number of future steps (y_1..y_n).")
    parser.add_argument("--n_components_pca", type=int, default=20,
                        help="Number of PCA components to plot.")
    parser.add_argument("--n_features_rfe", type=int, default=20,
                        help="Number of features to select via RFE.")
    parser.add_argument("--shap_sample_size", type=int, default=1000,
                        help="Sample size for SHAP analysis (to reduce compute).")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("FeatureSelection")

    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)

    # 1) Load + preprocess
    X, y, feature_names, timestamps = load_and_preprocess_data(
        training_csv=args.csv,
        num_future_steps=args.num_future_steps
    )
    logger.info(f"Loaded dataset: X.shape={X.shape}, y.shape={y.shape}")

    # # 2) Correlation analysis
    # #    We'll create a DataFrame so we can do correlation among X + y
    # df_features = pd.DataFrame(X, columns=feature_names)
    # df_features["target"] = y

    # logger.info("Running correlation analysis & generating heatmap...")
    # to_drop = run_correlation_analysis(
    #     df_features.drop(columns=["target"]),
    #     out_path=args.out_dir,
    #     threshold=args.corr_threshold
    # )
    # logger.info(f"[Correlation] Number of features correlated above {args.corr_threshold}: {len(to_drop)}")

    # # 3) PCA
    # logger.info("Running PCA to analyze explained variance...")
    # pca = run_pca(
    #     X, feature_names,
    #     out_path=args.out_dir,
    #     n_components=args.n_components_pca
    # )

    # 4) RFE
    logger.info("Running RFE with RandomForest...")
    X_train, X_test, y_train, y_test = train_test_split(
        X[:2000], y[:2000], test_size=args.test_size, random_state=42
    )
    rfe = run_rfe(
        X_train, y_train, feature_names,
        out_path=args.out_dir,
        n_features_to_select=args.n_features_rfe
    )

    # # 5) SHAP
    # logger.info("Running SHAP analysis with RandomForest on entire dataset (or sample)...")
    # run_shap(
    #     X, y, feature_names,
    #     out_path=args.out_dir,
    #     sample_size=args.shap_sample_size
    # )

    logger.info("Feature selection analysis completed.")


if __name__ == "__main__":
    main()
