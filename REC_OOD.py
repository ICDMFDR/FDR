import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from sklearn.model_selection import KFold
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from tabpfn import TabPFNRegressor

# ---------------------------- Utility Functions ----------------------------


def define_models(seed: int = 42) -> dict:
    # Define and return a dictionary of models with a fixed random seed
    return {
        "FDR": TabPFNRegressor(random_state=seed),
        "CB": CatBoostRegressor(),
        "NN": MLPRegressor(random_state=seed, max_iter=1000),
        "RF": RandomForestRegressor(random_state=seed, n_estimators=100),
        "SVR": SVR(),
        "XGB": XGBRegressor(random_state=seed),
        "LR": LinearRegression(),
    }

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_dataset(path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    return pd.read_csv(path, encoding=encoding, engine='python')

def preprocess_data(dataset: pd.DataFrame, outcome_col: str, remove_cols: list) -> tuple:
    X = dataset.drop(columns=remove_cols + [outcome_col], axis=1)
    y = dataset[outcome_col].astype(float)
    valid_idx = y.notna()
    X, y = X.loc[valid_idx], y[valid_idx]
    X = X.dropna(axis=1, how='all')
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns), y

def recommend_treatment(model, X_test: pd.DataFrame, treatment_plans: list) -> pd.DataFrame:
    """
    Recommend the best treatment plan for each sample based on the lowest predicted outcome.
    """
    predicted_outcomes = pd.DataFrame(index=X_test.index)

    for tp in treatment_plans:
        X_counterfactual = X_test.copy()
        X_counterfactual[treatment_plans] = 0  # Set all TPs to 0
        X_counterfactual[tp] = 1  # Set the current TP to 1
        predicted_outcomes[tp] = model.predict(X_counterfactual)

    # Recommend the TP with the lowest predicted outcome
    predicted_outcomes['REC_TP'] = predicted_outcomes.idxmin(axis=1)
    return predicted_outcomes


def compare_recommendations(recommended_df: pd.DataFrame, original_df: pd.DataFrame, outcome_col: str, tp_cols: list):
    recommended_df['CURRENT_TP'] = original_df[tp_cols].idxmax(axis=1)
    recommended_df['FOLLOW_REC'] = recommended_df['REC_TP'] == recommended_df['CURRENT_TP']
    combined = pd.concat([original_df.reset_index(drop=True), recommended_df.reset_index(drop=True)], axis=1)
    return combined, None, None

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred).ravel()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, mae, r2, y_pred


def save_summary_stats(summary_list, method_name, final_df, output_path, outcome_col):
    recovery_stats = final_df.groupby("FOLLOW_REC")["resp.pCR"].agg(['count', 'sum']).reset_index()
    rcb_stats = final_df.groupby("FOLLOW_REC")[outcome_col].agg(['count', 'mean']).reset_index()
    for _, row in recovery_stats.iterrows():
        summary_list['recovery'].append({
            "Method": method_name,
            "FOLLOW_REC": int(row["FOLLOW_REC"]),
            "Count": int(row["count"]),
            "Recovery": int(row["sum"])
        })
    for _, row in rcb_stats.iterrows():
        summary_list['rcb'].append({
            "Method": method_name,
            "FOLLOW_REC": int(row["FOLLOW_REC"]),
            "Count": int(row["count"]),
            "Avg_RCB_Score": round(row["mean"], 4)
        })

def run_analysis_pipeline(
    data_name: str,
    treatment_plans: list,
    outcome_col: str,
    remove_cols: list,
    base_path: str = os.getcwd(),
    seed: int = 42
):
    input_path = os.path.join(base_path, 'input', data_name)
    output_path = os.path.join(base_path, 'output', data_name)
    ensure_dir(output_path)
    train_file = os.path.join(input_path, f"{data_name}_train.csv")
    test_file = os.path.join(input_path, f"{data_name}_test.csv")
    ctr_file = os.path.join(output_path, f"{data_name}_CTR.csv")


    X_train, y_train = preprocess_data(load_dataset(train_file), outcome_col, remove_cols)
    test_data = load_dataset(test_file)
    X_test, y_test = preprocess_data(test_data, outcome_col, remove_cols)
    models = define_models(seed)
    summary_stats = {'recovery': [], 'rcb': []}
    performance_metrics = {}


    for name, model in models.items():
        model.fit(X_train, y_train)
        mse, mae, r2, _ = evaluate_model(model, X_test, y_test)
        performance_metrics[name] = {"Model": name, "MSE": mse, "MAE": mae, "R2": r2}
        rec_df = recommend_treatment(model, X_test, treatment_plans)
        combined_df, _, _ = compare_recommendations(rec_df, test_data, outcome_col, treatment_plans)
        combined_df.to_csv(os.path.join(output_path, f"{data_name}_{name}_REC.csv"), index=False)
        save_summary_stats(summary_stats, name, combined_df, output_path, outcome_col)
    if os.path.exists(ctr_file):
        ctr_results = load_dataset(ctr_file)
        save_summary_stats(summary_stats, "CausalTree", ctr_results, output_path, outcome_col)
    pd.DataFrame.from_dict(performance_metrics, orient='index').to_csv(
        os.path.join(output_path, f"{data_name}_model_performance.csv")
    )


    recovery_df = pd.DataFrame(summary_stats['recovery']).pivot(
        index="Method", columns="FOLLOW_REC", values=["Count", "Recovery"]
    )
    recovery_df.columns = ['NotFollowing_Count', 'Following_Count', 'NotFollowing_Recovery', 'Following_Recovery']
    recovery_df = recovery_df.reset_index()
    recovery_df.to_csv(os.path.join(output_path, f"{data_name}_Recovery_Comparison.csv"), index=False)


    rcb_df = pd.DataFrame(summary_stats['rcb']).pivot(
        index="Method", columns="FOLLOW_REC", values=["Count", "Avg_RCB_Score"]
    )
    rcb_df.columns = ['NotFollowing_Count', 'Following_Count', 'NotFollowing_Avg_RCB', 'Following_Avg_RCB']
    rcb_df = rcb_df.reset_index()
    rcb_df.to_csv(os.path.join(output_path, f"{data_name}_RCB_Score_Comparison.csv"), index=False)
    return {
        "performance": performance_metrics,
        "output_dir": output_path,
        "recovery_summary": recovery_df,
        "rcb_summary": rcb_df
    }



def run_multiple_analyses():
    datasets = [
        {
            "data_name": "OOD_multi_3TS_Trans_ART",
            "treatment_plans": ["TP1", "TP2", "TP3"]
        },
        {
            "data_name": "OOD_multi_Trans_ART",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"]
        }
    ]

    outcome_col = "RCB.score"
    remove_cols = [
        "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "resp.pCR", "RCB.category",
        "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline", "Chemo.second.Taxane",
        "Chemo.second.Anthracycline", "Chemo.any.Anthracycline", "Chemo.any.antiHER2"
    ]

    for cfg in datasets:
        print(f"\nRunning analysis for dataset: {cfg['data_name']}")
        run_analysis_pipeline(
            data_name=cfg["data_name"],
            treatment_plans=cfg["treatment_plans"],
            outcome_col=outcome_col,
            remove_cols=remove_cols
        )

# Execute the batch analysis
run_multiple_analyses()