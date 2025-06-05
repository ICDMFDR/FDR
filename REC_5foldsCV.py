import os
import numpy as np
import pandas as pd

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

def ensure_dir(path: str):
    # Ensure that the given directory path exists, creating it if it doesn't
    os.makedirs(path, exist_ok=True)

def load_dataset(path: str, encoding: str = "ISO-8859-1") -> pd.DataFrame:
    # Load a dataset from the specified file path using the given encoding
    return pd.read_csv(path, encoding=encoding, engine='python')

def preprocess_data(dataset: pd.DataFrame, outcome_name: str, remove_cols: list) -> tuple:
    # This function processes the dataset by removing specified columns, imputing missing values, and converting categorical variables.
    X = dataset.drop(columns=remove_cols + [outcome_name], axis=1)
    y = dataset[outcome_name].astype(float)
    valid_idx = y.notna()
    X, y = X.loc[valid_idx], y[valid_idx]
    X = X.dropna(axis=1, how='all')
    for col in X.select_dtypes(include=['object', 'category']).columns:
        X[col] = pd.factorize(X[col])[0]
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)
    return pd.DataFrame(X_imputed, columns=X.columns), y


# ---------------------------- Recommendation Functions ----------------------------

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

def make_recommendations_cv(
    models,
    data_name: str,
    treatment_plans: list,
    outcome_col: str,
    remove_cols: list,
    input_path,
    output_path,
    seed: int = 42,
    k: int = 5
):
    input_file = os.path.join(input_path, f"{data_name}.csv")
    df = load_dataset(input_file)
    X_full, y_full = preprocess_data(df, outcome_col, remove_cols)

    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    # Dictionary to store recommendations per model
    all_rec_dfs = {name: [] for name in models}

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_full)):
        print(f"Processing fold {fold + 1}/{k}")
        X_train, X_test = X_full.iloc[train_idx], X_full.iloc[test_idx]
        y_train, y_test = y_full.iloc[train_idx], y_full.iloc[test_idx]

        for name, model in models.items():
            model.fit(X_train, y_train)
            rec_df = recommend_treatment(model, X_test, treatment_plans)
            combined_df, _, _ = compare_recommendations(rec_df, df.iloc[test_idx], outcome_col, treatment_plans)

            all_rec_dfs[name].append(combined_df)

    # Save full rec_df for each model
    for name, df_list in all_rec_dfs.items():
        final_df = pd.concat(df_list, ignore_index=True)
        final_df.to_csv(os.path.join(output_path, f"{data_name}_{name}_REC.csv"), index=False)

    return final_df

def define_models(seed: int = 42) -> dict:
    # Define and return a dictionary of models with a fixed random seed
    return {
        "FDR": TabPFNRegressor(random_state=seed),
        "CB": CatBoostRegressor(),
        "NN": MLPRegressor(random_state=seed, max_iter=1000),
        "RF": RandomForestRegressor(random_state=seed, n_estimators=100),
        "SVR": SVR(),
        "XGB": XGBRegressor(random_state=seed),
        "LR": LinearRegression()
    }


# ---------------------------- Main Functions ----------------------------
def run_crossval_pipeline_entry(data_name, treatment_plans, outcome_col, remove_cols, base_path: str = os.getcwd(), seed=42):

    input_path = os.path.join(base_path, 'input', data_name)
    output_path = os.path.join(base_path, 'output', data_name)
    ensure_dir(output_path)

    models = define_models(seed)

    results = make_recommendations_cv(models,
                            data_name,
                            treatment_plans,
                            outcome_col,
                            remove_cols,
                            input_path,
                            output_path,
                            )

    return results


def run_all_datasets():
    datasets = [
        {
            "data_name": "clin_GSE25066",
            "treatment_plans": ["TP1", "TP2"],
            "outcome_col": "RCB.category",
            "remove_cols": ['Pathologic_response_pcr_rd', 'Pathologic_response_rcb_class',
                            'Chemosensitivity_prediction', 'Type_taxane', 'resp.pCR'
            ]
        },
        {
            "data_name": "clin_GSE41998",
            "treatment_plans": ["TP1", "TP2"],
            "outcome_col": "RCB.category",
            "remove_cols": [ 'Specimen_name', 'Treatment_arm', 'Ac_response', 'Pcr', 'Pcrrcb1'
            ]
        },
        {
            "data_name": "clin_TransNEO",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.category",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "resp.pCR", "RCB.score",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2"
            ]
        },
        {
            "data_name": "clin_ARTemis",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.category",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "resp.pCR", "RCB.score",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2"
            ]
        },
        {
            "data_name": "multi_Trans_ART",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.score",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "resp.pCR", "RCB.score",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2"
            ]
        },
        {
            "data_name": "multi_TransNEO",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.score",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "resp.pCR", "RCB.score",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2"
            ]
        },
        {
            "data_name": "multi_ARTemis",
            "treatment_plans": ["TP1", "TP2", "TP3", "TP4"],
            "outcome_col": "RCB.score",
            "remove_cols": [
                "Trial.ID", "resp.Chemosensitive", "resp.Chemoresistant", "resp.pCR", "RCB.score",
                "Chemo.NumCycles", "Chemo.first.Taxane", "Chemo.first.Anthracycline",
                "Chemo.second.Taxane", "Chemo.second.Anthracycline",
                "Chemo.any.Anthracycline", "Chemo.any.antiHER2"
            ]
        }
    ]

    for cfg in datasets:
        print(f"Running pipeline for dataset: {cfg['data_name']}")
        run_crossval_pipeline_entry(
            data_name=cfg['data_name'],
            treatment_plans=cfg['treatment_plans'],
            outcome_col=cfg['outcome_col'],
            remove_cols=cfg['remove_cols']
        )


# ---------------------------- Run Experiment ----------------------------
run_all_datasets()