"""Run the full mushroom classification workflow end to end.

Usage:  python main.py            (expects mushrooms.csv in this folder)
Outputs: CSV results in outputs/, figures in figures/, model in best_mushroom_model.pkl
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import cross_val_score

from src.data_preprocessing import (FIELD_FEATURES, build_preprocessor, clean_data,
                                    get_cv, load_data, split_data)
from src.eda import (descriptive_stats, feature_association_matrix, rare_categories,
                     save_eda_figures, target_association)
from src.evaluation import mcnemar_table, metrics_table, save_evaluation_figures, timing_table
from src.models import build_ensemble, build_pipelines, tune_models

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)
OUT = Path("outputs")
OUT.mkdir(exist_ok=True)


def run_scenario(name, X_train, X_test, y_train, y_test, cv, scoring):
    print(f"\n{'=' * 70}\nSCENARIO: {name}  ({X_train.shape[1]} features)\n{'=' * 70}")
    pipelines = build_pipelines(build_preprocessor(X_train.columns))

    default_cv = {n: cross_val_score(p, X_train, y_train, cv=cv).mean()
                  for n, p in pipelines.items()}
    grids = tune_models(pipelines, X_train, y_train, cv, scoring=scoring)
    tuned = {n: g.best_estimator_ for n, g in grids.items()}

    tuning = pd.DataFrame([{
        "Model": n,
        "Default CV Acc": round(default_cv[n], 5),
        "Tuned CV Acc": round(cross_val_score(tuned[n], X_train, y_train, cv=cv).mean(), 5),
        "Best Params": grids[n].best_params_,
    } for n in tuned]).sort_values("Tuned CV Acc", ascending=False)
    print("\nTuning summary:\n", tuning.to_string(index=False))

    # Pick the two best models by CV (never by test set) for the ensemble
    members = tuning["Model"].head(2).tolist()
    print(f"\nEnsemble members: {members}  |  excluded: {tuning['Model'].iloc[2]}")
    ensemble = build_ensemble(tuned, members).fit(X_train, y_train)

    all_models = {**tuned, "Ensemble (Soft Voting)": ensemble}
    results = metrics_table(all_models, X_train, y_train, X_test, y_test)
    print("\nTest-set comparison:\n", results.round(4).to_string(index=False))
    print("\nTiming:\n", timing_table(all_models, X_train, y_train, X_test).to_string(index=False))
    print("\nMcNemar (ensemble vs each base model):\n",
          mcnemar_table(ensemble, tuned, X_test, y_test).to_string(index=False))

    tag = name.lower().replace(" ", "_")
    tuning.to_csv(OUT / f"tuning_{tag}.csv", index=False)
    results.to_csv(OUT / f"model_comparison_{tag}.csv", index=False)
    save_evaluation_figures(all_models, X_train, y_train, X_test, y_test, cv, prefix=f"{tag}_")
    return tuned, ensemble


def main():
    df = load_data("mushrooms.csv")
    print("Dataset shape:", df.shape)

    # ---- EDA ----
    print("\nDescriptive statistics:\n", descriptive_stats(df).to_string())
    print("\nRare categories (<1% of rows):\n", rare_categories(df).to_string(index=False))
    assoc_target = target_association(df)
    print("\nCramér's V with class:\n", assoc_target.to_string(index=False))
    save_eda_figures(df, assoc_target, feature_association_matrix(df))

    # ---- Preparation ----
    df_clean = clean_data(df)
    df_clean.to_csv("mushrooms_cleaned.csv", index=False)
    X_train, X_test, y_train, y_test = split_data(df_clean)
    cv = get_cv()

    # ---- Scenario 1: all features ----
    tuned, _ = run_scenario("All features", X_train, X_test, y_train, y_test, cv, "accuracy")
    joblib.dump(tuned["Logistic Regression"], "best_mushroom_model.pkl")

    # ---- Scenario 2: field-observable features only (harder problem) ----
    run_scenario("Field features", X_train[FIELD_FEATURES], X_test[FIELD_FEATURES],
                 y_train, y_test, cv, "neg_log_loss")

    print("\nDone. Results in outputs/, figures in figures/.")


if __name__ == "__main__":
    main()
