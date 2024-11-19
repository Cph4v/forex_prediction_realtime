import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import time,gc
from utils.general_utils import cal_eval
import numpy as np

def split_time_series(
    df_all: pd.DataFrame,
    max_train_size: int,
    n_splits: int,
    test_size: int,
    train_test_gap: int,
):
    """
    Return a nested dictionary key is k number and value is dicitonary of train, valid and test Dates
    :max_train_size: maximum size we for train
    :n_splits: K in cross-folds
    :test_size: test size
    train_test_gap is the gap between train and valid/test sets 
    """
    all_dates = df_all.index.get_level_values("_time").unique().sort_values(["_time"])
    tscv = TimeSeriesSplit(
        gap=train_test_gap,
        max_train_size=max_train_size,
        n_splits=n_splits,
        test_size=test_size*2,
    )
    folds = {}
    for i, (train_index, test_valid_index) in enumerate(tscv.split(all_dates[0])):
        folds[i] = {
            "train_dates": all_dates[0][train_index],
            "valid_dates": all_dates[0][test_valid_index[:test_size]],
            "test_dates": all_dates[0][test_valid_index[test_size:]],
        }

    return folds

def quant_CV(
        df: pd.DataFrame,
        folds: dict[int,pd.DatetimeIndex],
        model,
        non_feature_columns: list[str],
        early_stopping_rounds: int|None = None,

        ):
    """
    This function runs Time Series CV with available embargo/purge 
    It also backtest model signals on each fold and the whole test and valid sets 
    """
    evals = pd.DataFrame(
        columns=[
            "dataset",
            "K",
            "f1_score",
            "precision",
            "recall",
            "Min_date",
            "Max_date",
            "train_duration",
            "n_unique_days",
            "n_max_daily_sig",
        ]
    )
    df["pred_as_val"] = -1
    df["pred_val_proba"] = -1
    df["pred_as_test"] = -1
    df["pred_test_proba"] = -1
    df["K"] = -1

    the_features = df.drop(columns=non_feature_columns).columns
    feature_importances = {feature: [] for feature in the_features}

    for i in list(folds.keys()):
        print("= "*35)
        print(f"Fold {i}:")
        tic = time.time()
        # sets,min_max_dates = data_split_loader(df,folds,i)

        train_min_max = [folds[i]["train_dates"].min(), folds[i]["train_dates"].max()]
        valid_min_max = [folds[i]["valid_dates"].min(), folds[i]["valid_dates"].max()]
        test_min_max = [folds[i]["test_dates"].min(), folds[i]["test_dates"].max()]
        min_max_dates = {
            "train_dates": train_min_max,
            "valid_dates": valid_min_max,
            "test_dates": test_min_max,
        }

        print(f"--> fold train size: {df.loc[folds[i]['train_dates']].shape}")
        print(f"--> fold valid size: {df.loc[folds[i]['valid_dates']].shape}")
        print(f"--> fold test size: {df.loc[folds[i]['test_dates']].shape}")

        if early_stopping_rounds is not None:
            print("early_stopping_rounds: ", early_stopping_rounds)

            eval_set = [
                (
                    df.loc[folds[i]["valid_dates"]].drop(
                        columns=non_feature_columns
                    ),
                    df.loc[folds[i]["valid_dates"]]["target"],
                )
            ]

            model.fit(
                df.loc[folds[i]["train_dates"]].drop(
                    columns=non_feature_columns
                ),
            )

        try:
            input_cols = model.feature_names_in_
        except:
            input_cols = model.feature_name_

        # Store feature importances for this fold
        if hasattr(model, 'feature_importances_'):
            for feature, importance in zip(input_cols, model.feature_importances_):
                feature_importances[feature].append(importance)

        toc = time.time()
        gc.collect()
        # repetetive part I can improve by a function
        for set_name in ["train_dates", "valid_dates", "test_dates"]:
            set_name_dict = {
                "train_dates": "train",
                "valid_dates": "valid",
                "test_dates": "test",
            }
            y_pred = model.predict(df.loc[folds[i][set_name]][input_cols]).reshape(
                -1, 1
            )

            y_real = df.loc[folds[i][set_name]][["target"]]    

            if set_name in ["valid_dates", "test_dates"]:
                pred_name = {
                "valid_dates": "val",
                "test_dates": "test"}
                df.loc[folds[i][set_name], "K"] = i
                df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] = y_pred
                #TODO: Add conformal prediction
                proba_pred = model.predict_proba(df.loc[folds[i][set_name]][input_cols])

                if np.shape(proba_pred)[1] > 1:
                    df.loc[
                        folds[i][set_name], f"pred_{pred_name[set_name]}_proba"
                    ] = proba_pred[:, 1]
                else:
                    print("Proba doesn't have class1")
                    df.loc[folds[i][set_name], f"pred_{pred_name[set_name]}_proba"] = 0

                # Calculate n_unique days and max daily n_signals in each fold
                fold_unique_days = pd.Series(df.loc[folds[i][set_name]].loc[
                            df.loc[folds[i][set_name], f"pred_as_{pred_name[set_name]}"] == 1].index.date).nunique()
                
             

                gc.collect()
            else:
                fold_unique_days = -1
                fold_max_daily_sig = None
            
            eval_list = (
                [set_name_dict[set_name], i]
                + cal_eval(y_real=y_real, y_pred=y_pred)
                + min_max_dates[set_name]
                + [str(round(toc - tic, 1))]
                + [fold_unique_days,fold_max_daily_sig]
            )

            evals.loc[len(evals)] = eval_list

        print(evals.iloc[-3:])
        input_cols_and_type = dict(df[input_cols].dtypes)

    
    # Create a DataFrame from the feature importances
    importance_df = pd.DataFrame(feature_importances)
    
    if hasattr(model, 'feature_importances_'):
        importance_df = importance_df.T.reset_index()
        importance_df.columns = ['feature_name'] + [f'importance_fold_{i}' for i in range(len(folds))]

        imp_cols = [f for f in importance_df if 'importance_fold' in f]
        importance_df['mean_importance'] = importance_df[imp_cols].mean(axis=1)
        importance_df['median_importance'] = importance_df[imp_cols].median(axis=1)
        importance_df['std_importance'] = importance_df[imp_cols].std(axis=1)
    
        # Calculate coefficient of variation (CV)
        importance_df['cv'] = importance_df['std_importance'] / importance_df['mean_importance']
        importance_df.sort_values('mean_importance', ascending=False, inplace=True)

    return (
        input_cols_and_type,
        input_cols,
        evals,
        df[df.pred_as_val != -1][["K", "pred_as_val", "pred_val_proba", "target"]],
        df[df.pred_as_test != -1][["K", "pred_as_test", "pred_test_proba", "target"]],
        importance_df
    )
