import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pygeohash as gh
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


def decode_geo(g):
    try:
        return gh.decode(g)
    except Exception:
        return np.nan, np.nan


def prepare_data():
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")

    for df in [train, test]:
        df["hour"] = df["timestamp"].str.split(":").str[0].astype(int)
        df["minute"] = df["timestamp"].str.split(":").str[1].astype(int)
        df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
        df["day_of_week"] = df["day"] % 7
        df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
        df["slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
        df["slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)
        df["time_slot_diff"] = df["time_slot"] - 8
        df["slot_sq"] = df["time_slot_diff"] ** 2

    for col in ["RoadType", "Weather"]:
        mode_by_geo = train.groupby("geohash")[col].agg(
            lambda x: x.mode()[0] if not x.mode().empty else np.nan
        )
        global_mode = train[col].mode()[0]
        train[col] = train[col].fillna(train["geohash"].map(mode_by_geo)).fillna(global_mode)
        test[col] = test[col].fillna(test["geohash"].map(mode_by_geo)).fillna(global_mode)

    temp_median = train.groupby("geohash")["Temperature"].median()
    global_temp_median = train["Temperature"].median()
    train["Temperature"] = (
        train["Temperature"].fillna(train["geohash"].map(temp_median)).fillna(global_temp_median)
    )
    test["Temperature"] = (
        test["Temperature"].fillna(test["geohash"].map(temp_median)).fillna(global_temp_median)
    )

    train["lat"], train["lng"] = zip(*train["geohash"].map(decode_geo))
    test["lat"], test["lng"] = zip(*test["geohash"].map(decode_geo))

    train_day48 = train[train["day"] == 48].copy()
    hubs = ["qp09d9", "qp09ft", "qp09e5", "qp09d8", "qp096x"]
    for idx, hub in enumerate(hubs):
        h_lat, h_lng = gh.decode(hub)
        for df in [train, test]:
            df[f"dist_to_hub_{idx}"] = np.sqrt((df["lat"] - h_lat) ** 2 + (df["lng"] - h_lng) ** 2)

    geo_stats = train_day48.groupby("geohash")["demand"].agg(
        ["mean", "median", "std", "max", "min"]
    ).reset_index()
    geo_stats.columns = [
        "geohash",
        "geo_demand_mean",
        "geo_demand_median",
        "geo_demand_std",
        "geo_demand_max",
        "geo_demand_min",
    ]
    train = train.merge(geo_stats, on="geohash", how="left")
    test = test.merge(geo_stats, on="geohash", how="left")

    slot_stats = train_day48.groupby("time_slot")["demand"].agg(["mean", "median", "std"]).reset_index()
    slot_stats.columns = ["time_slot", "slot_demand_mean", "slot_demand_median", "slot_demand_std"]
    train = train.merge(slot_stats, on="time_slot", how="left")
    test = test.merge(slot_stats, on="time_slot", how="left")

    for col in [
        "geo_demand_mean",
        "geo_demand_median",
        "geo_demand_std",
        "geo_demand_max",
        "geo_demand_min",
        "slot_demand_mean",
        "slot_demand_median",
        "slot_demand_std",
    ]:
        fill = train_day48["demand"].mean() if "mean" in col else train[col].median()
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)

    lag_df = train_day48[["geohash", "time_slot", "demand"]].copy()
    lag_df.columns = ["geohash", "time_slot", "demand_lag_96"]
    lag_df_95 = train_day48[["geohash", "time_slot", "demand"]].copy()
    lag_df_95["time_slot"] = (lag_df_95["time_slot"] + 1) % 96
    lag_df_95.columns = ["geohash", "time_slot", "demand_lag_95"]
    lag_df_97 = train_day48[["geohash", "time_slot", "demand"]].copy()
    lag_df_97["time_slot"] = (lag_df_97["time_slot"] - 1 + 96) % 96
    lag_df_97.columns = ["geohash", "time_slot", "demand_lag_97"]

    train_day48_full = train[train["day"] == 48].copy()
    train_day49_full = train[train["day"] == 49].copy()
    for col in ["demand_lag_95", "demand_lag_96", "demand_lag_97"]:
        train_day48_full[col] = np.nan

    for lag in [lag_df, lag_df_95, lag_df_97]:
        train_day49_full = train_day49_full.merge(lag, on=["geohash", "time_slot"], how="left")
        test = test.merge(lag, on=["geohash", "time_slot"], how="left")

    for col in ["demand_lag_95", "demand_lag_96", "demand_lag_97"]:
        train_day49_full[col] = train_day49_full[col].fillna(train_day49_full["geo_demand_mean"])
        test[col] = test[col].fillna(test["geo_demand_mean"])
        train_day48_full[col] = train_day48_full["geo_demand_mean"]

    def morning_features(df, prefix=""):
        morning = df[df["time_slot"] <= 8]
        stats_df = morning.groupby("geohash")["demand"].agg(["mean", "std", "max", "median"]).reset_index()
        stats_df.columns = ["geohash", "morning_mean", "morning_std", "morning_max", "morning_median"]
        latest = df[df["time_slot"] == 8][["geohash", "demand"]].copy()
        latest.columns = ["geohash", "latest_demand"]
        return stats_df.merge(latest, on="geohash", how="left")

    morning_stats_d48 = morning_features(train_day48_full)
    morning_stats_d49 = morning_features(train_day49_full)
    train_day48_full = train_day48_full.merge(morning_stats_d48, on="geohash", how="left")
    train_day49_full = train_day49_full.merge(morning_stats_d49, on="geohash", how="left")
    test = test.merge(morning_stats_d49, on="geohash", how="left")
    train = pd.concat([train_day48_full, train_day49_full], axis=0).reset_index(drop=True)

    for col in ["morning_mean", "morning_std", "morning_max", "morning_median", "latest_demand"]:
        train[col] = train[col].fillna(train["geo_demand_mean"])
        test[col] = test[col].fillna(test["geo_demand_mean"])

    for df in [train, test]:
        df["morning_mean_ratio"] = df["morning_mean"] / (df["geo_demand_mean"] + 1e-5)
        df["latest_demand_ratio"] = df["latest_demand"] / (df["geo_demand_mean"] + 1e-5)
        df["morning_max_ratio"] = df["morning_max"] / (df["geo_demand_max"] + 1e-5)
        df["morning_std_ratio"] = df["morning_std"] / (df["geo_demand_std"] + 1e-5)
        df["morning_mean_diff"] = df["morning_mean"] - df["geo_demand_mean"]
        df["latest_demand_diff"] = df["latest_demand"] - df["geo_demand_mean"]
        df["morning_mean_slot_cos"] = df["morning_mean"] * df["slot_cos"]
        df["morning_mean_slot_sin"] = df["morning_mean"] * df["slot_sin"]
        df["latest_demand_slot_cos"] = df["latest_demand"] * df["slot_cos"]
        df["latest_demand_slot_sin"] = df["latest_demand"] * df["slot_sin"]
        df["geo_demand_mean_slot_cos"] = df["geo_demand_mean"] * df["slot_cos"]
        df["geo_demand_mean_slot_sin"] = df["geo_demand_mean"] * df["slot_sin"]
        df["lag_96_ratio"] = df["demand_lag_96"] / (df["geo_demand_mean"] + 1e-5)
        df["lag_95_ratio"] = df["demand_lag_95"] / (df["geo_demand_mean"] + 1e-5)
        df["lag_97_ratio"] = df["demand_lag_97"] / (df["geo_demand_mean"] + 1e-5)
        df["lag_mean"] = df[["demand_lag_95", "demand_lag_96", "demand_lag_97"]].mean(axis=1)
        df["lag_std"] = df[["demand_lag_95", "demand_lag_96", "demand_lag_97"]].std(axis=1)
        df["lag_minus_geo"] = df["lag_mean"] - df["geo_demand_mean"]
        df["temp_x_weather_slot"] = df["Temperature"] * df["slot_sin"]

    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather", "geohash"]:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[col + "_enc"] = le.transform(train[col].astype(str))
        test[col + "_enc"] = le.transform(test[col].astype(str))

    for col in ["RoadType", "Weather", "LargeVehicles", "Landmarks", "geohash"]:
        col_mean = train_day48.groupby(col)["demand"].mean()
        global_col_mean = train_day48["demand"].mean()
        train[col + "_target_enc"] = train[col].map(col_mean).fillna(global_col_mean)
        test[col + "_target_enc"] = test[col].map(col_mean).fillna(global_col_mean)

    features = [
        "hour", "minute", "time_slot", "hour_sin", "hour_cos", "slot_sin", "slot_cos",
        "time_slot_diff", "day_of_week", "is_weekend", "lat", "lng",
        "geo_demand_mean", "geo_demand_median", "geo_demand_std", "geo_demand_max",
        "demand_lag_95", "demand_lag_96", "demand_lag_97",
        "morning_mean", "morning_std", "morning_max", "morning_median", "latest_demand",
        "morning_mean_ratio", "latest_demand_ratio", "morning_max_ratio", "morning_std_ratio",
        "morning_mean_diff", "latest_demand_diff",
        "morning_mean_slot_cos", "morning_mean_slot_sin",
        "latest_demand_slot_cos", "latest_demand_slot_sin",
        "geo_demand_mean_slot_cos", "geo_demand_mean_slot_sin",
        "lag_96_ratio", "lag_95_ratio", "lag_97_ratio",
        "RoadType_enc", "NumberofLanes", "LargeVehicles_enc", "Landmarks_enc", "Temperature",
        "RoadType_target_enc", "Weather_target_enc", "LargeVehicles_target_enc", "Landmarks_target_enc",
        "dist_to_hub_0", "dist_to_hub_1", "dist_to_hub_2", "dist_to_hub_3", "dist_to_hub_4",
    ]
    train["demand_transformed"], best_lambda = stats.boxcox(train["demand"] + 1e-6)
    return train, test, features, best_lambda


def inverse(preds, lam):
    return np.maximum(inv_boxcox(preds, lam) - 1e-6, 0)


def main():
    train, test, features, lam = prepare_data()
    train_set = train[train["day"] == 48]
    val_set = train[train["day"] == 49]

    xgb = XGBRegressor(
        n_estimators=1200,
        learning_rate=0.025,
        max_depth=7,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        reg_alpha=0.2,
        reg_lambda=1.5,
        tree_method="hist",
        n_jobs=4,
        random_state=42,
    )
    lgb = LGBMRegressor(
        n_estimators=1900,
        learning_rate=0.018,
        num_leaves=11,
        min_child_samples=12,
        subsample=0.92,
        colsample_bytree=0.92,
        reg_alpha=0.03,
        reg_lambda=0.7,
        n_jobs=4,
        random_state=52,
        verbose=-1,
    )
    cat = CatBoostRegressor(
        iterations=1200,
        learning_rate=0.025,
        depth=7,
        l2_leaf_reg=4,
        loss_function="RMSE",
        random_seed=62,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )

    models = [("lgb", lgb)]
    val_preds = []
    for name, model in models:
        model.fit(train_set[features], train_set["demand_transformed"])
        pred = inverse(model.predict(val_set[features]), lam)
        val_preds.append(pred)
        print(f"{name} R2: {r2_score(val_set['demand'], pred):.6f}", flush=True)

    x_full = train[features]
    y_full = train["demand_transformed"]
    predictions = []
    for _, model in models:
        model.fit(x_full, y_full)
        predictions.append(inverse(model.predict(test[features]), lam))

    final_preds = predictions[0]
    final_preds = np.clip(final_preds, 0, train["demand"].max())

    submission = pd.DataFrame({"Index": test["Index"], "demand": final_preds})
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv")
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    main()
