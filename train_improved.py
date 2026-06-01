import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import pygeohash as gh
from catboost import CatBoostRegressor, Pool
from lightgbm import LGBMRegressor
from scipy import stats
from scipy.special import inv_boxcox
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor


SEED = 42
CAT_COLS = ["geohash", "RoadType", "LargeVehicles", "Landmarks", "Weather"]


def add_time_features(df):
    parts = df["timestamp"].str.split(":", expand=True).astype(int)
    df["hour"] = parts[0]
    df["minute"] = parts[1]
    df["time_slot"] = df["hour"] * 4 + df["minute"] // 15
    df["slot_from_start"] = df["time_slot"] - 8
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["slot_sin"] = np.sin(2 * np.pi * df["time_slot"] / 96)
    df["slot_cos"] = np.cos(2 * np.pi * df["time_slot"] / 96)
    df["is_known_morning"] = (df["time_slot"] <= 8).astype(int)
    return df


def decode_geo(series):
    coords = series.map(lambda g: gh.decode(g))
    return pd.DataFrame(coords.tolist(), columns=["lat", "lng"], index=series.index)


def fill_missing(train, test):
    for col in ["RoadType", "LargeVehicles", "Landmarks", "Weather"]:
        by_geo = train.groupby("geohash")[col].agg(
            lambda s: s.mode().iat[0] if not s.mode().empty else np.nan
        )
        global_mode = train[col].mode(dropna=True).iat[0]
        for df in (train, test):
            df[col] = df[col].fillna(df["geohash"].map(by_geo)).fillna(global_mode)

    temp_by_geo = train.groupby("geohash")["Temperature"].median()
    global_temp = train["Temperature"].median()
    for df in (train, test):
        df["Temperature"] = (
            df["Temperature"].fillna(df["geohash"].map(temp_by_geo)).fillna(global_temp)
        )
    return train, test


def merge_stats(train, test):
    day48 = train[train["day"] == 48].copy()
    day49_known = train[train["day"] == 49].copy()

    geo_stats = day48.groupby("geohash")["demand"].agg(
        geo_mean="mean",
        geo_median="median",
        geo_std="std",
        geo_min="min",
        geo_max="max",
        geo_q25=lambda s: s.quantile(0.25),
        geo_q75=lambda s: s.quantile(0.75),
    )

    slot_stats = day48.groupby("time_slot")["demand"].agg(
        slot_mean="mean", slot_median="median", slot_std="std"
    )

    geo_slot = day48[["geohash", "time_slot", "demand"]].rename(
        columns={"demand": "d48_same_slot"}
    )
    geo_slot_prev = day48[["geohash", "time_slot", "demand"]].copy()
    geo_slot_prev["time_slot"] = (geo_slot_prev["time_slot"] + 1) % 96
    geo_slot_prev = geo_slot_prev.rename(columns={"demand": "d48_prev_slot"})
    geo_slot_next = day48[["geohash", "time_slot", "demand"]].copy()
    geo_slot_next["time_slot"] = (geo_slot_next["time_slot"] - 1) % 96
    geo_slot_next = geo_slot_next.rename(columns={"demand": "d48_next_slot"})

    morning48 = day48[day48["time_slot"] <= 8].groupby("geohash")["demand"].agg(
        d48_morning_mean="mean",
        d48_morning_sum="sum",
        d48_morning_std="std",
        d48_morning_max="max",
        d48_latest=lambda s: s.iloc[-1],
    )
    morning49 = day49_known.groupby("geohash")["demand"].agg(
        d49_morning_mean="mean",
        d49_morning_sum="sum",
        d49_morning_std="std",
        d49_morning_max="max",
        d49_latest=lambda s: s.iloc[-1],
    )

    geos = pd.concat([geo_stats, morning48, morning49], axis=1)
    global_values = geos.median(numeric_only=True)
    geos = geos.fillna(global_values)
    geos["morning_ratio"] = geos["d49_morning_sum"] / (geos["d48_morning_sum"] + 1e-5)
    geos["morning_ratio"] = geos["morning_ratio"].clip(0.35, 2.5)
    geos["latest_ratio"] = geos["d49_latest"] / (geos["d48_latest"] + 1e-5)
    geos["latest_ratio"] = geos["latest_ratio"].clip(0.35, 2.5)
    geos["morning_delta"] = geos["d49_morning_mean"] - geos["d48_morning_mean"]
    geos["latest_delta"] = geos["d49_latest"] - geos["d48_latest"]

    def add_merged_features(df):
        df = df.merge(geos.reset_index(), on="geohash", how="left")
        df = df.merge(slot_stats.reset_index(), on="time_slot", how="left")
        df = df.merge(geo_slot, on=["geohash", "time_slot"], how="left")
        df = df.merge(geo_slot_prev, on=["geohash", "time_slot"], how="left")
        df = df.merge(geo_slot_next, on=["geohash", "time_slot"], how="left")
        return df

    train = add_merged_features(train)
    test = add_merged_features(test)

    numeric_cols = [
        c for c in train.columns if c in test.columns and train[c].dtype.kind in "fc"
    ]
    for col in numeric_cols:
        fill = train[col].median()
        train[col] = train[col].fillna(fill)
        test[col] = test[col].fillna(fill)

    for df in (train, test):
        use_day49 = df["day"].eq(49)
        df["current_morning_mean"] = np.where(
            use_day49, df["d49_morning_mean"], df["d48_morning_mean"]
        )
        df["current_morning_sum"] = np.where(
            use_day49, df["d49_morning_sum"], df["d48_morning_sum"]
        )
        df["current_morning_std"] = np.where(
            use_day49, df["d49_morning_std"], df["d48_morning_std"]
        )
        df["current_morning_max"] = np.where(
            use_day49, df["d49_morning_max"], df["d48_morning_max"]
        )
        df["current_latest"] = np.where(use_day49, df["d49_latest"], df["d48_latest"])
        df["current_to_d48_morning"] = df["current_morning_sum"] / (
            df["d48_morning_sum"] + 1e-5
        )
        df["current_to_d48_latest"] = df["current_latest"] / (df["d48_latest"] + 1e-5)
        df["d48_scaled_by_morning"] = df["d48_same_slot"] * df["morning_ratio"]
        df["d48_scaled_by_latest"] = df["d48_same_slot"] * df["latest_ratio"]
        df["d48_plus_morning_delta"] = df["d48_same_slot"] + df["morning_delta"]
        df["d48_to_geo_mean"] = df["d48_same_slot"] / (df["geo_mean"] + 1e-5)
        df["current_morning_to_geo_mean"] = df["current_morning_mean"] / (
            df["geo_mean"] + 1e-5
        )
        df["slot_geo_interaction"] = df["slot_mean"] * df["geo_mean"]
    return train, test


def build_features(train, test):
    train = add_time_features(train.copy())
    test = add_time_features(test.copy())
    train, test = fill_missing(train, test)

    coords_train = decode_geo(train["geohash"])
    coords_test = decode_geo(test["geohash"])
    train[["lat", "lng"]] = coords_train
    test[["lat", "lng"]] = coords_test

    train, test = merge_stats(train, test)

    for col in CAT_COLS:
        le = LabelEncoder()
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        le.fit(combined)
        train[f"{col}_le"] = le.transform(train[col].astype(str))
        test[f"{col}_le"] = le.transform(test[col].astype(str))

    target_day48 = train[train["day"] == 48]
    global_mean = target_day48["demand"].mean()
    for col in CAT_COLS:
        enc = target_day48.groupby(col)["demand"].mean()
        train[f"{col}_te"] = train[col].map(enc).fillna(global_mean)
        test[f"{col}_te"] = test[col].map(enc).fillna(global_mean)

    raw_morning_cols = {
        "d48_morning_mean",
        "d48_morning_sum",
        "d48_morning_std",
        "d48_morning_max",
        "d48_latest",
        "d49_morning_mean",
        "d49_morning_sum",
        "d49_morning_std",
        "d49_morning_max",
        "d49_latest",
    }
    feature_cols = [
        c
        for c in train.columns
        if c not in ["demand", "timestamp", "Index"]
        and c not in raw_morning_cols
        and (train[c].dtype.kind in "ifc" or c in CAT_COLS)
    ]
    return train, test, feature_cols


def train_and_predict():
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")
    train, test, features = build_features(train_raw, test_raw)

    train["target_bc"], lam = stats.boxcox(train["demand"] + 1e-6)

    val_mask = train["day"].eq(49)
    tr_mask = train["day"].eq(48)
    xgb_features = [c for c in features if c not in CAT_COLS]

    sample_weight = np.where(
        train.loc[tr_mask, "time_slot"].between(9, 55), 2.0, 1.0
    )

    xgb = XGBRegressor(
        n_estimators=450,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=2,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        tree_method="hist",
        n_jobs=4,
        random_state=SEED,
    )
    xgb.fit(
        train.loc[tr_mask, xgb_features],
        train.loc[tr_mask, "target_bc"],
        sample_weight=sample_weight,
        verbose=False,
    )

    val_xgb = np.maximum(inv_boxcox(xgb.predict(train.loc[val_mask, xgb_features]), lam) - 1e-6, 0)
    print("Day49 known morning XGB R2:", round(r2_score(train.loc[val_mask, "demand"], val_xgb), 6), flush=True)

    lgb = LGBMRegressor(
        n_estimators=450,
        learning_rate=0.05,
        num_leaves=64,
        max_depth=-1,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_alpha=0.02,
        reg_lambda=0.8,
        random_state=SEED,
        n_jobs=4,
        verbose=-1,
    )
    lgb.fit(
        train.loc[tr_mask, xgb_features],
        train.loc[tr_mask, "target_bc"],
        sample_weight=sample_weight,
    )
    val_lgb = np.maximum(inv_boxcox(lgb.predict(train.loc[val_mask, xgb_features]), lam) - 1e-6, 0)
    print("Day49 known morning LGBM R2:", round(r2_score(train.loc[val_mask, "demand"], val_lgb), 6), flush=True)

    cat_features_idx = [features.index(c) for c in CAT_COLS]
    cat = CatBoostRegressor(
        iterations=450,
        learning_rate=0.06,
        depth=8,
        loss_function="RMSE",
        random_seed=SEED,
        thread_count=4,
        verbose=False,
        allow_writing_files=False,
    )
    cat.fit(
        Pool(train.loc[tr_mask, features], train.loc[tr_mask, "target_bc"], cat_features=cat_features_idx, weight=sample_weight)
    )
    val_cat = np.maximum(inv_boxcox(cat.predict(Pool(train.loc[val_mask, features], cat_features=cat_features_idx)), lam) - 1e-6, 0)
    print("Day49 known morning CatBoost R2:", round(r2_score(train.loc[val_mask, "demand"], val_cat), 6), flush=True)

    blend = 0.35 * val_xgb + 0.25 * val_lgb + 0.40 * val_cat
    print("Day49 known morning blend R2:", round(r2_score(train.loc[val_mask, "demand"], blend), 6), flush=True)

    full_weight = np.ones(len(train))
    full_weight[train["day"].eq(49)] = 3.0
    full_weight[train["time_slot"].between(9, 55)] *= 1.5

    xgb.fit(train[xgb_features], train["target_bc"], sample_weight=full_weight, verbose=False)
    lgb.fit(train[xgb_features], train["target_bc"], sample_weight=full_weight)
    cat.fit(Pool(train[features], train["target_bc"], cat_features=cat_features_idx, weight=full_weight))

    pred_xgb = np.maximum(inv_boxcox(xgb.predict(test[xgb_features]), lam) - 1e-6, 0)
    pred_lgb = np.maximum(inv_boxcox(lgb.predict(test[xgb_features]), lam) - 1e-6, 0)
    pred_cat = np.maximum(inv_boxcox(cat.predict(Pool(test[features], cat_features=cat_features_idx)), lam) - 1e-6, 0)

    final = 0.35 * pred_xgb + 0.25 * pred_lgb + 0.40 * pred_cat
    final = np.clip(final, 0, train["demand"].max())

    submission = pd.DataFrame({"Index": test["Index"], "demand": final})
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv", submission.shape)
    print(submission.head().to_string(index=False))


if __name__ == "__main__":
    train_and_predict()
