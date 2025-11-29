# tsforge_demo.py
# Compatible with: statsforecast 1.7.5, mlforecast 0.13.4, neuralforecast 1.7.2, pandas 2.3+

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

# External libs
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import STL

from lightgbm import LGBMRegressor  # (kept for future use)
from catboost import CatBoostRegressor  # (kept for future use)
from xgboost import XGBRegressor  # (kept for future use)

from statsforecast import StatsForecast as SFCore
from statsforecast.models import (
    Naive,
    SeasonalNaive,
    AutoETS,
    AutoARIMA,
    AutoTheta,
)
# AutoTBATS is not always available; import guarded
try:
    from statsforecast.models import AutoTBATS  # noqa: F401
except Exception:
    pass

from mlforecast import MLForecast as MLFCore
from neuralforecast import NeuralForecast as NFCore
from neuralforecast.models import NHITS

from utilsforecast.preprocessing import fill_gaps

# ------------------------------------------------------------
# fill_gaps_plus
# ------------------------------------------------------------

def fill_gaps_plus(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    static_cols: list[str] | None = None,
    dynamic_fill: dict[str, str] | None = None,
    target_fill: str = "zero",
    strategy: str = "zero",  # preferred
    freq: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Enhanced gap filling that extends Nixtla's fill_gaps to handle
    static, dynamic, and flexible fill strategies for the target.
    """
    if verbose:
        print(f"[fill_gaps_plus] Input shape: {df.shape}")

    # ensure datetime
    _to_datetime(df, time_col)

    # Base with only target
    df_base = df[[id_col, time_col, target_col]].copy()

    # Run Nixtla fill_gaps (freq required)
    if not freq:
        raise ValueError("freq must be specified (e.g., 'D', 'W', 'M')")
    filled = fill_gaps(df_base, freq=freq, id_col=id_col, time_col=time_col)

    # Target fill / strategy
    fill_strategy = (strategy or target_fill).lower().strip()
    if fill_strategy == "zero":
        filled[target_col] = filled[target_col].fillna(0)
    elif fill_strategy == "mean":
        filled[target_col] = filled.groupby(id_col)[target_col].transform(lambda x: x.fillna(x.mean()))
    elif fill_strategy == "median":
        filled[target_col] = filled.groupby(id_col)[target_col].transform(lambda x: x.fillna(x.median()))
    elif fill_strategy == "nan":
        pass
    else:
        raise ValueError(f"Unknown target fill strategy '{fill_strategy}'")

    # Add static columns if they exist
    if static_cols:
        static_cols_existing = [c for c in static_cols if c in df.columns]
        if static_cols_existing:
            static_df = df[[id_col] + static_cols_existing].drop_duplicates(subset=[id_col])
            filled = filled.merge(static_df, on=id_col, how="left")

    # Add dynamic columns if they exist
    ignore_cols = [id_col, time_col, target_col] + (static_cols or [])
    dyn_cols = [c for c in df.columns if c not in ignore_cols]
    if dyn_cols:
        dyn_df = df[[id_col, time_col] + dyn_cols].copy()
        filled = filled.merge(dyn_df, on=[id_col, time_col], how="left")

        # Apply per-column fill strategies
        if dynamic_fill:
            for col, dyn_strategy in dynamic_fill.items():
                if col not in filled.columns:
                    continue
                if dyn_strategy == "zero":
                    filled[col] = filled[col].fillna(0)
                elif dyn_strategy == "ffill":
                    filled[col] = filled.groupby(id_col)[col].ffill()
                elif dyn_strategy == "interpolate":
                    filled[col] = filled.groupby(id_col)[col].transform(lambda x: x.interpolate())
                elif dyn_strategy == "nan":
                    pass
                else:
                    raise ValueError(f"Unknown fill strategy {dyn_strategy} for column {col}")

    if verbose:
        added = filled.shape[0] - df.shape[0]
        print(f"[fill_gaps_plus] Output shape: {filled.shape}")
        print(f"[fill_gaps_plus] Added {added} rows using strategy='{fill_strategy}'.")

    return filled


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def _safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)

def _ensure_cols(df: pd.DataFrame, cols: Sequence[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

def _to_datetime(df: pd.DataFrame, col: str):
    if not pd.api.types.is_datetime64_any_dtype(df[col]):
        df[col] = pd.to_datetime(df[col])

def make_future_frame(
    df: pd.DataFrame, h: int, freq: str,
    id_col: str = "unique_id", time_col: str = "ds"
) -> pd.DataFrame:
    _ensure_cols(df, [id_col, time_col])
    _to_datetime(df, time_col)
    last = df.groupby(id_col)[time_col].max().rename("last_ds").to_frame()
    rows = []
    for uid, last_ds in last.itertuples():
        future_dates = pd.date_range(last_ds + pd.tseries.frequencies.to_offset(freq), periods=h, freq=freq)
        rows.append(pd.DataFrame({id_col: uid, time_col: future_dates}))
    return pd.concat(rows, ignore_index=True)

def _empirical_bias(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.nanmean(yhat - y))

def _mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.nanmean(np.abs(yhat - y)))

def _rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean((yhat - y) ** 2)))

def _mape(y: np.ndarray, yhat: np.ndarray) -> float:
    mask = y != 0
    return float(np.nanmean(np.abs((yhat[mask] - y[mask]) / y[mask]))) if mask.any() else np.nan


# ------------------------------------------------------------
# CV Plan
# ------------------------------------------------------------

@dataclass
class CVPlan:
    cutoffs: List[pd.Timestamp]
    h: int
    step_size: int
    freq: str

    @property
    def min_cutoff(self) -> pd.Timestamp:
        return min(self.cutoffs)

    @classmethod
    def from_nixtla(
        cls,
        df: pd.DataFrame,
        id_col: str,
        time_col: str,
        target_col: str,
        h: int,
        n_windows: int,
        step_size: int,
        freq: str,
    ) -> "CVPlan":
        _ensure_cols(df, [id_col, time_col, target_col])
        _to_datetime(df, time_col)
        last_ds = df[time_col].max()
        cutoffs = [last_ds - pd.to_timedelta(step_size * (n_windows - i), unit="D") for i in range(n_windows)]
        return cls(cutoffs=cutoffs, h=h, step_size=step_size, freq=freq)


# ------------------------------------------------------------
# Recipe system
# ------------------------------------------------------------

class Recipe:
    def __init__(self, name: str = ""):
        self.name = name
        self.steps: List[Tuple[str, Callable]] = []
        self._target_transform: Optional[str] = None

    def add_recipe(self, other: "Recipe") -> "Recipe":
        self.steps.extend(other.steps)
        if other._target_transform and not self._target_transform:
            self._target_transform = other._target_transform
        return self

    def fill_gaps_plus(self, strategy: str = "zero", **kwargs) -> "Recipe":
        def _fn(df, id_col, time_col, target_col):
            if "fill_gaps_plus" not in globals():
                raise RuntimeError("fill_gaps_plus not found in global scope.")
            return globals()["fill_gaps_plus"](
                df, id_col=id_col, time_col=time_col, target_col=target_col,
                strategy=strategy, **kwargs
            )
        self.steps.append(("fill_gaps_plus", _fn))
        return self

    def detect_outliers(self, method="zscore", threshold=3.0, deseasonalize=False, seasonal_period=7) -> "Recipe":
        def _fn(df, id_col, time_col, target_col):
            return _detect_outliers_df(
                df, id_col, time_col, target_col,
                method=method, threshold=threshold,
                deseasonalize=deseasonalize, seasonal_period=seasonal_period
            )
        self.steps.append(("detect_outliers", _fn))
        return self

    def correct_outliers(self, strategy: str = "linear_interp") -> "Recipe":
        def _fn(df, id_col, time_col, target_col):
            return _correct_outliers_df(df, id_col, time_col, target_col, strategy=strategy)
        self.steps.append(("correct_outliers", _fn))
        return self

    def stabilize_variance(self, method="log1p") -> "Recipe":
        if method != "log1p":
            raise NotImplementedError("Only method='log1p' supported in this demo.")
        self._target_transform = "log1p"
        def _fn(df, id_col, time_col, target_col):
            df = _safe_copy(df)
            if (df[target_col] < -0.999).any():
                raise ValueError("log1p requires nonnegative data.")
            df[target_col] = np.log1p(df[target_col])
            return df
        self.steps.append(("stabilize_variance", _fn))
        return self

    def decompose(self, signal_col: str = "y", method: str = "stl", seasonal_period: int = 7, store_only: bool = True) -> "Recipe":
        if method.lower() != "stl":
            raise NotImplementedError("Only method='stl' is supported in this demo.")
        def _fn(df, id_col, time_col, target_col):
            return _stl_decompose_df(df, id_col, time_col, signal_col, seasonal_period, add_as_features=store_only)
        self.steps.append(("decompose_stl", _fn))
        return self

    def add_dynamic_features(self, exog_df: pd.DataFrame, join_on: List[str], forward_fill: bool = True) -> "Recipe":
        def _fn(df, id_col, time_col, target_col):
            tmp = _safe_copy(df)
            ex = _safe_copy(exog_df)
            _to_datetime(tmp, time_col)
            if time_col in join_on:
                _to_datetime(ex, time_col)
            tmp = tmp.merge(ex, on=join_on, how="left")
            if forward_fill:
                tmp = tmp.sort_values([id_col, time_col]).groupby(id_col).ffill()
            return tmp
        self.steps.append(("add_dynamic_features", _fn))
        return self

    def add_exog_lags(self, cols: List[str], lags: List[int]) -> "Recipe":
        def _fn(df, id_col, time_col, target_col):
            tmp = _safe_copy(df).sort_values([id_col, time_col])
            for c in cols:
                for L in lags:
                    tmp[f"{c}_lag{L}"] = tmp.groupby(id_col)[c].shift(L)
            return tmp
        self.steps.append(("add_exog_lags", _fn))
        return self

    def add_timeseries_signature(self, time_col: str = "ds") -> "Recipe":
        def _fn(df, id_col, time_col_, target_col):
            tmp = _safe_copy(df)
            _to_datetime(tmp, time_col_)
            dt = tmp[time_col_]
            tmp["year"] = dt.dt.year
            tmp["month"] = dt.dt.month
            tmp["day"] = dt.dt.day
            tmp["dayofweek"] = dt.dt.dayofweek
            tmp["is_month_end"] = dt.dt.is_month_end.astype(int)
            tmp["is_year_end"] = dt.dt.is_year_end.astype(int)
            return tmp
        self.steps.append(("add_timeseries_signature", _fn))
        return self

    def add_dummy(self, cols: List[str], one_hot: bool = True) -> "Recipe":
        def _fn(df, id_col, time_col, target_col):
            tmp = _safe_copy(df)
            if one_hot:
                tmp = pd.get_dummies(tmp, columns=cols, drop_first=False)
            else:
                for c in cols:
                    tmp[c] = tmp[c].astype("category")
            return tmp
        self.steps.append(("add_dummy", _fn))
        return self

    def add_mean_encoder(self, cols: List[str]) -> "Recipe":
        def _fn(df, id_col, time_col, target_col):
            tmp = _safe_copy(df)
            for c in cols:
                gp = tmp.groupby(c)[target_col].mean()
                tmp[f"{c}_mean_y"] = tmp[c].map(gp)
            return tmp
        self.steps.append(("add_mean_encoder", _fn))
        return self

    def add_static_features(self, static_df: pd.DataFrame, on: Optional[List[str]] = None) -> "Recipe":
        def _fn(df, id_col, time_col, target_col):
            tmp = _safe_copy(df)
            st = _safe_copy(static_df)
            key = on if on else [id_col]
            return tmp.merge(st, on=key, how="left")
        self.steps.append(("add_static_features", _fn))
        return self

    def apply(self, df: pd.DataFrame, id_col: str, time_col: str, target_col: str) -> pd.DataFrame:
        out = _safe_copy(df)
        active_col = target_col
        for name, fn in self.steps:
            out = fn(out, id_col, time_col, active_col)
            if "_tsforge_target_col" in out.columns and out["_tsforge_target_col"].notna().any():
                new_active = out["_tsforge_target_col"].dropna().iloc[0]
                if new_active != active_col:
                    active_col = new_active
                    print(f"[apply] Active target column updated → '{active_col}'")
        return out
    # -----------------------------------
    # Step: cast selected columns to category
    # -----------------------------------
    def _categorical_cast_df(self, df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
        """Helper to safely cast selected columns to category dtype."""
        tmp = df.copy()
        for c in cols:
            if c in tmp.columns:
                before = tmp[c].dtype
                tmp[c] = tmp[c].astype("category")
                print(f"[categorical_cast] {c}: {before} → category ({tmp[c].nunique()} levels)")
            else:
                print(f"[categorical_cast] skipped missing column: {c}")
        return tmp

    def add_categorical_cast(self, cols: list[str]) -> "Recipe":
        """Convert selected columns to pandas categorical dtype."""
        def _fn(df, id_col, time_col, target_col):
            return self._categorical_cast_df(df, cols)
        self.steps.append(("categorical_cast", _fn))
        return self


    def invert_predictions(self, preds: pd.DataFrame, target_col: str = "y", yhat_col: str = "yhat") -> pd.DataFrame:
        if self._target_transform == "log1p":
            preds = _safe_copy(preds)
            preds[yhat_col] = np.expm1(preds[yhat_col])
        return preds


# ------------------------------------------------------------
# Outlier helpers
# ------------------------------------------------------------

def _series_outlier_flags(
    s: pd.Series,
    method: str = "zscore",
    threshold: float = 3.0,
) -> pd.Series:
    x = s.values.astype(float)
    mask = np.zeros_like(x, dtype=bool)
    if method == "zscore":
        mu = np.nanmean(x); sd = np.nanstd(x)
        if sd == 0 or np.isnan(sd):
            return pd.Series(mask, index=s.index)
        z = (x - mu) / sd
        mask = np.abs(z) > threshold
    elif method.lower() in ("3iqr", "iqr"):
        q1, q3 = np.nanpercentile(x, 25), np.nanpercentile(x, 75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
        mask = (x < lower) | (x > upper)
    elif method == "sigma":
        mu = np.nanmean(x); sd = np.nanstd(x)
        if sd == 0 or np.isnan(sd):
            return pd.Series(mask, index=s.index)
        mask = np.abs(x - mu) > threshold * sd
    else:
        raise ValueError(f"Unknown outlier method: {method}")
    return pd.Series(mask, index=s.index)

def _deseasonalize_if_ok(s: pd.Series, period: int):
    x = s.values.astype(float)
    if len(x) < 2 * period:
        return s, None
    nz = (x > 0).sum()
    adi = (len(x) / nz) if nz > 0 else np.inf
    if adi >= 1.3:
        return s, None
    try:
        adf_p = adfuller(x, autolag="AIC", maxlag=min(10, len(x)//4))[1]
    except Exception:
        adf_p = 1.0
    if adf_p >= 0.05:
        return s, None
    mu = np.nanmean(x); sd = np.nanstd(x)
    cv = (sd / mu) if mu != 0 else np.inf
    if cv >= 1.5:
        return s, None
    try:
        stl = STL(x, period=period, robust=True).fit()
        resid = s - stl.seasonal
        return resid, stl.seasonal
    except Exception:
        return s, None

def _detect_outliers_df(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    method: str,
    threshold: float,
    deseasonalize: bool,
    seasonal_period: int,
) -> pd.DataFrame:
    tmp = _safe_copy(df).sort_values([id_col, time_col])
    flags = []
    for _, g in tmp.groupby(id_col, group_keys=False):
        s = g[target_col]
        if deseasonalize:
            s_proc, seasonal = _deseasonalize_if_ok(s, seasonal_period)
        else:
            s_proc, seasonal = s, None
        f = _series_outlier_flags(s_proc, method=method, threshold=threshold)
        out = g.copy()
        out["_is_outlier"] = f.values
        if seasonal is not None:
            out["_seasonal_component"] = seasonal.values
        flags.append(out)
    return pd.concat(flags, ignore_index=True)

def _correct_outliers_df(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    target_col: str,
    strategy: str = "linear_interp",
) -> pd.DataFrame:
    tmp = _safe_copy(df).sort_values([id_col, time_col])
    if "_is_outlier" not in tmp.columns:
        return tmp

    def _winsorize(vals: np.ndarray, lower_q=5.0, upper_q=95.0):
        lo = np.nanpercentile(vals, lower_q)
        hi = np.nanpercentile(vals, upper_q)
        return np.clip(vals, lo, hi)

    rows = []
    for _, g in tmp.groupby(id_col, group_keys=False):
        y = g[target_col].values.astype(float)
        mask = g["_is_outlier"].values.astype(bool)
        if not mask.any():
            rows.append(g); continue

        corrected = y.copy()
        if strategy == "linear_interp":
            idx = np.arange(len(y))
            good = ~mask & ~np.isnan(y)
            if good.sum() >= 2:
                corrected[mask] = np.interp(idx[mask], idx[good], y[good])
        elif strategy == "mean":
            corrected[mask] = np.nanmean(y[~mask])
        elif strategy == "median":
            corrected[mask] = np.nanmedian(y[~mask])
        elif strategy == "winsorize":
            corrected = _winsorize(y)
        else:
            raise ValueError(f"Unknown outlier correction strategy: {strategy}")

        gg = g.copy()
        gg[target_col] = corrected
        rows.append(gg)

    out = pd.concat(rows, ignore_index=True)
    return out.drop(columns=[c for c in ["_is_outlier", "_seasonal_component"] if c in out.columns])


def _stl_decompose_df(
    df: pd.DataFrame,
    id_col: str,
    time_col: str,
    signal_col: str,
    seasonal_period: int,
    add_as_features: bool = True,
) -> pd.DataFrame:
    tmp = _safe_copy(df).sort_values([id_col, time_col])
    parts = []
    for _, g in tmp.groupby(id_col, group_keys=False):
        x = g[signal_col].astype(float).values
        if len(x) < 2 * seasonal_period:
            gg = g.copy()
            gg["trend"] = np.nan; gg["seasonal"] = np.nan; gg["resid"] = np.nan
        else:
            stl = STL(x, period=seasonal_period, robust=True).fit()
            gg = g.copy()
            gg["trend"] = stl.trend; gg["seasonal"] = stl.seasonal; gg["resid"] = stl.resid
        parts.append(gg)
    out = pd.concat(parts, ignore_index=True)
    return out if add_as_features else df


# ------------------------------------------------------------
# Model specs (thin wrappers)
# ------------------------------------------------------------

@dataclass
class StatsForecastSpec:
    models: List[object]
    freq: str = "D"
    n_jobs: int = -1

@dataclass
class MLForecastSpec:
    models: List[object]
    freq: str = "D"
    lags: List[int] = field(default_factory=lambda: [1, 7, 28])
    lag_transforms: Optional[dict] = None
    date_features: Optional[List[str]] = None
    static_features: Optional[List[str]] = None
    n_jobs: int = -1

@dataclass
class NeuralForecastSpec:
    models: List[str]  # e.g., ["NHITS"]
    freq: str = "D"
    max_steps : int = 50
    batch_size: int = 64

def StatsForecast(models: List[object], freq: str = "D", n_jobs: int = -1) -> StatsForecastSpec:
    return StatsForecastSpec(models=models, freq=freq, n_jobs=n_jobs)

def MLForecast(models: List[object], freq: str = "D", lags: List[int] = None,
               lag_transforms: Optional[dict] = None, date_features: Optional[List[str]] = None,
               static_features: Optional[List[str]] = None, n_jobs: int = -1,
               target_transforms: Optional[List] = None) -> MLForecastSpec:
    return MLForecastSpec(models=models, freq=freq, lags=lags or [1, 7, 28],
                          lag_transforms=lag_transforms, date_features=date_features,
                          static_features=static_features, n_jobs=n_jobs)

def NeuralForecast(models: List[object], freq: str = "D", max_steps : int = 50, batch_size: int = 64, **kwargs) -> NeuralForecastSpec:
    names = []
    for m in models:
        if isinstance(m, str):
            names.append(m.upper())
        else:
            names.append("NHITS" if m.__class__.__name__.upper() == "NHITS" else "NHITS")
    return NeuralForecastSpec(models=list(dict.fromkeys(names)), freq=freq, max_steps =max_steps , batch_size=batch_size)


# ------------------------------------------------------------
# Workflow
# ------------------------------------------------------------

class Workflow:
    def __init__(self, engine: str, id_col: str, time_col: str, target_col: str, name: str):
        self.engine = engine.lower()
        self.id_col = id_col
        self.time_col = time_col
        self.target_col = target_col
        self.name = name
        self.recipe_obj: Optional[Recipe] = None
        self.spec: Optional[Union[StatsForecastSpec, MLForecastSpec, NeuralForecastSpec]] = None

    def add_recipe(self, recipe: Recipe) -> "Workflow":
        self.recipe_obj = recipe
        return self

    def add_model_spec(self, spec: Union[StatsForecastSpec, MLForecastSpec, NeuralForecastSpec]) -> "Workflow":
        self.spec = spec
        return self

    def cross_validation(self, df: pd.DataFrame, h: int, n_windows: int, step_size: int, level: Optional[List[int]] = None) -> pd.DataFrame:
        if self.spec is None:
            raise RuntimeError("Model spec not added to workflow.")
        id_col, time_col, target_col = self.id_col, self.time_col, self.target_col
        _ensure_cols(df, [id_col, time_col, target_col])
        _to_datetime(df, time_col)

        freq = (self.spec.freq if hasattr(self.spec, "freq") else "D")
        last_ds = df[time_col].max()
        cutoffs = [last_ds - pd.to_timedelta(step_size * (n_windows - i), unit="D") for i in range(n_windows)]

        cv_rows = []
        for cutoff in cutoffs:
            df_train = df[df[time_col] <= cutoff]
            df_test = df[(df[time_col] > cutoff) & (df[time_col] <= cutoff + pd.to_timedelta(h, unit="D"))]

            df_train_proc = self.recipe_obj.apply(df_train, id_col, time_col, target_col) if self.recipe_obj else df_train

            if self.engine == "statsforecast":
                preds_long = self._cv_statsforecast(df_train_proc, cutoff, h)
            elif self.engine == "mlforecast":
                preds_long = self._cv_mlforecast(df_train_proc, cutoff, h, df_test=df_test)
            elif self.engine == "neuralforecast":
                preds_long = self._cv_neuralforecast(df_train_proc, cutoff, h)
            else:
                raise ValueError(f"Unknown engine: {self.engine}")

            y_true = df_test[[id_col, time_col, target_col]].rename(columns={target_col: "y"})
            out = preds_long.merge(y_true, on=[id_col, time_col], how="left")
            out["cutoff"] = cutoff
            out["workflow"] = self.name
            out["engine"] = self.engine
            cv_rows.append(out)

        cv_df = pd.concat(cv_rows, ignore_index=True)
        if self.recipe_obj:
            cv_df["yhat"] = cv_df["yhat"].astype(float)
            cv_df = self.recipe_obj.invert_predictions(cv_df, target_col="y", yhat_col="yhat")
        return cv_df[[self.id_col, self.time_col, "cutoff", "y", "yhat", "workflow", "engine", "model"]]

    def fit(self, df: pd.DataFrame):
        self._fit_df_ = self.recipe_obj.apply(df, self.id_col, self.time_col, self.target_col) if self.recipe_obj else _safe_copy(df)
        return self

    def forecast(self, df: pd.DataFrame, h: int, level: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Forecast using the fitted model on full data.
        """
        id_col, time_col, target_col = self.id_col, self.time_col, self.target_col

        # ✅ Use preprocessed training data if stored
        if hasattr(self, "_fit_df_") and isinstance(self._fit_df_, pd.DataFrame):
            train_proc = self._fit_df_
        else:
            train_proc = self.recipe_obj.apply(df, id_col, time_col, target_col) if self.recipe_obj else df

        # ✅ Dispatch based on engine
        if self.engine == "statsforecast":
            preds_long = self._forecast_statsforecast(train_proc, h=h, level=level)
        elif self.engine == "mlforecast":
            preds_long = self._forecast_mlforecast(train_proc, h=h)
        elif self.engine == "neuralforecast":
            preds_long = self._forecast_neuralforecast(train_proc, h=h)
        else:
            raise ValueError(f"Unknown engine: {self.engine}")

        # ✅ Add metadata
        preds_long["workflow"] = self.name
        preds_long["engine"] = self.engine
        preds_long["model"] = getattr(self.spec, "model_name", None)

        return preds_long

    def _cv_statsforecast(self, df_train_proc: pd.DataFrame, cutoff: pd.Timestamp, h: int) -> pd.DataFrame:
        id_col, time_col, target_col = self.id_col, self.time_col, self.target_col
        spec: StatsForecastSpec = self.spec  # type: ignore
        out = []
        for m in spec.models:
            sf = SFCore(models=[m], freq=spec.freq, n_jobs=spec.n_jobs)
            train_df = df_train_proc[[id_col, time_col, target_col]].rename(
                columns={id_col: "unique_id", time_col: "ds", target_col: "y"}
            )
            sf.fit(df=train_df)
            preds = sf.forecast(df=train_df, h=h)
            model_name = m.__class__.__name__
            long = preds.rename(columns={"unique_id": id_col, "ds": time_col, model_name: "yhat"})
            long["model"] = model_name
            out.append(long[[id_col, time_col, "yhat", "model"]])
        return pd.concat(out, ignore_index=True)

    def _cv_mlforecast(self, df_train_proc: pd.DataFrame, cutoff: pd.Timestamp, h: int, df_test: pd.DataFrame) -> pd.DataFrame:
        return self._predict_mlforecast(df_train_proc, cutoff, h, X_future=df_test[[self.id_col, self.time_col]].copy())

    def _predict_mlforecast(self, df_train_proc: pd.DataFrame, cutoff: pd.Timestamp, h: int, X_future: pd.DataFrame) -> pd.DataFrame:
        id_col, time_col, target_col = self.id_col, self.time_col, self.target_col
        spec: MLForecastSpec = self.spec  # type: ignore
        mlf = MLFCore(models=spec.models, freq=spec.freq, lags=spec.lags)
        static_feats = spec.static_features if spec.static_features else None

        spec: MLForecastSpec = self.spec  # your existing line

        # detect categorical columns
        cat_cols = [
            c for c in df_train_proc.columns
            if c not in [self.id_col, self.time_col, self.target_col]
            and str(df_train_proc[c].dtype) in ("category", "object")
        ]

        # patch CatBoost models
        for m in spec.models:
            if "catboost" in m.__class__.__name__.lower():
                m.set_params(cat_features=cat_cols)
                print(f"[Workflow] CatBoost cat_features set to: {cat_cols}")

        # convert any object columns to category dtype for LightGBM compatibility
        for c in cat_cols:
            if df_train_proc[c].dtype == "object":
                df_train_proc[c] = df_train_proc[c].astype("category")
                print(f"[Workflow] Converted {c} from object → category for LightGBM compatibility")


        mlf = mlf.fit(
            df_train_proc[[id_col, time_col, target_col] + [c for c in df_train_proc.columns if c not in [id_col, time_col, target_col]]],
            id_col=id_col, time_col=time_col, target_col=target_col, static_features=static_feats
        )
        extra_cols = [c for c in df_train_proc.columns if c not in [id_col, time_col, target_col]]
        Xf = _safe_copy(X_future)
        for c in extra_cols:
            if c not in Xf.columns:
                if static_feats and (c in static_feats):
                    static_map = df_train_proc[[id_col, c]].drop_duplicates(subset=[id_col])
                    Xf = Xf.merge(static_map, on=id_col, how="left")
        preds = mlf.predict(h=h, X_df=None)
        long_list = []
        for col in preds.columns:
            if col in [id_col, time_col]:
                continue
            tmp = preds[[id_col, time_col, col]].rename(columns={col: "yhat"})
            tmp["model"] = col
            long_list.append(tmp)
        return pd.concat(long_list, ignore_index=True)

    def _cv_neuralforecast(self, df_train_proc: pd.DataFrame, cutoff: pd.Timestamp, h: int) -> pd.DataFrame:
        id_col, time_col, target_col = self.id_col, self.time_col, self.target_col
        spec: NeuralForecastSpec = self.spec  # type: ignore

        # Pull parameters from spec, or fall back to defaults
        input_size = getattr(spec, "input_size", 7 * h)
        horizon = getattr(spec, "h", h)
        max_steps  = getattr(spec, "max_steps ", 50)
        batch_size = getattr(spec, "batch_size", 64)
        freq = getattr(spec, "freq", "D")

        # Build model list
        models = [
            NHITS(
                input_size=input_size,
                h=horizon,
                max_steps =max_steps ,
                batch_size=batch_size,
            )
        ]

        # Initialize NeuralForecast
        nf = NFCore(models=models, freq=freq)

        # Prepare data
        train = df_train_proc[[id_col, time_col, target_col]].rename(
            columns={id_col: "unique_id", time_col: "ds", target_col: "y"}
        )

        # Fit and predict
        nf.fit(train)
        preds = nf.predict()

        preds = preds.rename(columns={"unique_id": id_col, "ds": time_col, "NHITS": "yhat"})
        preds["model"] = "NHITS"
        preds["engine"] = "neuralforecast"

        return preds



def workflow(engine: str, id_col: str, time_col: str, target_col: str, name: str) -> Workflow:
    return Workflow(engine=engine, id_col=id_col, time_col=time_col, target_col=target_col, name=name)


# ------------------------------------------------------------
# WorkflowCollection
# ------------------------------------------------------------

class WorkflowCollection:
    def __init__(self, workflows: Optional[List[Workflow]] = None):
        self.workflows: List[Workflow] = workflows or []

    def add(self, wf: Workflow):
        self.workflows.append(wf)

    def cross_validation(self, df: pd.DataFrame, h: int, n_windows: int, step_size: int, level: Optional[List[int]] = None) -> pd.DataFrame:
        parts = [wf.cross_validation(df, h=h, n_windows=n_windows, step_size=step_size, level=level) for wf in self.workflows]
        return pd.concat(parts, ignore_index=True)

    def fit(self, df: pd.DataFrame) -> "WorkflowCollection":
        for wf in self.workflows:
            wf.fit(df)
        return self

    def forecast(self, df: pd.DataFrame, h: int, level: Optional[List[int]] = None) -> pd.DataFrame:
        all_preds = []
        for wf in self.workflows:
            preds = wf.forecast(df, h=h, level=level)
            all_preds.append(preds)
        return pd.concat(all_preds, ignore_index=True)


# ------------------------------------------------------------
# ResultsTable / Ensembles
# ------------------------------------------------------------

class ResultsTable:
    def __init__(self, df_long: pd.DataFrame, id_col: str = "unique_id", time_col: str = "ds"):
        self.id_col = id_col
        self.time_col = time_col
        self.df = _safe_copy(df_long)

    def add(self, new_df: pd.DataFrame) -> "ResultsTable":
        self.df = pd.concat([self.df, new_df], ignore_index=True)
        return self

    def add_ensembles(self, name, members, method="mean", weights=None):
        """
        Add ensemble forecasts by averaging or weighting member yhat values.

        Parameters
        ----------
        name : str
            Name of the ensemble.
        members : list[str] or list[tuple]
            If list of strings → workflows.
            If list of tuples → (workflow, model) pairs.
        method : {'mean', 'median', 'weighted'}
            How to combine predictions.
        weights : list[float], optional
            Required if method='weighted'.
        """
        df = self.df.copy()

        # 1️⃣ Collect member predictions
        parts = []
        for m in members:
            if isinstance(m, tuple):
                wf, model = m
                part = df[(df.workflow == wf) & (df.model == model)].copy()
            else:
                part = df[df.workflow == m].copy()
            part = part[["unique_id", "ds", "cutoff", "y", "yhat"]].rename(columns={"yhat": f"yhat_{m}"})
            parts.append(part)

        # 2️⃣ Merge on unique_id, ds, cutoff
        ens_df = parts[0]
        for p in parts[1:]:
            ens_df = ens_df.merge(p, on=["unique_id", "ds", "cutoff", "y"], how="inner")

        # 3️⃣ Compute ensemble yhat
        yhat_cols = [c for c in ens_df.columns if c.startswith("yhat_")]
        if method == "mean":
            ens_df["yhat"] = ens_df[yhat_cols].mean(axis=1)
        elif method == "median":
            ens_df["yhat"] = ens_df[yhat_cols].median(axis=1)
        elif method == "weighted":
            if weights is None or len(weights) != len(yhat_cols):
                raise ValueError("weights must be same length as members for weighted ensemble")
            ens_df["yhat"] = np.average(ens_df[yhat_cols].values, axis=1, weights=weights)
        else:
            raise ValueError(f"Unknown method: {method}")

        # 4️⃣ Add metadata
        ens_df["workflow"] = name
        ens_df["model"] = method

        # 5️⃣ Append back
        self.df = pd.concat([self.df, ens_df[["unique_id", "ds", "cutoff", "y", "yhat", "workflow", "model"]]], ignore_index=True)
        return self

    def dataframe(self) -> pd.DataFrame:
        return _safe_copy(self.df)


# ------------------------------------------------------------
# Scoreboard
# ------------------------------------------------------------

#class ScoreboardTable:
    # def __init__(self, cv_df: pd.DataFrame, metrics: List[str] = None, id_col: str = "unique_id", time_col: str = "ds"):
    #     self.df = _safe_copy(cv_df)
    #     self.id_col = id_col
    #     self.time_col = time_col
    #     self.metrics = metrics or ["mae", "rmse", "mape", "bias"]

    # def dataframe(self) -> pd.DataFrame:
    #     df = self.df
    #     grp = df.groupby(["workflow", "engine", "model"], as_index=False)
    #     rows = []
    #     for (wf, eng, model), g in grp:
    #         y = g["y"].values.astype(float)
    #         yhat = g["yhat"].values.astype(float)
    #         out = {"workflow": wf, "engine": eng, "model": model}
    #         if "mae" in self.metrics: out["mae"] = _mae(y, yhat)
    #         if "rmse" in self.metrics: out["rmse"] = _rmse(y, yhat)
    #         if "mape" in self.metrics: out["mape"] = _mape(y, yhat)
    #         if "bias" in self.metrics: out["bias"] = _empirical_bias(y, yhat)
    #         rows.append(out)
    #     return pd.DataFrame(rows).sort_values(self.metrics, ascending=True)
def compute_metrics(df: pd.DataFrame, metrics=None, groupby=None):
    """
    Compute forecasting metrics from a long-format results DataFrame.
    Expected columns: y, yhat, and any grouping columns (workflow, model, cutoff, unique_id, etc.)
    """
    metrics = metrics or ["mae", "rmse", "smape", "wape"]
    groupby = groupby or []

    required = {"y", "yhat"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    def _mae(y, yhat):
        return np.mean(np.abs(y - yhat))

    def _rmse(y, yhat):
        return np.sqrt(np.mean((y - yhat) ** 2))

    def _smape(y, yhat):
        denom = np.abs(y) + np.abs(yhat)
        return np.mean(200 * np.abs(y - yhat) / np.where(denom == 0, 1, denom))

    def _wape(y, yhat):
        return np.sum(np.abs(y - yhat)) / np.sum(np.abs(y)) * 100

    metric_fns = {"mae": _mae, "rmse": _rmse, "smape": _smape, "wape": _wape}

    results = []
    for keys, group in df.groupby(groupby):
        y, yhat = group["y"].values, group["yhat"].values
        row = dict(zip(groupby, keys if isinstance(keys, tuple) else (keys,)))
        for m in metrics:
            fn = metric_fns.get(m)
            if fn:
                row[m.upper()] = fn(y, yhat)
        results.append(row)

    return pd.DataFrame(results)

class Scoreboard:
    def __init__(self, df: pd.DataFrame):
        """Initialize a Scoreboard from CV results."""
        self.df = df.copy()
        self.aggregate = None
        self.by_fold = None
        self.by_id = None
        self.by_fold_id = None

    def compute(self, metrics=None):
        metrics = metrics or ["mae", "rmse", "smape", "wape"]
        df = self.df

        # 1️⃣ aggregate (all folds + all IDs)
        self.aggregate = compute_metrics(df, metrics=metrics,
                                         groupby=["workflow", "model"])

        # 2️⃣ by_fold (aggregate across IDs, per fold)
        self.by_fold = compute_metrics(df, metrics=metrics,
                                       groupby=["workflow", "model", "cutoff"])

        # 3️⃣ by_id (aggregate across folds, per SKU)
        self.by_id = compute_metrics(df, metrics=metrics,
                                     groupby=["workflow", "model", "unique_id"])

        # 4️⃣ by_fold_id (full CV view, lowest level)
        self.by_fold_id = compute_metrics(df, metrics=metrics,
                                          groupby=["workflow", "model", "unique_id", "cutoff"])

        return self

    def show(self, mode: str | list[str] = "aggregate", include_source: bool = True) -> pd.DataFrame:
        """Display one or more scoreboard views."""
        valid_modes = {
            "aggregate": self.aggregate,
            "by_fold": self.by_fold,
            "by_id": self.by_id,
            "by_fold_id": self.by_fold_id,
        }

        if isinstance(mode, str):
            mode = [mode]

        invalid = [m for m in mode if m not in valid_modes]
        if invalid:
            raise ValueError(f"Invalid mode(s): {invalid}. Must be one of {list(valid_modes)}")

        dfs = []
        for m in mode:
            df = valid_modes[m]
            if df is not None:
                tmp = df.copy()
                if include_source:
                    tmp["_source"] = m
                dfs.append(tmp)

        if not dfs:
            raise ValueError("No available DataFrames for requested mode(s).")

        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    def dataframe(self, mode: str = "all") -> pd.DataFrame:
        """Return raw DataFrames for advanced analysis or export."""
        mapping = {
            "aggregate": self.aggregate,
            "by_fold": self.by_fold,
            "by_id": self.by_id,
            "by_fold_id": self.by_fold_id,
        }

        if mode == "all":
            dfs = []
            for name, df in mapping.items():
                if df is not None:
                    tmp = df.copy()
                    tmp["_mode"] = name
                    dfs.append(tmp)
            return pd.concat(dfs, ignore_index=True)
        elif mode in mapping:
            return mapping[mode]
        else:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of {list(mapping)} or 'all'.")
# ------------------------------------------------------------
# Model Selection
# ------------------------------------------------------------

class ModelSelection:
    def __init__(self, cv_df: pd.DataFrame, metric: str = "mae", group_by: str = "unique_id", top_n: int = 1):
        self.cv = _safe_copy(cv_df)
        self.metric = metric
        self.group_by = group_by
        self.top_n = top_n
        agg_rows = []
        for (uid, wf, eng, model), g in self.cv.groupby([group_by, "workflow", "engine", "model"]):
            y = g["y"].values.astype(float)
            yhat = g["yhat"].values.astype(float)
            agg_rows.append({
                group_by: uid,
                "workflow": wf,
                "engine": eng,
                "model": model,
                "mae": _mae(y, yhat),
                "rmse": _rmse(y, yhat),
                "mape": _mape(y, yhat),
                "bias": _empirical_bias(y, yhat),
            })
        self.tbl = pd.DataFrame(agg_rows)

    def select(self) -> pd.DataFrame:
        metric = self.metric.lower()
        if metric not in {"mae", "rmse", "mape", "bias"}:
            raise ValueError("metric must be one of ['mae','rmse','mape','bias']")
        df = self.tbl.copy()
        key = "bias" if metric != "bias" else None
        if metric == "bias":
            df["_sel"] = df["bias"].abs(); key = "_sel"
        sel = df.sort_values([self.group_by, key], ascending=[True, True]).groupby(self.group_by).head(self.top_n).drop(columns=["_sel"], errors="ignore")
        return sel

    def summary(self) -> pd.DataFrame:
        return _safe_copy(self.tbl)


def recipe(name: str = "") -> "Recipe":
    """Convenience alias for Recipe(name)."""
    return Recipe(name=name)
