# """
# manager.py - Enhanced WorkflowManager with Lineage Tracking
# ============================================================
# """

import pandas as pd
import numpy as np
from typing import Optional, List, Callable, Dict, Any
import time
import warnings
from joblib import Parallel, delayed

from statsforecast import StatsForecast
from mlforecast import MLForecast
from neuralforecast import NeuralForecast

from tsforge.evaluation import score_all

# ============================================
# WORKFLOW MANAGER
# ============================================

class WorkflowManager:
    """
    Manager for running multiple forecasting workflows.
    
    Clean separation between business logic and logging via callbacks.
    """
    
    def __init__(self, workflows, id_col="unique_id", time_col="date", 
                 target_col="sales", callbacks: Optional[Dict[str, Callable]] = None):
        """
        Initialize WorkflowManager.
        
        Parameters
        ----------
        workflows : list
            List of workflow dictionaries
        id_col : str
            Column name for series identifiers
        time_col : str
            Column name for time/date
        target_col : str
            Column name for target variable
        callbacks : dict, optional
            Dictionary of callback functions for logging:
            - 'on_init': Called at initialization with (workflows,)
            - 'on_cv_start': Called at CV start with (h, n_windows, step_size, num_workflows, level)
            - 'on_cv_complete': Called at CV end with (num_preds, num_models, num_workflows)
            - 'on_forecast_start': Called at forecast start with (h, num_workflows, level)
            - 'on_forecast_complete': Called at forecast end with (num_preds, num_models)
            - 'on_workflow_start': Called when workflow starts with (name, num_series, num_rows)
            - 'on_workflow_complete': Called when workflow ends with (name,)
            - 'on_training_start': Called when training starts with (name, engine, num_models)
            - 'on_training_complete': Called when training ends with (name,)
            - 'on_ensemble_start': Called when ensemble starts with (name, method, num_members)
            - 'on_ensemble_member': Called for each ensemble member with (member_name, idx, total)
            - 'on_ensemble_complete': Called when ensemble ends
            - 'on_recipe_apply': Called when recipe is applied with (workflow_name, recipe_name)
            - 'on_recipe_complete': Called when recipe completes with (workflow_name,)
            - 'on_warning': Called on warnings with (name, message)
        
        Examples
        --------
        >>> # Without logging
        >>> manager = WorkflowManager(workflows)
        >>> 
        >>> # With logging
        >>> from tsforge.logging.logger import WorkflowLogger, get_manager_callbacks
        >>> logger = WorkflowLogger(verbose=2)
        >>> manager = WorkflowManager(workflows, callbacks=get_manager_callbacks(logger))
        """
        self.workflows = workflows
        self.id_col, self.time_col, self.target_col = id_col, time_col, target_col
        self.last_cv_params = None
        self.last_cv_preds = None
        self._pred_cache: dict = {}
        self.callbacks = callbacks or {}
        
        # Callback: on_init
        self._fire('on_init', workflows)

    def _fire(self, event: str, *args, **kwargs):
        """Fire a callback event if it exists."""
        if event in self.callbacks:
            self.callbacks[event](*args, **kwargs)

    # --------------------------
    # Helpers
    # --------------------------
    def _ensure_types(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out[self.time_col] = pd.to_datetime(out[self.time_col], errors="coerce")
        out[self.target_col] = pd.to_numeric(out[self.target_col], errors="coerce")
        return out.dropna(subset=[self.id_col, self.time_col, self.target_col])

    def _core_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[[self.id_col, self.time_col, self.target_col]].copy()

    def _to_sf(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={self.time_col: "ds", self.target_col: "y"})

    def _to_nf(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.rename(columns={self.time_col: "ds", self.target_col: "y"})

    def _reset_index_if_needed(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.index, pd.MultiIndex) or df.index.name in [self.id_col, self.time_col, "cutoff"]:
            return df.reset_index()
        return df

    def _standardize_preds(self, preds: pd.DataFrame, name: str, engine: str) -> pd.DataFrame:
        """
        Standardize Nixtla outputs to tsforge format:
        - cross_validation: keep truth ('y' → target_col)
        - forecast: no truth column
        - one row per model, with yhat + optional lo-XX/hi-XX columns
        """
        df = self._reset_index_if_needed(preds.copy())

        # 1) Rename Nixtla-style columns
        if "ds" in df.columns:
            df = df.rename(columns={"ds": self.time_col})
        if "y" in df.columns:
            df = df.rename(columns={"y": self.target_col})
        if "y_hat" in df.columns:
            df = df.rename(columns={"y_hat": "yhat"})
        if "mean" in df.columns:
            df = df.rename(columns={"mean": "yhat"})

        keys = [c for c in [self.id_col, self.time_col, "cutoff"] if c in df.columns]
        reserved = set(keys + [self.target_col, "yhat", "workflow", "engine", "model"])

        def _is_interval_col(c: str) -> bool:
            c = str(c)
            return ("-lo-" in c) or ("-hi-" in c) or c.startswith("lo-") or c.startswith("hi-")

        point_cols = [c for c in df.columns if c not in reserved and not _is_interval_col(c)]

        rows = []
        if len(point_cols) == 0:
            base = df[keys].copy()
            if self.target_col in df.columns:
                base[self.target_col] = df[self.target_col]
            if "yhat" in df.columns:
                base["yhat"] = df["yhat"]
            for c in df.columns:
                if c.startswith("lo-") or c.startswith("hi-"):
                    base[c] = df[c]
            base["model"] = name
            rows.append(base)
        else:
            for m in point_cols:
                out = df[keys].copy()
                if self.target_col in df.columns:
                    out[self.target_col] = df[self.target_col]
                out["yhat"] = df[m]

                m_lo = [c for c in df.columns if f"{m}-lo-" in str(c)]
                m_hi = [c for c in df.columns if f"{m}-hi-" in str(c)]
                for col in m_lo:
                    lvl = str(col).split("-")[-1]
                    out[f"lo-{lvl}"] = df[col]
                for col in m_hi:
                    lvl = str(col).split("-")[-1]
                    out[f"hi-{lvl}"] = df[col]

                out["model"] = m
                rows.append(out)

        std = pd.concat(rows, ignore_index=True)
        std["workflow"] = name
        std["engine"] = engine

        if engine != "ensemble" and "cutoff" in std.columns and self.target_col not in std.columns:
            raise ValueError(
                f"Cross-validation results missing target column '{self.target_col}'. "
                f"Incoming columns: {preds.columns.tolist()}"
            )

        return std

    def _fingerprint(self, X: pd.DataFrame, h: int, mode: str,
                     n_windows: int | None, step_size: int | None) -> tuple:
        """Lightweight fingerprint of data+settings for caching."""
        return (
            X[self.time_col].min(),
            X[self.time_col].max(),
            int(X[self.id_col].nunique()),
            int(len(X)),
            mode, int(h), int(n_windows or 0), int(step_size or 0),
        )

    def _get_member_preds(self, member, X, h, mode, n_windows=None, step_size=None, level=None):
        """
        Return STANDARDIZED predictions (so ensemble can rely on yhat + intervals).
        - Workflow-level: whole workflow standardized
        - Model-level: filter on standardized 'model' column
        """
        if isinstance(member, dict):
            wf_dict, filter_model = member, None
        else:
            wf_dict, filter_model = member

        wf_name = wf_dict["name"]
        wf_engine = wf_dict["engine"]

        member_key = f"std:{wf_name}"
        fp = (member_key, self._fingerprint(X, h, mode, n_windows, step_size))

        if fp in self._pred_cache:
            std = self._pred_cache[fp].copy()
        else:
            raw = self._run_workflow_raw(wf_dict, X, h, mode, n_windows, step_size, level)
            std = self._standardize_preds(raw, name=wf_name, engine=wf_engine)
            self._pred_cache[fp] = std.copy()

        if filter_model:
            base = str(filter_model).strip()
            paren_idx = base.find("(")
            if paren_idx != -1:
                base = base[:paren_idx].strip()
            if base.endswith("()"):
                base = base[:-2]

            mnames = std["model"].astype(str)
            exact_mask = (mnames == base)
            if exact_mask.any():
                std = std.loc[exact_mask].copy()
            else:
                starts_mask = mnames.str.startswith(base)
                if starts_mask.sum() == 1:
                    std = std.loc[starts_mask].copy()
                elif starts_mask.sum() > 1:
                    found = sorted(std.loc[starts_mask, "model"].unique().tolist())
                    raise ValueError(
                        f"Ambiguous model '{filter_model}'. Candidates: {found}. "
                        f"Please use an explicit alias (e.g., '{found[0]}')."
                    )
                else:
                    found = sorted(std["model"].unique().tolist())
                    raise ValueError(
                        f"Model '{filter_model}' not found. Available: {found}. "
                        f"Tip: use alias names from make_sf_workflow (e.g., 'SeasonalNaive' or 'SeasonalNaive_1')."
                    )

        return std

    def _run_workflow_raw(self, wf, X, h, mode, n_windows=None, step_size=None, level=None):
        """Execute single workflow - core business logic."""
        name, engine, params = wf["name"], wf["engine"], wf["params"]
        recipe, static_feats = wf.get("recipe"), wf.get("static_features")
        
        # Apply recipe if present
        if recipe:
            self._fire('on_recipe_apply', name, recipe.name)
            X = recipe.bake(X)
            self._fire('on_recipe_complete', name)
        
        X = self._ensure_types(X)

        if engine == "mlforecast":
            mlf = MLForecast(**params)
            pi = wf.get("prediction_intervals")
            try:
                if mode == "forecast":
                    mlf.fit(X, id_col=self.id_col, time_col=self.time_col,
                            target_col=self.target_col, static_features=static_feats)
                    return mlf.predict(h=h, level=level, prediction_intervals=pi)
                else:  # cv
                    return mlf.cross_validation(
                        X, id_col=self.id_col, time_col=self.time_col,
                        target_col=self.target_col,
                        h=h, n_windows=n_windows, step_size=step_size,
                        static_features=static_feats,
                        level=level,
                        prediction_intervals=pi
                    )
            except ValueError as e:
                self._fire('on_warning', name, f"Prediction intervals failed: {str(e)[:60]}")
                if mode == "forecast":
                    mlf.fit(X, id_col=self.id_col, time_col=self.time_col,
                            target_col=self.target_col, static_features=static_feats)
                    preds = mlf.predict(h=h)
                else:
                    preds = mlf.cross_validation(
                        X, id_col=self.id_col, time_col=self.time_col,
                        target_col=self.target_col,
                        h=h, n_windows=n_windows, step_size=step_size,
                        static_features=static_feats
                    )
                preds = self._safe_conformal(preds, level)
                return preds

        elif engine == "statsforecast":
            sf = StatsForecast(**params)
            sf_df = self._to_sf(self._core_cols(X))
            sf.fit(sf_df[[self.id_col, "ds", "y"]])
            if mode == "forecast":
                return sf.forecast(h=h, level=level).reset_index()
            else:
                return sf.cross_validation(df=sf_df,
                                           h=h,
                                           n_windows=n_windows,
                                           step_size=step_size,
                                           level=level)

        elif engine == "neuralforecast":
            nf_df = self._to_nf(self._core_cols(X))
            nf = NeuralForecast(models=params["models"], freq=params["freq"])
            nf.fit(nf_df)
            if mode == "forecast":
                return nf.predict(h=h, level=level).reset_index()
            else:
                return nf.cross_validation(df=nf_df,
                                           n_windows=n_windows,
                                           step_size=step_size,
                                           level=level,
                                           verbose=0)

        else:
            raise ValueError(f"Unknown engine {engine}")

    @staticmethod
    def _safe_conformal(preds: pd.DataFrame, level: Optional[List[int]]) -> pd.DataFrame:
        """
        Ensure interval columns exist.
        Works both on standardized frames (with 'yhat') and raw Nixtla wide frames (model columns).
        If no intervals exist, creates dummy zero-width ones.
        """
        if level is None:
            return preds

        base_cols = {"unique_id", "ds", "date", "y", "sales", "cutoff"}
        has_yhat = "yhat" in preds.columns

        if has_yhat:
            for lv in level:
                lo, hi = f"lo-{lv}", f"hi-{lv}"
                if lo not in preds.columns:
                    preds[lo] = preds["yhat"]
                if hi not in preds.columns:
                    preds[hi] = preds["yhat"]
        else:
            model_cols = [c for c in preds.columns if c not in base_cols and not str(c).startswith(("lo-", "hi-"))]
            for m in model_cols:
                for lv in level:
                    lo, hi = f"{m}-lo-{lv}", f"{m}-hi-{lv}"
                    if lo not in preds.columns:
                        preds[lo] = preds[m]
                    if hi not in preds.columns:
                        preds[hi] = preds[m]

        return preds

    # --------------------------
    # Public Methods
    # --------------------------

    def forecast(self, df: pd.DataFrame, h: int, level: Optional[List[int]] = None) -> pd.DataFrame:
        """Run forecast across all workflows."""
        # Callback: on_forecast_start
        self._fire('on_forecast_start', h, len(self.workflows), level)
        
        def _run_one(wf):
            name, engine, params = wf["name"], wf["engine"], wf["params"]
            
            # Callback: on_workflow_start
            self._fire('on_workflow_start', name, df[self.id_col].nunique(), len(df))
            
            recipe, static_feats = wf.get("recipe"), wf.get("static_features")
            X = self._ensure_types(recipe.bake(df) if recipe else df)

            # Callback: on_training_start
            self._fire('on_training_start', name, engine, len(params.get('models', [])))

            if engine == "mlforecast":
                mlf = MLForecast(**params)
                pi = wf.get("prediction_intervals")
                mlf.fit(X, id_col=self.id_col, time_col=self.time_col,
                        target_col=self.target_col, static_features=static_feats)
                try:
                    preds = mlf.predict(h=h, level=level, prediction_intervals=pi)
                except ValueError as e:
                    self._fire('on_warning', name, f"Conformal intervals failed: {str(e)[:60]}")
                    preds = mlf.predict(h=h)
                    preds = self._safe_conformal(preds, level)

            elif engine == "statsforecast":
                sf = StatsForecast(**params)
                sf_df = self._to_sf(self._core_cols(X))
                sf.fit(sf_df[[self.id_col, "ds", "y"]])
                preds = sf.forecast(h=h, level=level).reset_index()

            elif engine == "neuralforecast":
                nf_df = self._to_nf(self._core_cols(X))
                nf = NeuralForecast(models=params["models"], freq=params["freq"])
                nf.fit(nf_df)
                preds = nf.predict(h=h, level=level).reset_index()

            elif engine == "ensemble":
                result = self._run_ensemble(X, h=h, mode="forecast",
                                          params=params, name=name)
                self._fire('on_training_complete', name)
                self._fire('on_workflow_complete', name)
                return result

            elif engine == "custom":
                preds = params["func"](X, h, self.id_col, self.time_col, self.target_col)

            else:
                raise ValueError(f"Unknown engine {engine}")

            self._fire('on_training_complete', name)
            result = self._standardize_preds(preds, name, engine)
            self._fire('on_workflow_complete', name)
            return result

        if len(self.workflows) == 1:
            rows = [_run_one(self.workflows[0])]
        else:
            rows = Parallel(n_jobs=-1, backend="threading")(
                delayed(_run_one)(wf) for wf in self.workflows
            )

        result_df = pd.concat(rows, ignore_index=True)
        
        # Callback: on_forecast_complete
        self._fire('on_forecast_complete', len(result_df), result_df['model'].nunique())
        
        return result_df

    def cross_validation(self,
                         df: pd.DataFrame,
                         h: int,
                         n_windows: int = 1,
                         step_size: Optional[int] = None,
                         level: Optional[List[int]] = None) -> pd.DataFrame:
        """Run rolling-origin cross-validation across all workflows."""
        step = step_size or h
        self.last_cv_params = dict(h=h, n_windows=n_windows, step_size=step, level=level)
        self.last_cv_preds = None

        # Callback: on_cv_start
        self._fire('on_cv_start', h, n_windows, step, len(self.workflows), level)

        def _run_one(wf):
            name, engine, params = wf["name"], wf["engine"], wf["params"]
            
            # Callback: on_workflow_start
            self._fire('on_workflow_start', name, df[self.id_col].nunique(), len(df))
            
            recipe, static_feats = wf.get("recipe"), wf.get("static_features")
            X = self._ensure_types(recipe.bake(df) if recipe else df)

            # Callback: on_training_start
            self._fire('on_training_start', name, engine, len(params.get('models', [])))

            if engine == "mlforecast":
                mlf = MLForecast(**params)
                pi = wf.get("prediction_intervals")
                try:
                    preds = mlf.cross_validation(
                        X, id_col=self.id_col, time_col=self.time_col,
                        target_col=self.target_col,
                        h=h, n_windows=n_windows, step_size=step,
                        static_features=static_feats,
                        level=level,
                        prediction_intervals=pi
                    )
                except ValueError as e:
                    self._fire('on_warning', name, f"Conformal intervals failed: {str(e)[:60]}")
                    preds = mlf.cross_validation(
                        X, id_col=self.id_col, time_col=self.time_col,
                        target_col=self.target_col,
                        h=h, n_windows=n_windows, step_size=step,
                        static_features=static_feats
                    )
                    preds = self._safe_conformal(preds, level)

            elif engine == "statsforecast":
                sf = StatsForecast(**params)
                sf_df = self._to_sf(self._core_cols(X))
                preds = sf.cross_validation(df=sf_df, h=h, n_windows=n_windows,
                                            step_size=step, level=level)

            elif engine == "neuralforecast":
                nf_df = self._to_nf(self._core_cols(X))
                nf = NeuralForecast(models=params["models"], freq=params["freq"])
                preds = nf.cross_validation(df=nf_df, n_windows=n_windows,
                                            step_size=step, level=level,
                                            verbose=0)

            elif engine == "ensemble":
                result = self._run_ensemble(X, h=h, mode="cv",
                                          params=params, name=name,
                                          n_windows=n_windows, step_size=step)
                self._fire('on_training_complete', name)
                self._fire('on_workflow_complete', name)
                return result

            elif engine == "custom":
                preds = params["func"](X, h, self.id_col, self.time_col, self.target_col)

            else:
                raise ValueError(f"Unknown engine {engine}")

            self._fire('on_training_complete', name)
            result = self._standardize_preds(preds, name, engine)
            self._fire('on_workflow_complete', name)
            return result

        if len(self.workflows) == 1:
            rows = [_run_one(self.workflows[0])]
        else:
            rows = Parallel(n_jobs=-1, backend="threading")(
                delayed(_run_one)(wf) for wf in self.workflows
            )

        preds_df = pd.concat(rows, ignore_index=True)
        self.last_cv_preds = preds_df
        
        # Callback: on_cv_complete
        self._fire('on_cv_complete', len(preds_df), preds_df['model'].nunique(), len(self.workflows))
        
        return preds_df

    # --------------------------
    # Ensemble logic
    # --------------------------
    def _run_ensemble(self, X, h, mode, params, name,
                    n_windows=None, step_size=None):
        """
        Ensemble for standardized frames:
        - input members provide standardized frames with yhat + optional lo-XX/hi-XX + sales
        - we tag interval columns to avoid collisions, then aggregate back to lo-XX/hi-XX
        """
        method = params.get("method", "mean")
        members = params["members"]
        
        # Callback: on_ensemble_start
        self._fire('on_ensemble_start', name, method, len(members))
        
        keys = [self.id_col, self.time_col]
        if mode == "cv":
            keys.append("cutoff")

        member_frames = []
        for idx, member in enumerate(members, 1):
            # Callback: on_ensemble_member
            member_name = member["name"] if isinstance(member, dict) else member[0]["name"]
            self._fire('on_ensemble_member', member_name, idx, len(members))
            
            preds_m = self._get_member_preds(member, X, h, mode,
                                            n_windows=n_windows,
                                            step_size=step_size)

            if isinstance(member, dict):
                tag = member["name"]
            else:
                tag = f"{member[0]['name']}|{member[1]}"

            rename_map = {}
            for c in preds_m.columns:
                if c in ("yhat",) or c.startswith("lo-") or c.startswith("hi-"):
                    rename_map[c] = f"{tag}:{c}"
            preds_m = preds_m.rename(columns=rename_map)

            member_frames.append(preds_m)

        from functools import reduce
        merged = reduce(lambda l, r: pd.merge(l, r, on=keys, how="inner"), member_frames)

        yhat_cols = [c for c in merged.columns if c.endswith(":yhat")]
        if not yhat_cols:
            raise ValueError("No member forecasts found (yhat missing).")

        M_yhat = merged[yhat_cols].to_numpy(dtype=float)

        weights = params.get("weights")

        def combine(M):
            if method == "mean":
                return np.nanmean(M, axis=1)
            elif method == "median":
                return np.nanmedian(M, axis=1)
            elif method == "weighted":
                if weights is None:
                    raise ValueError("Weights must be provided for weighted ensemble")
                w = np.array(weights, dtype=float)
                if len(w) != M.shape[1]:
                    raise ValueError(f"weights length {len(w)} != num members {M.shape[1]}")
                w = w / w.sum()
                return np.nansum(M * w, axis=1)
            else:
                raise ValueError("method must be one of {'mean','median','weighted'}")

        out = merged[keys].copy()

        if mode == "cv":
            if self.target_col in merged.columns:
                out[self.target_col] = merged[self.target_col].bfill(axis=1).iloc[:, 0]

        out["yhat"] = combine(M_yhat)

        levels = sorted({c.split("-")[-1] for c in merged.columns if ":lo-" in c or ":hi-" in c})
        for lvl in levels:
            lo_cols = [c for c in merged.columns if c.endswith(f":lo-{lvl}")]
            hi_cols = [c for c in merged.columns if c.endswith(f":hi-{lvl}")]
            if lo_cols and hi_cols:
                out[f"lo-{lvl}"] = combine(merged[lo_cols].to_numpy(dtype=float))
                out[f"hi-{lvl}"] = combine(merged[hi_cols].to_numpy(dtype=float))
            else:
                out[f"lo-{lvl}"] = out["yhat"]
                out[f"hi-{lvl}"] = out["yhat"]

        out["workflow"] = name
        out["engine"] = "ensemble"
        out["model"] = f"Ensemble-{method}"
        
        # Callback: on_ensemble_complete
        self._fire('on_ensemble_complete')
        
        return out
    
    # --------------------------
    # Alias Methods 
    # --------------------------
    def predict(self, df: pd.DataFrame, h: int, level: Optional[List[int]] = None) -> pd.DataFrame:
        """
        
        MLForecast and NeuralForecast use .predict(), while StatsForecast uses .forecast().
        This method provides a consistent interface regardless of which you prefer.
        
        Parameters
        ----------
        df : pd.DataFrame
            Training data
        h : int
            Forecast horizon
        level : Optional[List[int]]
            Prediction interval levels (e.g., [80, 95])
        
        Returns
        -------
        pd.DataFrame
            Predictions with columns: unique_id, date, yhat, workflow, engine, model
        
        Examples
        --------
        >>> manager = WorkflowManager(workflows)
        >>> predictions = manager.predict(train_df, h=28)
        >>> # Equivalent to:
        >>> predictions = manager.forecast(train_df, h=28)
        """
        return self.forecast(df, h, level)