"""
workflow.py
"""

import importlib
import warnings
from typing import Any, Dict, List, Optional, Union


# ============================================
# NIXTLA MODEL REGISTRY
# ============================================

class NixtlaModels:
    """
    Registry and helper for all Nixtla models.
    
    Examples
    --------
    >>> # Access by engine
    >>> NixtlaModels.mlforecast.lgbm(n_estimators=100)
    >>> NixtlaModels.statsforecast.naive()
    >>> NixtlaModels.neuralforecast.nbeats(input_size=28, h=7)
    >>> 
    >>> # Or by string
    >>> NixtlaModels.get("lgbm", n_estimators=100)
    """
    
    REGISTRY = {
        # MLForecast models
        "lgbm": ("lightgbm", "LGBMRegressor", {"verbose": -1, "random_state": 42}),
        "xgboost": ("xgboost", "XGBRegressor", {"verbosity": 0, "random_state": 42}),
        "catboost": ("catboost", "CatBoostRegressor", {"verbose": False, "random_state": 42}),
        "rf": ("sklearn.ensemble", "RandomForestRegressor", {"random_state": 42}),
        "linear": ("sklearn.linear_model", "LinearRegression", {}),
        "ridge": ("sklearn.linear_model", "Ridge", {"random_state": 42}),
        
        # StatsForecast models
        "naive": ("statsforecast.models", "Naive", {}),
        "seasonal_naive": ("statsforecast.models", "SeasonalNaive", {}),
        "auto_arima": ("statsforecast.models", "AutoARIMA", {}),
        "auto_ets": ("statsforecast.models", "AutoETS", {}),
        "auto_theta": ("statsforecast.models", "AutoTheta", {}),
        
        # NeuralForecast models
        "nbeats": ("neuralforecast.models", "NBEATS", {}),
        "nhits": ("neuralforecast.models", "NHITS", {}),
        "mlp": ("neuralforecast.models", "MLP", {}),
    }
    
    @classmethod
    def get(cls, model_name: str, **kwargs) -> Any:
        """Get any Nixtla model by name."""
        if model_name not in cls.REGISTRY:
            raise ValueError(f"Model '{model_name}' not found. Use NixtlaModels.list_all()")
        
        module_path, class_name, defaults = cls.REGISTRY[model_name]
        
        try:
            module = importlib.import_module(module_path)
            model_class = getattr(module, class_name)
            params = {**defaults, **kwargs}
            return model_class(**params)
        except ImportError as e:
            package = module_path.split('.')[0]
            raise ImportError(f"Install: pip install {package}") from e
    
    @classmethod
    def list_all(cls) -> Dict[str, List[str]]:
        """List all available models by engine."""
        by_engine = {"mlforecast": [], "statsforecast": [], "neuralforecast": []}
        for name, (module, _, _) in cls.REGISTRY.items():
            if "statsforecast" in module:
                by_engine["statsforecast"].append(name)
            elif "neuralforecast" in module:
                by_engine["neuralforecast"].append(name)
            else:
                by_engine["mlforecast"].append(name)
        return by_engine
    
    # Convenience accessors
    class mlforecast:
        """MLForecast models."""
        @staticmethod
        def lgbm(**kwargs): return NixtlaModels.get("lgbm", **kwargs)
        @staticmethod
        def xgboost(**kwargs): return NixtlaModels.get("xgboost", **kwargs)
        @staticmethod
        def catboost(**kwargs): return NixtlaModels.get("catboost", **kwargs)
        @staticmethod
        def rf(**kwargs): return NixtlaModels.get("rf", **kwargs)
        @staticmethod
        def linear(**kwargs): return NixtlaModels.get("linear", **kwargs)
    
    class statsforecast:
        """StatsForecast models."""
        @staticmethod
        def naive(**kwargs): return NixtlaModels.get("naive", **kwargs)
        @staticmethod
        def seasonal_naive(**kwargs): return NixtlaModels.get("seasonal_naive", **kwargs)
        @staticmethod
        def auto_arima(**kwargs): return NixtlaModels.get("auto_arima", **kwargs)
        @staticmethod
        def auto_ets(**kwargs): return NixtlaModels.get("auto_ets", **kwargs)
        @staticmethod
        def auto_theta(**kwargs): return NixtlaModels.get("auto_theta", **kwargs)
    
    class neuralforecast:
        """NeuralForecast models."""
        @staticmethod
        def nbeats(**kwargs): return NixtlaModels.get("nbeats", **kwargs)
        @staticmethod
        def nhits(**kwargs): return NixtlaModels.get("nhits", **kwargs)
        @staticmethod
        def mlp(**kwargs): return NixtlaModels.get("mlp", **kwargs)


# ============================================
# WORKFLOW CLASS (EXPLICIT ENGINE)
# ============================================

class Workflow:
    """
    Unified workflow builder with EXPLICIT Nixtla engine specification.
    
    Recommended Usage (Explicit Engine):
    ------------------------------------
    >>> # MLForecast (explicit)
    >>> wf = (Workflow.mlforecast("LGBM")
    ...     .add_model("lgbm", n_estimators=100)
    ...     .add_model("xgboost", n_estimators=100)
    ...     .with_lags([7, 14, 21])
    ...     .build())
    
    >>> # StatsForecast (explicit)
    >>> wf = (Workflow.statsforecast("Stats")
    ...     .add_model("naive")
    ...     .add_model("seasonal_naive", season_length=7)
    ...     .add_model("auto_arima")
    ...     .build())
    
    >>> # NeuralForecast (explicit)
    >>> wf = (Workflow.neuralforecast("Neural")
    ...     .add_model("nbeats", input_size=28, h=7)
    ...     .add_model("nhits", input_size=28, h=7)
    ...     .build())
    
    Backward Compatible (Auto-detect):
    ----------------------------------
    >>> # Still works (auto-detects engine from models)
    >>> wf = (Workflow("LGBM")
    ...     .add_model("lgbm")
    ...     .with_lags([7, 14])
    ...     .build())
    """
    
    def __init__(self, name: str, engine: Optional[str] = None):
        self.name = name
        self._engine = engine  # Explicit engine (if specified)
        self._models = []
        self._recipe = None
        self._lags = None
        self._lag_transforms = None  # ✅ Add this
        self._date_features = None
        self._static_features = None
        self._freq = "D"
        self._prediction_intervals = None
        self._ensemble_members = []
        self._ensemble_method = None
        self._ensemble_weights = None
        self._custom_func = None
        self._config = {}
    
    # ============================================
    # CLASS METHODS (EXPLICIT ENGINE)
    # ============================================
    
    @classmethod
    def mlforecast(cls, name: str) -> 'Workflow':
        """
        Create a workflow that uses Nixtla's MLForecast engine.
        
        MLForecast supports ML models: LightGBM, XGBoost, CatBoost, sklearn, etc.
        
        Parameters
        ----------
        name : str
            Workflow name
        
        Returns
        -------
        Workflow
            Workflow instance with MLForecast engine
        
        Examples
        --------
        >>> wf = (Workflow.mlforecast("LGBM")
        ...     .add_model("lgbm", n_estimators=100)
        ...     .add_model("xgboost", n_estimators=100)
        ...     .with_lags([7, 14, 21])
        ...     .with_date_features(["dayofweek", "month"])
        ...     .build())
        """
        return cls(name, engine="mlforecast")
    
    @classmethod
    def statsforecast(cls, name: str) -> 'Workflow':
        """
        Create a workflow that uses Nixtla's StatsForecast engine.
        
        StatsForecast supports statistical models: ARIMA, ETS, Theta, Naive, etc.
        
        Parameters
        ----------
        name : str
            Workflow name
        
        Returns
        -------
        Workflow
            Workflow instance with StatsForecast engine
        
        Examples
        --------
        >>> wf = (Workflow.statsforecast("Baselines")
        ...     .add_model("naive")
        ...     .add_model("seasonal_naive", season_length=7)
        ...     .add_model("auto_arima", season_length=7)
        ...     .build())
        """
        return cls(name, engine="statsforecast")
    
    @classmethod
    def neuralforecast(cls, name: str) -> 'Workflow':
        """
        Create a workflow that uses Nixtla's NeuralForecast engine.
        
        NeuralForecast supports deep learning models: NBEATS, NHITS, MLP, RNN, etc.
        
        Parameters
        ----------
        name : str
            Workflow name
        
        Returns
        -------
        Workflow
            Workflow instance with NeuralForecast engine
        
        Examples
        --------
        >>> wf = (Workflow.neuralforecast("DeepLearning")
        ...     .add_model("nbeats", input_size=28, h=7)
        ...     .add_model("nhits", input_size=28, h=7)
        ...     .build())
        """
        return cls(name, engine="neuralforecast")
    
    @classmethod
    def ensemble(cls, name: str) -> 'Workflow':
        """
        Create an ensemble workflow (TSForge feature).
        
        Combines predictions from multiple workflows.
        
        Parameters
        ----------
        name : str
            Ensemble name
        
        Returns
        -------
        Workflow
            Workflow instance for ensemble
        
        Examples
        --------
        >>> wf1 = Workflow.statsforecast("Stats").add_model("naive").build()
        >>> wf2 = Workflow.mlforecast("ML").add_model("lgbm").with_lags([7]).build()
        >>> 
        >>> ensemble = (Workflow.ensemble("Combined")
        ...     .add_member(wf1)
        ...     .add_member(wf2)
        ...     .combine_using("mean")
        ...     .build())
        """
        return cls(name, engine="ensemble")
    
    @classmethod
    def custom(cls, name: str, func: callable) -> 'Workflow':
        """
        Create a workflow with custom forecasting function.
        
        Parameters
        ----------
        name : str
            Workflow name
        func : callable
            Custom function: func(df, h, id_col, time_col, target_col) -> predictions
        
        Returns
        -------
        Workflow
            Workflow instance with custom function
        
        Examples
        --------
        >>> def my_forecast(df, h, id_col, time_col, target_col):
        ...     # Custom logic
        ...     return predictions_df
        >>> 
        >>> wf = Workflow.custom("MyMethod", my_forecast).build()
        """
        instance = cls(name, engine="custom")
        instance._custom_func = func
        return instance
    
    # ============================================
    # MODEL METHODS
    # ============================================
    
    def add_model(self, model: Union[str, Any], **kwargs) -> 'Workflow':
        """
        Add a model to this workflow.
        
        Can be called multiple times to add multiple models.
        All models must be compatible with the specified engine.
        
        Parameters
        ----------
        model : str or model instance
            - String: Model name (e.g., "lgbm", "naive")
            - Instance: Pre-configured model object
        **kwargs
            Model parameters (for string models)
        
        Returns
        -------
        Workflow
            Self for chaining
        
        Examples
        --------
        >>> # String model
        >>> wf.add_model("lgbm", n_estimators=100)
        >>> 
        >>> # Pre-configured instance
        >>> import lightgbm as lgb
        >>> wf.add_model(lgb.LGBMRegressor(n_estimators=100))
        """
        if isinstance(model, str):
            model_instance = NixtlaModels.get(model, **kwargs)
        else:
            model_instance = model
        
        # Validate model matches engine (if engine was explicitly set)
        if self._engine and self._engine not in ("ensemble", "custom"):
            model_engine = self._detect_engine(model_instance)
            if model_engine != self._engine:
                raise ValueError(
                    f"Model '{type(model_instance).__name__}' is for {model_engine}, "
                    f"but workflow uses {self._engine} engine. "
                    f"Use Workflow.{model_engine}('{self.name}') instead."
                )
        
        self._models.append(model_instance)
        return self
    
    def add_models(self, models: List[Union[str, Any]], **shared_kwargs) -> 'Workflow':
        """
        Add multiple models at once.
        
        Parameters
        ----------
        models : list
            List of model names or instances
        **shared_kwargs
            Parameters shared by all string models
        
        Returns
        -------
        Workflow
            Self for chaining
        """
        for model in models:
            self.add_model(model, **shared_kwargs)
        return self
    
    # ============================================
    # CONFIGURATION METHODS
    # ============================================
    
    def use_recipe(self, recipe) -> 'Workflow':
        """Attach preprocessing recipe."""
        self._recipe = recipe
        return self
    
    def with_lags(self, lags: List[int]) -> 'Workflow':
        """Add lag features (MLForecast only)."""
        if self._engine == "statsforecast":
            warnings.warn("StatsForecast doesn't use lags (ignored)", UserWarning)
        self._lags = lags
        return self
    
    def with_lag_transforms(self, transforms: Dict) -> 'Workflow':
        """Add lag transformations (rolling, expanding features)."""
        if self._engine == "statsforecast":
            warnings.warn("StatsForecast doesn't use lag_transforms (ignored)", UserWarning)
        self._lag_transforms = transforms
        return self

    def with_date_features(self, features: List[str]) -> 'Workflow':
        """Add date features (MLForecast only)."""
        if self._engine == "statsforecast":
            warnings.warn("StatsForecast doesn't use date features (ignored)", UserWarning)
        self._date_features = features
        return self
    
    def with_static_features(self, features: List[str]) -> 'Workflow':
        """Specify static features."""
        self._static_features = features
        return self
    
    def with_freq(self, freq: str) -> 'Workflow':
        """Set time series frequency."""
        self._freq = freq
        return self
    
    def with_prediction_intervals(self, h: Optional[int] = None, 
                                  n_windows: int = 2,
                                  method: str = "conformal_error") -> 'Workflow':
        """Enable prediction intervals."""
        self._prediction_intervals = {"h": h, "n_windows": n_windows, "method": method}
        return self
    
    def configure(self, **kwargs) -> 'Workflow':
        """Set custom parameters."""
        self._config.update(kwargs)
        return self
    
    # ============================================
    # ENSEMBLE METHODS
    # ============================================
    
    def add_member(self, workflow: Union['Workflow', Dict, tuple]) -> 'Workflow':
        """Add member to ensemble."""
        if isinstance(workflow, Workflow):
            self._ensemble_members.append(workflow.build())
        else:
            self._ensemble_members.append(workflow)
        return self
    
    def combine_using(self, method: str = "mean", 
                     weights: Optional[List[float]] = None) -> 'Workflow':
        """Set ensemble combination method."""
        self._ensemble_method = method
        self._ensemble_weights = weights
        return self
    
    # ============================================
    # BUILD
    # ============================================
    
    def build(self) -> Dict[str, Any]:
        """Finalize and validate workflow."""
        # Use explicit engine if set, otherwise auto-detect
        if self._engine == "custom":
            return self._build_custom()
        elif self._engine == "ensemble":
            return self._build_ensemble()
        elif self._custom_func:
            return self._build_custom()
        elif self._ensemble_members:
            return self._build_ensemble()
        elif not self._models:
            raise ValueError(f"Workflow '{self.name}' has no models. Use .add_model()")
        else:
            # If no explicit engine, auto-detect
            if not self._engine:
                self._engine = self._detect_engine(self._models[0])
            
            return self._build_standard()
    
    def _build_standard(self) -> Dict[str, Any]:
        """Build standard workflow."""
        engine = self._engine
        
        if engine == "mlforecast":
            params = {
                "models": self._models,
                "freq": self._freq,
                "lags": self._lags,
                "date_features": self._date_features,
                **self._config
            }
            
            pi = None
            if self._prediction_intervals:
                if self._prediction_intervals.get("h") is None:
                    raise ValueError("MLForecast requires 'h' for prediction intervals")
                from mlforecast.utils import PredictionIntervals
                pi = PredictionIntervals(
                    n_windows=self._prediction_intervals["n_windows"],
                    h=self._prediction_intervals["h"],
                    method=self._prediction_intervals.get("method", "conformal_error")
                )
            
            return {
                "name": self.name,
                "engine": "mlforecast",
                "models": self._models,
                "lags": self._lags,
                "date_features": self._date_features,
                "static_features": self._static_features,
                "recipe": self._recipe,
                "prediction_intervals": pi,
                "params": params
            }
        
        elif engine == "statsforecast":
            self._assign_aliases(self._models)
            
            params = {
                "models": self._models,
                "freq": self._freq,
                **self._config
            }
            
            if self._prediction_intervals:
                from statsforecast.utils import ConformalIntervals
                for m in self._models:
                    m.prediction_intervals = ConformalIntervals(
                        n_windows=self._prediction_intervals["n_windows"],
                        h=self._prediction_intervals.get("h", 1),
                        method=self._prediction_intervals.get("method", "conformal_distribution")
                    )
            
            return {
                "name": self.name,
                "engine": "statsforecast",
                "models": self._models,
                "recipe": self._recipe,
                "params": params
            }
        
        elif engine == "neuralforecast":
            params = {
                "models": self._models,
                "freq": self._freq,
                **self._config
            }
            
            return {
                "name": self.name,
                "engine": "neuralforecast",
                "models": self._models,
                "recipe": self._recipe,
                "params": params
            }
        
        else:
            raise ValueError(f"Unknown engine: {engine}")
    
    def _build_ensemble(self) -> Dict[str, Any]:
        """Build ensemble workflow."""
        if not self._ensemble_method:
            self._ensemble_method = "mean"
        
        return {
            "name": self.name,
            "engine": "ensemble",
            "params": {
                "members": self._ensemble_members,
                "method": self._ensemble_method,
                "weights": self._ensemble_weights
            }
        }
    
    def _build_custom(self) -> Dict[str, Any]:
        """Build custom workflow."""
        return {
            "name": self.name,
            "engine": "custom",
            "params": {"func": self._custom_func}
        }
    
    # ============================================
    # UTILITIES
    # ============================================
    
    def _detect_engine(self, model: Any) -> str:
        """Auto-detect engine from model type."""
        model_module = model.__class__.__module__
        
        if "statsforecast" in model_module:
            return "statsforecast"
        if "neuralforecast" in model_module:
            return "neuralforecast"
        return "mlforecast"
    
    def _assign_aliases(self, models: List[Any]) -> None:
        """Assign unique aliases to models."""
        seen = {}
        for m in models:
            if not getattr(m, "alias", None):
                base = m.__class__.__name__
                if base in seen:
                    m.alias = f"{base}_{seen[base]}"
                    seen[base] += 1
                else:
                    m.alias = base
                    seen[base] = 1
    
    def describe(self) -> None:
        """Print workflow description."""
        print(f"\n📋 Workflow: '{self.name}'")
        print(f"   Engine: {self._engine or 'auto-detect'}")
        print(f"   Models: {len(self._models)}")
        for i, m in enumerate(self._models, 1):
            print(f"      {i}. {m.__class__.__name__}")
    
    @classmethod
    def list_available_models(cls) -> Dict[str, List[str]]:
        """List all available models by engine."""
        return NixtlaModels.list_all()
    
    def __repr__(self) -> str:
        engine_str = f", engine='{self._engine}'" if self._engine else ""
        return f"Workflow(name='{self.name}'{engine_str}, models={len(self._models)})"