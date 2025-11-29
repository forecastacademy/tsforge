from dataclasses import dataclass, field
from typing import Callable, List, Any, Optional, Dict
import pandas as pd
import numpy as np

Step = Callable[[pd.DataFrame], pd.DataFrame]

@dataclass
class Recipe:
    def __init__(self, name: str = "Unnamed Recipe", callbacks: Optional[Dict[str, Callable]] = None):
        """
        Initialize a Recipe with an optional name and callbacks.
        
        Parameters
        ----------
        name : str
            Name of the recipe
        callbacks : dict, optional
            Dictionary of callback functions with keys:
            - 'on_start': Called at recipe start with (name, num_steps, input_shape)
            - 'on_step': Called after each step with (step_num, step_name, before_shape, after_shape)
            - 'on_complete': Called at recipe end with (name, final_shape)
        """
        self.name = name
        self.steps = []
        self.step_names = []
        self.callbacks = callbacks or {}
    
    def add_step(self, factory: Callable[..., Step], *args: Any, **kwargs: Any) -> "Recipe":
        """Add a step created by a factory function (returns a Step)."""
        step = factory(*args, **kwargs)
        if not callable(step):
            raise TypeError("Factory must return a callable Step(df)->df")
        self.steps.append(step)
        
        # Store step name for tracking
        step_name = factory.__name__
        if args or kwargs:
            arg_strs = [str(a) for a in args[:2]]
            kwarg_strs = [f"{k}={v}" for k, v in list(kwargs.items())[:2]]
            params = ", ".join(arg_strs + kwarg_strs)
            if len(args) > 2 or len(kwargs) > 2:
                params += ", ..."
            step_name += f"({params})"
        self.step_names.append(step_name)
        
        return self
    
    def summary(self) -> pd.DataFrame:
        """Return a summary of the recipe steps as a DataFrame."""
        if not self.steps:
            return pd.DataFrame({"Step": [], "Function": []})
        
        summary_data = {
            "Step": list(range(1, len(self.steps) + 1)),
            "Function": self.step_names
        }
        return pd.DataFrame(summary_data)
    
    def bake(self, df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
        """
        Apply all steps sequentially with optional logging.
        
        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe
        verbose : bool, optional
            If True and no callbacks provided, print simple progress messages
        
        Returns
        -------
        pd.DataFrame
            Transformed dataframe
        """
        # Callback: on_start
        if 'on_start' in self.callbacks:
            self.callbacks['on_start'](self.name, len(self.steps), df.shape)
        elif verbose:
            print(f"\nRecipe: {self.name} | Steps: {len(self.steps)} | Input: {df.shape}")
        
        out = df.copy()
        
        for i, (fn, step_name) in enumerate(zip(self.steps, self.step_names), 1):
            before_shape = out.shape
            out = fn(out)
            after_shape = out.shape
            
            # Callback: on_step
            if 'on_step' in self.callbacks:
                self.callbacks['on_step'](i, step_name, before_shape, after_shape)
            elif verbose:
                print(f"  [{i}] {step_name}: {before_shape} → {after_shape}")
        
        # Callback: on_complete
        if 'on_complete' in self.callbacks:
            self.callbacks['on_complete'](self.name, out.shape)
        elif verbose:
            print(f"✓ Complete | Output: {out.shape}\n")
        
        return out