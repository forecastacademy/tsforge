from __future__ import annotations

import sys
from types import ModuleType
from typing import Sequence

from tsforge.workflows.recipe import Recipe as _WorkflowRecipe
from tsforge.workflows.recipe import Step

from .encode_features import as_category
from .rolling_features import *
from .summarize import *
from .time_features import *


class _LegacyRecipe(_WorkflowRecipe):
    """Backwards-compatible helper exposing legacy recipe conveniences."""

    def step_as_category(self, cols: Sequence[str]) -> "_LegacyRecipe":
        return self.add_step(as_category, cols)


# Register a synthetic ``tsforge.feature_engineering.recipes`` module so that
# older imports continue to work even though the implementation now lives in
# ``tsforge.workflows.recipe``.
_recipes_module = ModuleType(f"{__name__}.recipes")
_recipes_module.Recipe = _LegacyRecipe
_recipes_module.Step = Step
_recipes_module.__all__ = ["Recipe", "Step"]
sys.modules[_recipes_module.__name__] = _recipes_module

# Re-export for callers doing ``from tsforge.feature_engineering import Recipe``
Recipe = _LegacyRecipe
