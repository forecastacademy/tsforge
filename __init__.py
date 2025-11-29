# """
# forecast_pkg - Streamlined Forecasting Package
# ===============================================

# A beginner-friendly forecasting package with unified API and full traceability.
# """

# from .workflow import Workflow
# from .recipe import Recipe
# from .manager import WorkflowManager
# from .lineage import WorkflowLineage, PipelineResults

# # Version
# __version__ = "0.1.0"

# # Main exports
# __all__ = [
#     "Workflow",
#     "Recipe",
#     "WorkflowManager",
#     "WorkflowLineage",
#     "PipelineResults",
# ]

# Backward compatibility (deprecated)
# from .workflow import (
#     make_mlf_workflow,
#     make_sf_workflow,
#     make_nf_workflow,
#     make_ensemble,
#     make_custom_workflow,
# )

from .baselines import *
from .manager import *
from .workflow import *
from .recipe import *