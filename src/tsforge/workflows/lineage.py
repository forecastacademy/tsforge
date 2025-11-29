"""
lineage.py - Pipeline Lineage and Results
==========================================
"""

import pandas as pd
import json
from typing import List, Dict, Optional
from datetime import datetime


class WorkflowLineage:
    """
    Tracks complete lineage of a workflow execution.
    """
    
    def __init__(self, workflow_name: str):
        self.workflow_name = workflow_name
        self.timestamp = datetime.now()
        self.input_shape = None
        self.output_shape = None
        self.recipe_steps = []
        self.model_config = {}
        self.execution_time = None
        self.metrics = {}
        
    def log_input(self, df: pd.DataFrame):
        """Log input data characteristics."""
        self.input_shape = df.shape
        self.input_cols = list(df.columns)
        self.input_dtypes = {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        self.input_memory_mb = round(df.memory_usage(deep=True).sum() / 1024**2, 2)
        
    def log_recipe(self, recipe):
        """Log recipe execution details."""
        if recipe and hasattr(recipe, '_execution_log'):
            self.recipe_steps = recipe._execution_log
            self.recipe_name = recipe.name
        
    def log_model(self, engine: str, models: list, params: dict):
        """Log model configuration."""
        self.engine = engine
        self.model_config = {
            "engine": engine,
            "models": [str(m) for m in (models if isinstance(models, list) else [models])],
            "params": {k: str(v) for k, v in params.items() if k != "models"}
        }
        
    def log_output(self, preds: pd.DataFrame):
        """Log output characteristics."""
        self.output_shape = preds.shape
        self.output_cols = list(preds.columns)
        
    def to_dict(self) -> dict:
        """Export lineage as dictionary."""
        return {
            "workflow_name": self.workflow_name,
            "timestamp": self.timestamp.isoformat(),
            "input": {
                "shape": self.input_shape,
                "columns": self.input_cols[:10] if hasattr(self, 'input_cols') else [],
                "memory_mb": self.input_memory_mb if hasattr(self, 'input_memory_mb') else 0
            },
            "recipe": {
                "name": getattr(self, 'recipe_name', None),
                "steps": self.recipe_steps
            },
            "model": self.model_config,
            "output": {
                "shape": self.output_shape,
                "columns": self.output_cols if hasattr(self, 'output_cols') else []
            },
            "execution_time_sec": self.execution_time
        }
    
    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            f"\n{'='*60}",
            f"WORKFLOW LINEAGE: {self.workflow_name}",
            f"{'='*60}",
            f"\n📅 Executed: {self.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
            f"⏱️  Duration: {self.execution_time:.2f}s" if self.execution_time else "",
            f"\n📊 INPUT DATA:",
            f"   Shape: {self.input_shape}",
            f"   Memory: {self.input_memory_mb if hasattr(self, 'input_memory_mb') else 0} MB",
            f"\n🔧 RECIPE: {getattr(self, 'recipe_name', 'None')}",
        ]
        
        if self.recipe_steps:
            lines.append(f"   Steps executed: {len(self.recipe_steps)}")
            for step in self.recipe_steps:
                if step.get('features_added'):
                    lines.append(f"   ✓ {step['phase']}: +{len(step['features_added'])} features")
        
        lines.extend([
            f"\n🤖 MODEL:",
            f"   Engine: {self.model_config.get('engine', 'Unknown')}",
            f"   Models: {', '.join(self.model_config.get('models', []))}",
            f"\n📈 OUTPUT:",
            f"   Shape: {self.output_shape}",
            f"\n{'='*60}\n"
        ])
        
        return "\n".join(lines)


class PipelineResults:
    """
    Enhanced results with full traceability.
    """
    
    def __init__(self, predictions: pd.DataFrame, lineages: List[WorkflowLineage], 
                 cv_params: dict = None):
        self.predictions = predictions
        self.lineages = lineages
        self.cv_params = cv_params or {}
        
    def summary(self, metric: str = "mae") -> pd.DataFrame:
        """
        Return performance summary across all workflows.
        
        Parameters
        ----------
        metric : str
            Metric to display
            
        Returns
        -------
        pd.DataFrame
            Summary table
        """
        from tsforge.evaluation import score_all
        
        if "cutoff" not in self.predictions.columns:
            raise ValueError("Summary requires CV results (no 'cutoff' column)")
        
        results = []
        target_col = self.cv_params.get("target_col", "sales")
        
        for (wf, model), group in self.predictions.groupby(["workflow", "model"]):
            scores = score_all(group, target_col=target_col, pred_col="yhat")
            
            lineage = next((l for l in self.lineages if l.workflow_name == wf), None)
            
            results.append({
                "workflow": wf,
                "model": model,
                "engine": group["engine"].iloc[0],
                "recipe": lineage.recipe_name if lineage and hasattr(lineage, 'recipe_name') else None,
                **scores
            })
        
        df = pd.DataFrame(results)
        
        if metric in df.columns:
            df = df.sort_values(metric, ascending=True)
        
        return df
    
    def best_model(self, metric: str = "mae") -> dict:
        """Get best performing model with lineage."""
        summary = self.summary(metric)
        best = summary.iloc[0]
        
        lineage = next((l for l in self.lineages if l.workflow_name == best["workflow"]), None)
        
        return {
            "workflow": best["workflow"],
            "model": best["model"],
            "engine": best["engine"],
            metric: best[metric],
            "lineage": lineage.to_dict() if lineage else None
        }
    
    def trace_workflow(self, workflow_name: str) -> WorkflowLineage:
        """Get complete lineage for a workflow."""
        lineage = next((l for l in self.lineages if l.workflow_name == workflow_name), None)
        if not lineage:
            raise ValueError(f"No lineage for '{workflow_name}'")
        return lineage
    
    def compare_workflows(self, workflows: List[str], 
                         metrics: List[str] = None) -> pd.DataFrame:
        """Compare specific workflows."""
        summary = self.summary()
        comparison = summary[summary["workflow"].isin(workflows)]
        
        if metrics:
            cols = ["workflow", "model", "engine"] + [m for m in metrics if m in comparison.columns]
            comparison = comparison[cols]
        
        return comparison
    
    def feature_importance_trace(self, workflow_name: str) -> pd.DataFrame:
        """Trace which features came from which steps."""
        lineage = self.trace_workflow(workflow_name)
        
        if not lineage.recipe_steps:
            return pd.DataFrame(columns=["feature", "source_step", "phase"])
        
        rows = []
        for step in lineage.recipe_steps:
            for feat in step.get("features_added", []):
                rows.append({
                    "feature": feat,
                    "source_step": step["function"],
                    "phase": step["phase"],
                    "step_order": len(rows) + 1
                })
        
        return pd.DataFrame(rows)
    
    def export_lineages(self, path: str):
        """Export all lineages to JSON."""
        data = {
            "cv_params": self.cv_params,
            "lineages": [l.to_dict() for l in self.lineages]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✅ Lineages exported to {path}")
    
    def plot_forecasts(self, workflow: str = None):
        """Plot forecasts (placeholder - implement visualization)."""
        import matplotlib.pyplot as plt
        
        if workflow:
            data = self.predictions[self.predictions["workflow"] == workflow]
        else:
            data = self.predictions
        
        print(f"📊 Plotting forecasts for: {workflow or 'all workflows'}")
        # TODO: Implement actual plotting
        
    def plot_pipeline_dag(self, workflow_name: str):
        """Visualize pipeline as DAG (requires graphviz)."""
        try:
            from graphviz import Digraph
        except ImportError:
            print("⚠️  graphviz not installed. Run: pip install graphviz")
            return
        
        lineage = self.trace_workflow(workflow_name)
        
        dot = Digraph(comment=f"Pipeline: {workflow_name}")
        dot.attr(rankdir="LR")
        
        # Input
        dot.node("input", f"Input\n{lineage.input_shape}", 
                shape="box", style="filled", fillcolor="lightblue")
        
        # Recipe steps
        if lineage.recipe_steps:
            prev = "input"
            for i, step in enumerate(lineage.recipe_steps):
                node_id = f"step_{i}"
                label = f"{step['phase']}\n{step['function']}"
                if step['features_added']:
                    label += f"\n+{len(step['features_added'])}"
                dot.node(node_id, label, shape="box", 
                        style="filled", fillcolor="lightyellow")
                dot.edge(prev, node_id)
                prev = node_id
            last_node = prev
        else:
            last_node = "input"
        
        # Model
        dot.node("model", 
                f"Model\n{lineage.engine}\n{', '.join(lineage.model_config.get('models', []))}",
                shape="box", style="filled", fillcolor="lightgreen")
        dot.edge(last_node, "model")
        
        # Output
        dot.node("output", f"Output\n{lineage.output_shape}", 
                shape="box", style="filled", fillcolor="lightcoral")
        dot.edge("model", "output")
        
        return dot