import pandas as pd

class CalculateThresholds:
    """
    Analyze and compare data-driven vs textbook thresholds.
    
    Attributes:
        DEFAULT: Standard thresholds from literature
        computed: Data-driven thresholds (median, p25, p75) for each metric
        comparison_df: Table comparing your data vs textbook thresholds
    
    Example:
        analyzer = CalculateThresholds(df, ['trend', 'entropy', 'adi'])

        # Access standard thresholds
        analyzer.DEFAULT['entropy']  # 0.9
        
        # Access computed thresholds
        analyzer['entropy']['median']  # your data's median
        analyzer['entropy']['p75']     # 75th percentile
        
        # Get comparison table
        analyzer.comparison_df
    """
    
    # Classic thresholds from literature
    DEFAULT = {
        'trend': 0.6,
        'seasonal_strength': 0.6,
        'entropy': 0.9,
        'adi': 1.32,        # Syntetos-Boylan
        'cv2': 0.49,        # Syntetos-Boylan
        'lumpiness': 0.8,
        'x_acf1': 0.3,
    }
    
    def __init__(self, df: pd.DataFrame, cols: list):
        """
        Compute thresholds from data.
        
        Args:
            df: DataFrame containing the metric columns
            cols: List of column names to analyze
        """
        self._computed = {}
        self._comparison_rows = []
        
        for col in cols:
            if col not in df.columns:
                continue
            
            series = df[col].dropna()
            
            # Compute data-driven thresholds
            self._computed[col] = {
                'median': series.median(),
                'mean': series.mean(),
                'p25': series.quantile(0.25),
                'p75': series.quantile(0.75),
                'p90': series.quantile(0.90),
                'p95': series.quantile(0.95),
                'min': series.min(),
                'max': series.max(),
                'default': self.DEFAULT.get(col),
            }
            
            # Build comparison row
            median = self._computed[col]['median']
            default = self.DEFAULT.get(col)
            
            if default and default != 0:
                diff_pct = (median - default) / default * 100
                diff_str = f"{diff_pct:+.0f}%"
            else:
                diff_str = "—"
            
            self._comparison_rows.append({
                'Metric': col,
                'Computed Threshold (Median)': f"{median:.2f}",
                'Standard': f"{default:.2f}" if default else "—",
                'Difference': diff_str,
                'Computed p25': f"{self._computed[col]['p25']:.2f}",
                'Computed p75': f"{self._computed[col]['p75']:.2f}",
            })
        
        self.comparison_df = pd.DataFrame(self._comparison_rows).set_index('Metric')
    
    def __getitem__(self, metric: str) -> dict:
        """Access computed thresholds for a metric: analyzer['entropy']"""
        return self._computed[metric]
    
    def get(self, metric: str, stat: str = 'median') -> float:
        """Get a specific statistic for a metric: analyzer.get('entropy', 'p75')"""
        return self._computed[metric][stat]
    
    def get_default(self, metric: str) -> float:
        """Get the classic/textbook threshold for a metric."""
        return self.DEFAULT.get(metric)
    
    @property
    def computed(self) -> dict:
        """All computed thresholds as a dict."""
        return self._computed
    
    @property
    def metrics(self) -> list:
        """List of analyzed metrics."""
        return list(self._computed.keys())
    
    def display(self, caption: str = 'Computed Thresholds vs Standard Thresholds'):
        """Display styled comparison table."""
        return self.comparison_df.style.set_caption(caption)
    
    def __repr__(self):
        return f"CalculateThresholds({self.metrics})"
