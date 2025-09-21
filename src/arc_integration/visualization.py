"""
Visualization tools for ARC training metrics and scorecards.
"""
import os
import json
import time
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from datetime import datetime

from .arc_api_client import ARCScorecard, ScorecardTracker

class TrainingVisualizer:
    """Handles visualization of training metrics and scorecards."""
    
    def __init__(self, output_dir: str = "training_visualizations"):
        """Initialize the visualizer with output directory.
        
        Args:
            output_dir: Directory to save visualization files.
        """
        # Database-only mode: No file-based visualization output
        self.output_dir = None  # Disabled for database-only mode
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def plot_scorecard_metrics(
        self, 
        tracker: ScorecardTracker,
        title: str = "Training Progress",
        show: bool = False,
        save: bool = True
    ) -> Optional[plt.Figure]:
        """Plot metrics from scorecards over time.
        
        Args:
            tracker: ScorecardTracker with metrics to plot.
            title: Title for the plot.
            show: Whether to show the plot.
            save: Whether to save the plot to a file.
            
        Returns:
            The matplotlib Figure object if show=True, else None.
        """
        if not tracker.scorecards:
            print("No scorecards to visualize")
            return None
            
        metrics = tracker.metrics
        timestamps = [
            datetime.fromtimestamp(ts) for ts in metrics['timestamps']
        ]
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16)
        
        # Plot Score Over Time
        axs[0, 0].plot(timestamps, metrics['scores'], 'b-o')
        axs[0, 0].set_title('Score Over Time')
        axs[0, 0].set_ylabel('Score')
        axs[0, 0].grid(True)
        
        # Plot Component Metrics
        axs[0, 1].plot(timestamps, metrics['accuracies'], 'g-o', label='Accuracy')
        axs[0, 1].plot(timestamps, metrics['efficiencies'], 'r-o', label='Efficiency')
        axs[0, 1].plot(timestamps, metrics['generalizations'], 'm-o', label='Generalization')
        axs[0, 1].set_title('Component Metrics')
        axs[0, 1].set_ylabel('Metric Value')
        axs[0, 1].legend()
        axs[0, 1].grid(True)
        
        # Plot Score Distribution
        axs[1, 0].hist(metrics['scores'], bins=10, alpha=0.7, color='c')
        axs[1, 0].axvline(
            np.mean(metrics['scores']), 
            color='r', 
            linestyle='dashed', 
            linewidth=2,
            label=f'Mean: {np.mean(metrics["scores"]):.2f}'
        )
        axs[1, 0].set_title('Score Distribution')
        axs[1, 0].set_xlabel('Score')
        axs[1, 0].set_ylabel('Frequency')
        axs[1, 0].legend()
        
        # Plot Moving Average
        window_size = max(1, len(metrics['scores']) // 5)  # Dynamic window size
        if len(metrics['scores']) > window_size:
            cumsum = np.cumsum(metrics['scores'])
            moving_avg = (cumsum[window_size:] - cumsum[:-window_size]) / window_size
            axs[1, 1].plot(
                timestamps[window_size-1:], 
                metrics['scores'][window_size-1:], 
                'b-', 
                alpha=0.3,
                label='Raw Scores'
            )
            axs[1, 1].plot(
                timestamps[window_size-1:], 
                moving_avg, 
                'r-', 
                label=f'Moving Avg (window={window_size})'
            )
            axs[1, 1].set_title('Score Trend with Moving Average')
            axs[1, 1].set_ylabel('Score')
            axs[1, 1].legend()
            axs[1, 1].grid(True)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        if save:
            filename = self.output_dir / f"training_metrics_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Saved metrics plot to {filename}")
            
        if show:
            plt.show()
            return fig
            
        plt.close()
        return None
    
    def generate_html_report(
        self, 
        tracker: ScorecardTracker,
        title: str = "Training Report",
        save: bool = True
    ) -> Optional[str]:
        """Generate an HTML report of training progress.
        
        Args:
            tracker: ScorecardTracker with metrics to include.
            title: Title for the report.
            save: Whether to save the report to a file.
            
        Returns:
            The HTML content if save=False, else None.
        """
        if not tracker.scorecards:
            return "<p>No scorecards available for report.</p>"
            
        summary = tracker.get_summary()
        
        # Extract variables for f-string
        total = summary.get('total_scorecards', 0)
        avg_score = summary.get('avg_score', 0)
        max_score = summary.get('max_score', 0)
        min_score = summary.get('min_score', 0)
        latest_score = summary.get('latest_score', 0)
        improvement = summary.get('improvement', 0)
        trend_class = 'trend-up' if improvement > 0 else ('trend-down' if improvement < 0 else 'trend-neutral')
        timestamp = self.timestamp
        date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Generate HTML
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                .header {{ text-align: center; margin-bottom: 30px; }}
                .summary {{ 
                    background-color: #f5f5f5; 
                    padding: 15px; 
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .metrics {{ 
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    margin-bottom: 20px;
                }}
                .metric-card {{
                    background-color: #fff;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 15px;
                    text-align: center;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }}
                .metric-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #2c3e50;
                    margin: 10px 0;
                }}
                .metric-label {{
                    color: #7f8c8d;
                    font-size: 14px;
                }}
                .trend-up {{ color: #27ae60; }}
                .trend-down {{ color: #e74c3c; }}
                .trend-neutral {{ color: #7f8c8d; }}
                img {{ max-width: 100%; height: auto; margin: 10px 0; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>{title}</h1>
                    <p>Generated on {date}</p>
                </div>
                
                <div class="summary">
                    <h2>Training Summary</h2>
                    <p>Total scorecards: {total}</p>
                    <p>Latest score: <strong>{latest_score:.2f}</strong> (Range: {min_score:.2f} - {max_score:.2f})</p>
                    <p>Overall improvement: {improvement:+.2f}</p>
                </div>
                
                <div class="metrics">
                    <div class="metric-card">
                        <div class="metric-label">Average Score</div>
                        <div class="metric-value">{avg_score:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Best Score</div>
                        <div class="metric-value">{max_score:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Latest Score</div>
                        <div class="metric-value {trend_class}">{latest_score:.2f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Improvement</div>
                        <div class="metric-value {trend_class}">{improvement:+.2f}</div>
                    </div>
                </div>
                
                <h2>Training Metrics</h2>
                <img src="training_metrics_{timestamp}.png" alt="Training Metrics">
                
                <h2>Recent Scorecards</h2>
                <table border="1" cellpadding="8" cellspacing="0" style="width:100%; border-collapse: collapse;">
                    <tr style="background-color: #f2f2f2;">
                        <th>Timestamp</th>
                        <th>Score</th>
                        <th>Accuracy</th>
                        <th>Efficiency</th>
                        <th>Generalization</th>
                    </tr>
        """.format(
            title=title,
            date=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total=summary.get('total_scorecards', 0),
            avg_score=summary.get('avg_score', 0),
            max_score=summary.get('max_score', 0),
            min_score=summary.get('min_score', 0),
            latest_score=summary.get('latest_score', 0),
            improvement=summary.get('improvement', 0),
            trend_class='trend-up' if summary.get('improvement', 0) > 0 else 
                     ('trend-down' if summary.get('improvement', 0) < 0 else 'trend-neutral'),
            timestamp=self.timestamp
        )
        
        # Add recent scorecards to the table
        for scorecard in tracker.scorecards[-10:]:  # Show last 10 scorecards
            html_content += f"""
                <tr>
                    <td>{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(scorecard.timestamp))}</td>
                    <td>{scorecard.score:.2f}</td>
                    <td>{scorecard.accuracy:.2f}</td>
                    <td>{scorecard.efficiency:.2f}</td>
                    <td>{scorecard.generalization:.2f}</td>
                </tr>
            """
            
        html_content += """
                </table>
            </div>
        </body>
        </html>
        """
        
        if save:
            # Database-only mode: No file saving
            html_filename = f"training_report_{timestamp}.html"
            # Save the metrics plot
            self.plot_scorecard_metrics(tracker, save=True, show=False)
            
            print(f"Saved training report to {html_filename}")
            return None
            
        return html_content
