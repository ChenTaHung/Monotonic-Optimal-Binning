# `plot_binning_stability` Function Documentation

## Overview
The `plot_binning_stability` function compares binning results between training and test datasets (or across time periods) to assess the stability and robustness of the binning solution. It helps identify distribution shifts, validates model generalization, and ensures consistent bin performance.

## Function Signature
```python
def plot_binning_stability(
    train_summary: pd.DataFrame,
    test_summary: pd.DataFrame,
    *,
    figsize: Tuple[float, float] = (16, 10),
    title: Optional[str] = None,
    metrics: List[str] = None,
    plot_types: List[str] = None,
    color_train: str = "#1E88E5",
    color_test: str = "#43A047",
    show_psi: bool = True,
    psi_threshold: float = 0.1,
    show_divergence: bool = True,
    show_correlation: bool = True,
    confidence_level: float = 0.95,
    show_confidence: bool = True,
    layout: str = "grid",
    save_path: Optional[str] = None,
    dpi: int = 100
) -> Figure
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| **train_summary** | `pd.DataFrame` | required | Training set binning summary |
| **test_summary** | `pd.DataFrame` | required | Test set binning summary |
| **figsize** | `Tuple[float, float]` | `(16, 10)` | Figure size |
| **title** | `Optional[str]` | `None` | Main figure title |
| **metrics** | `List[str]` | `None` | Metrics to compare ('count_pct', 'mean', 'woe') |
| **plot_types** | `List[str]` | `None` | Types of plots ('bars', 'lines', 'scatter', 'heatmap') |
| **color_train** | `str` | `"#1E88E5"` | Color for training data (blue) |
| **color_test** | `str` | `"#43A047"` | Color for test data (green) |
| **show_psi** | `bool` | `True` | Calculate and show PSI |
| **psi_threshold** | `float` | `0.1` | PSI threshold for stability |
| **show_divergence** | `bool` | `True` | Show distribution divergence |
| **show_correlation** | `bool` | `True` | Show metric correlations |
| **confidence_level** | `float` | `0.95` | Confidence level for intervals |
| **show_confidence** | `bool` | `True` | Show confidence bands |
| **layout** | `str` | `"grid"` | Layout type ('grid', 'vertical', 'horizontal') |
| **save_path** | `Optional[str]` | `None` | Path to save figure |
| **dpi** | `int` | `100` | DPI for saved figure |

## Returns
- **Figure**: Matplotlib Figure object with stability analysis plots

## Default Metrics
If not specified, the function compares:
- Sample distribution (`count_pct`)
- Event rates (`mean`)
- Weight of Evidence (`woe`) for binary targets
- Information Value (`iv`) for binary targets

## Usage Examples

### Basic Stability Check
```python
from MOBPY.plot import plot_binning_stability

# Fit binner on train and apply to test
train_summary = binner.summary_()  # From training data

# Apply same bins to test data
test_binner = MonotonicBinner(test_df, x='feature', y='target')
test_binner.fit()
test_summary = test_binner.summary_()

# Compare stability
fig = plot_binning_stability(train_summary, test_summary)
plt.show()
```

### Custom Metrics Comparison
```python
fig = plot_binning_stability(
    train_summary, test_summary,
    metrics=['count_pct', 'mean', 'woe'],
    plot_types=['bars', 'lines'],
    title="Model Stability Analysis: Train vs Test",
    show_psi=True,
    show_divergence=True
)

plt.tight_layout()
plt.show()
```

### Time Period Comparison
```python
# Compare different time periods
q1_summary = binner_q1.summary_()
q2_summary = binner_q2.summary_()

fig = plot_binning_stability(
    q1_summary, q2_summary,
    title="Temporal Stability: Q1 vs Q2",
    color_train='#1976D2',  # Q1 in blue
    color_test='#F57C00',    # Q2 in orange
    show_psi=True,
    psi_threshold=0.25  # Higher threshold for temporal
)
```

## Stability Metrics

### Population Stability Index (PSI)
```python
PSI = Σ[(Test% - Train%) × ln(Test% / Train%)]
```

**Interpretation:**
- PSI < 0.1: No significant shift
- PSI 0.1-0.25: Small shift, monitor
- PSI > 0.25: Significant shift, investigate

### Characteristic Stability Index (CSI)
Similar to PSI but for event rates rather than population

### Divergence Metrics
- KL Divergence
- JS Divergence
- Wasserstein Distance
- Kolmogorov-Smirnov Statistic

## Advanced Features

### Detailed PSI Analysis
```python
def calculate_detailed_psi(train_summary, test_summary):
    """Calculate PSI with bin-level contributions."""
    
    train_pct = train_summary['count_pct'] / 100
    test_pct = test_summary['count_pct'] / 100
    
    # Add small constant to avoid log(0)
    epsilon = 1e-10
    train_pct = train_pct + epsilon
    test_pct = test_pct + epsilon
    
    # PSI calculation
    psi_components = (test_pct - train_pct) * np.log(test_pct / train_pct)
    total_psi = psi_components.sum()
    
    # Create detailed report
    psi_report = pd.DataFrame({
        'bin': train_summary['bucket'],
        'train_pct': train_pct,
        'test_pct': test_pct,
        'difference': test_pct - train_pct,
        'psi_contribution': psi_components,
        'psi_pct': psi_components / total_psi * 100
    })
    
    return total_psi, psi_report

# Use in plot
fig = plot_binning_stability(train_summary, test_summary)

# Add PSI details
psi, psi_details = calculate_detailed_psi(train_summary, test_summary)
print(f"Total PSI: {psi:.4f}")
print("\nTop PSI Contributors:")
print(psi_details.nlargest(3, 'psi_contribution'))
```

### Confidence Intervals
```python
import scipy.stats as stats

fig = plot_binning_stability(
    train_summary, test_summary,
    show_confidence=True,
    confidence_level=0.95
)

# Add custom confidence bands
ax = fig.axes[0]  # Get first subplot

for i, (idx, row) in enumerate(train_summary.iterrows()):
    n = row['count']
    p = row['mean']
    
    # Calculate confidence interval
    se = np.sqrt(p * (1 - p) / n)
    ci = stats.norm.ppf(0.975) * se
    
    # Plot confidence band
    ax.fill_between([i-0.2, i+0.2], 
                    [p-ci, p-ci], [p+ci, p+ci],
                    alpha=0.3, color='blue')
```

### Stability Over Time
```python
# Multiple time periods
periods = {
    'Jan': jan_summary,
    'Feb': feb_summary,
    'Mar': mar_summary,
    'Apr': apr_summary
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
baseline = periods['Jan']

for ax, (month, summary) in zip(axes.flat, list(periods.items())[1:]):
    # Calculate PSI vs baseline
    psi = calculate_psi(baseline, summary)
    
    # Plot comparison
    x = range(len(summary))
    width = 0.35
    
    ax.bar(x - width/2, baseline['count_pct'], width, 
           label='Baseline (Jan)', alpha=0.7)
    ax.bar(x + width/2, summary['count_pct'], width, 
           label=f'{month}', alpha=0.7)
    
    ax.set_title(f'{month} vs Baseline (PSI: {psi:.3f})')
    ax.legend()

fig.suptitle('Temporal Stability Analysis', fontsize=16, fontweight='bold')
plt.tight_layout()
```

## Visualization Types

### Side-by-Side Bars
```python
fig = plot_binning_stability(
    train_summary, test_summary,
    plot_types=['bars'],
    title="Distribution Comparison"
)
```

### Line Comparison
```python
fig = plot_binning_stability(
    train_summary, test_summary,
    plot_types=['lines'],
    title="Trend Comparison"
)
```

### Scatter Plot
```python
fig = plot_binning_stability(
    train_summary, test_summary,
    plot_types=['scatter'],
    title="Correlation Analysis"
)
```

### Heatmap Comparison
```python
fig = plot_binning_stability(
    train_summary, test_summary,
    plot_types=['heatmap'],
    title="Stability Heatmap"
)
```

## Custom Layouts

### Vertical Layout
```python
fig = plot_binning_stability(
    train_summary, test_summary,
    layout='vertical',
    figsize=(10, 16),
    metrics=['count_pct', 'mean', 'woe', 'iv']
)
```

### Dashboard Layout
```python
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Main comparison
ax_main = fig.add_subplot(gs[:2, :2])
plot_distribution_comparison(train_summary, test_summary, ax=ax_main)

# PSI gauge
ax_psi = fig.add_subplot(gs[0, 2])
plot_psi_gauge(train_summary, test_summary, ax=ax_psi)

# Correlation
ax_corr = fig.add_subplot(gs[1, 2])
plot_correlation(train_summary, test_summary, ax=ax_corr)

# Metrics table
ax_table = fig.add_subplot(gs[2, :])
plot_metrics_table(train_summary, test_summary, ax=ax_table)

fig.suptitle('Comprehensive Stability Dashboard', fontsize=16, fontweight='bold')
plt.tight_layout()
```

## Quality Assessment

### Stability Score
```python
def calculate_stability_score(train_summary, test_summary):
    """Calculate overall stability score (0-100)."""
    
    score = 100.0
    
    # PSI component (40 points)
    psi = calculate_psi(train_summary, test_summary)
    if psi < 0.1:
        psi_score = 40
    elif psi < 0.25:
        psi_score = 40 * (1 - (psi - 0.1) / 0.15)
    else:
        psi_score = 0
    
    # Event rate correlation (30 points)
    corr = np.corrcoef(train_summary['mean'], test_summary['mean'])[0, 1]
    corr_score = 30 * max(0, corr)
    
    # Sample distribution similarity (30 points)
    from scipy.stats import wasserstein_distance
    dist = wasserstein_distance(
        train_summary['count_pct'], 
        test_summary['count_pct']
    )
    dist_score = 30 * max(0, 1 - dist / 100)
    
    total_score = psi_score + corr_score + dist_score
    
    return {
        'total': total_score,
        'psi_score': psi_score,
        'correlation_score': corr_score,
        'distribution_score': dist_score,
        'grade': 'A' if total_score >= 90 else 
                 'B' if total_score >= 75 else
                 'C' if total_score >= 60 else
                 'D' if total_score >= 40 else 'F'
    }

stability = calculate_stability_score(train_summary, test_summary)
print(f"Stability Grade: {stability['grade']} ({stability['total']:.1f}/100)")
```

### Warning Detection
```python
def detect_stability_warnings(train_summary, test_summary):
    """Detect potential stability issues."""
    
    warnings = []
    
    # Check PSI
    psi = calculate_psi(train_summary, test_summary)
    if psi > 0.25:
        warnings.append(f"HIGH PSI: {psi:.3f} indicates significant distribution shift")
    elif psi > 0.1:
        warnings.append(f"MODERATE PSI: {psi:.3f} suggests some distribution change")
    
    # Check event rate changes
    rate_changes = abs(test_summary['mean'] - train_summary['mean'])
    if (rate_changes > 0.1).any():
        warnings.append("Large event rate changes detected in some bins")
    
    # Check sample distribution
    pct_changes = abs(test_summary['count_pct'] - train_summary['count_pct'])
    if (pct_changes > 20).any():
        warnings.append("Significant sample redistribution across bins")
    
    # Check for empty bins
    if (test_summary['count'] == 0).any():
        warnings.append("Empty bins detected in test data")
    
    return warnings

warnings = detect_stability_warnings(train_summary, test_summary)
for warning in warnings:
    print(f"⚠️ {warning}")
```

## Common Issues and Solutions

### Issue: Misaligned Bins
```python
# Solution: Ensure same binning applied
# Use transform method instead of refitting
test_transformed = train_binner.transform(test_df['feature'])
```

### Issue: PSI Calculation Errors
```python
# Solution: Handle zero frequencies
def safe_psi_calculation(train_pct, test_pct):
    # Add small constant
    epsilon = 1e-10
    train_pct = np.maximum(train_pct, epsilon)
    test_pct = np.maximum(test_pct, epsilon)
    
    # Normalize to sum to 1
    train_pct = train_pct / train_pct.sum()
    test_pct = test_pct / test_pct.sum()
    
    # Calculate PSI
    return np.sum((test_pct - train_pct) * np.log(test_pct / train_pct))
```

### Issue: Visual Clutter
```python
# Solution: Separate into multiple figures
metrics = ['count_pct', 'mean', 'woe', 'iv']
figs = []

for metric in metrics:
    fig = plot_binning_stability(
        train_summary, test_summary,
        metrics=[metric],
        figsize=(12, 6),
        title=f'Stability Analysis: {metric}'
    )
    figs.append(fig)

# Save all figures
for i, fig in enumerate(figs):
    fig.savefig(f'stability_{metrics[i]}.png')
```

## Integration with Model Monitoring

### Automated Stability Reporting
```python
def generate_stability_report(train_summary, test_summary, output_path='stability_report.html'):
    """Generate HTML stability report."""
    
    import base64
    from io import BytesIO
    
    # Create plots
    fig = plot_binning_stability(train_summary, test_summary)
    
    # Convert to base64 for HTML embedding
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode()
    
    # Calculate metrics
    psi = calculate_psi(train_summary, test_summary)
    stability_score = calculate_stability_score(train_summary, test_summary)
    
    # Generate HTML
    html = f"""
    <html>
    <head>
        <title>Binning Stability Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; }}
            .warning {{ color: red; font-weight: bold; }}
            .pass {{ color: green; font-weight: bold; }}
        </style>
    </head>
    <body>
        <h1>Binning Stability Analysis</h1>
        
        <div class="metric">
            <h2>Overall Stability Score: {stability_score['grade']} ({stability_score['total']:.1f}/100)</h2>
        </div>
        
        <div class="metric">
            <h3>Population Stability Index (PSI): {psi:.4f}</h3>
            <p class="{'warning' if psi > 0.25 else 'pass'}">
                {'⚠️ Significant distribution shift detected' if psi > 0.25 else '✓ Distribution stable'}
            </p>
        </div>
        
        <h2>Visual Analysis</h2>
        <img src="data:image/png;base64,{img_base64}" width="100%">
        
        <h2>Recommendations</h2>
        <ul>
            {'<li>Consider retraining the model</li>' if psi > 0.25 else ''}
            {'<li>Monitor closely for further shifts</li>' if 0.1 < psi <= 0.25 else ''}
            {'<li>Current binning remains stable</li>' if psi <= 0.1 else ''}
        </ul>
        
        <footer>
            <p>Generated: {pd.Timestamp.now()}</p>
        </footer>
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html)
    
    return output_path

# Generate report
report_path = generate_stability_report(train_summary, test_summary)
print(f"Report saved to: {report_path}")
```

## See Also
- [`plot_bin_statistics`](./plot_bin_statistics.md) - Comprehensive single-dataset analysis
- [`plot_sample_distribution`](./plot_sample_distribution.md) - Distribution comparison
- [`MonotonicBinner`](../binning/mob.md) - Main binning class
- [`calculate_psi`](../core/utils.md#calculate_psi) - PSI calculation details
