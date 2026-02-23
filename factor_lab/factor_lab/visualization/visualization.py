"""
visualization.py - Rich Visualizations for Manifold Analysis
=============================================================

Provides comprehensive visualizations using seaborn and plotly for
manifold analysis results.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Any, Optional

# Try plotly import
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False
    print("Warning: plotly not available. Some visualizations disabled.")


def set_style():
    """Set consistent plotting style."""
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.1)
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300


def create_manifold_dashboard(results: Dict[str, Any], output_path: Optional[Path] = None):
    """
    Create comprehensive dashboard for manifold analysis.
    
    Uses seaborn for clean, publication-quality matplotlib figures.
    """
    set_style()
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # 1. Eigenvalue Comparison
    if 'true_eigenvalues' in results and 'sample_eigenvalues' in results:
        ax = fig.add_subplot(gs[0, 0])
        true_ev = results['true_eigenvalues']
        sample_ev = results['sample_eigenvalues']
        
        x = np.arange(1, len(true_ev) + 1)
        ax.plot(x, true_ev, 'o-', label='True (Implicit)', linewidth=2, markersize=8)
        ax.plot(x, sample_ev, 's-', label='Sample (PCA)', linewidth=2, markersize=8)
        
        ax.set_xlabel('Eigenvalue Index', fontsize=12)
        ax.set_ylabel('Eigenvalue', fontsize=12)
        ax.set_title('Eigenvalue Spectrum Comparison', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
    
    # 2. Principal Angles
    if 'principal_angles' in results:
        ax = fig.add_subplot(gs[0, 1])
        angles = results['principal_angles']
        
        x = np.arange(len(angles))
        sns.barplot(x=x, y=angles, ax=ax, hue=x, palette='viridis', legend=False)
        ax.set_xlabel('Subspace Dimension', fontsize=12)
        ax.set_ylabel('Principal Angle (radians)', fontsize=12)
        ax.set_title('Principal Angles', fontweight='bold', fontsize=14)
        
        # Add degree labels
        for i, angle in enumerate(angles):
            ax.text(i, angle, f'{np.degrees(angle):.1f}°', 
                   ha='center', va='bottom', fontsize=9)
    
    # 3. Manifold Distances
    if all(k in results for k in ['dist_grassmannian', 'dist_procrustes', 'dist_chordal']):
        ax = fig.add_subplot(gs[0, 2])
        
        metrics = ['Grassmannian\n(Subspace)', 'Procrustes\n(Aligned)', 'Chordal\n(Raw)']
        values = [
            results['dist_grassmannian'],
            results['dist_procrustes'],
            results['dist_chordal'],
        ]
        colors = ['#2ecc71', '#3498db', '#e74c3c']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title('Manifold Distances', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # 4. Eigenvalue Errors
    if 'eigenvalue_errors' in results:
        ax = fig.add_subplot(gs[0, 3])
        errors = results['eigenvalue_errors']
        
        x = np.arange(1, len(errors) + 1)
        ax.plot(x, errors, 'o-', linewidth=2, markersize=8, color='#e74c3c')
        ax.axhline(0, color='black', linestyle='--', linewidth=2, alpha=0.5)
        ax.fill_between(x, 0, errors, alpha=0.3, color='#e74c3c')
        
        ax.set_xlabel('Eigenvalue Index', fontsize=12)
        ax.set_ylabel('Error (Sample - True)', fontsize=12)
        ax.set_title('Eigenvalue Errors', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3)
    
    # 5. Eigenvector Correlations
    if 'vector_correlations' in results:
        ax = fig.add_subplot(gs[1, 0])
        corrs = results['vector_correlations']
        
        colors = ['#2ecc71' if c > 0.9 else '#f39c12' if c > 0.7 else '#e74c3c' for c in corrs]
        bars = ax.bar(np.arange(1, len(corrs) + 1), corrs, color=colors, 
                     alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.axhline(0.9, color='green', linestyle='--', alpha=0.5, linewidth=2, label='90% threshold')
        ax.axhline(0.7, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='70% threshold')
        
        ax.set_xlabel('Eigenvector Index', fontsize=12)
        ax.set_ylabel('Absolute Correlation', fontsize=12)
        ax.set_title('Eigenvector Correlations', fontweight='bold', fontsize=14)
        ax.set_ylim([0, 1.05])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    
    # 6. Relative Eigenvalue Errors
    if 'eigenvalue_relative_errors' in results:
        ax = fig.add_subplot(gs[1, 1])
        rel_errors = results['eigenvalue_relative_errors']
        
        x = np.arange(1, len(rel_errors) + 1)
        ax.bar(x, np.abs(rel_errors) * 100, alpha=0.7, color='#3498db', edgecolor='black')
        
        ax.set_xlabel('Eigenvalue Index', fontsize=12)
        ax.set_ylabel('Relative Error (%)', fontsize=12)
        ax.set_title('Relative Eigenvalue Errors', fontweight='bold', fontsize=14)
        ax.grid(True, alpha=0.3, axis='y')
    
    # 7. Summary Statistics Box
    ax = fig.add_subplot(gs[1, 2:])
    ax.axis('off')
    
    summary_text = "Summary Statistics\n" + "="*50 + "\n\n"
    
    if 'dist_grassmannian' in results:
        summary_text += f"Grassmannian Distance: {results['dist_grassmannian']:.6f}\n"
    if 'dist_procrustes' in results:
        summary_text += f"Procrustes Distance:   {results['dist_procrustes']:.6f}\n"
    if 'eigenvalue_rmse' in results:
        summary_text += f"Eigenvalue RMSE:       {results['eigenvalue_rmse']:.6f}\n"
    if 'mean_correlation' in results:
        summary_text += f"Mean Eigenvector Corr: {results['mean_correlation']:.4f}\n"
    if 'subspace_distance' in results:
        summary_text += f"Subspace Distance:     {results['subspace_distance']:.6f}\n"
    
    ax.text(0.1, 0.5, summary_text, fontsize=14, family='monospace',
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 8. Loading Heatmap (if available)
    if 'true_eigenvectors' in results and 'sample_eigenvectors' in results:
        ax = fig.add_subplot(gs[2, :2])
        true_ev = results['true_eigenvectors']
        
        # Show first few eigenvectors
        n_show = min(3, true_ev.shape[0])
        n_assets = min(50, true_ev.shape[1])
        
        sns.heatmap(true_ev[:n_show, :n_assets], 
                   cmap='RdBu_r', center=0, ax=ax,
                   cbar_kws={'label': 'Loading'})
        ax.set_xlabel('Asset Index', fontsize=12)
        ax.set_ylabel('Eigenvector', fontsize=12)
        ax.set_title('True Eigenvector Loadings (Heatmap)', fontweight='bold', fontsize=14)
    
    # 9. Comparison Heatmap
    if 'sample_eigenvectors' in results:
        ax = fig.add_subplot(gs[2, 2:])
        sample_ev = results['sample_eigenvectors']
        
        n_show = min(3, sample_ev.shape[0])
        n_assets = min(50, sample_ev.shape[1])
        
        sns.heatmap(sample_ev[:n_show, :n_assets],
                   cmap='RdBu_r', center=0, ax=ax,
                   cbar_kws={'label': 'Loading'})
        ax.set_xlabel('Asset Index', fontsize=12)
        ax.set_ylabel('Eigenvector', fontsize=12)
        ax.set_title('Sample Eigenvector Loadings (Heatmap)', fontweight='bold', fontsize=14)
    
    plt.suptitle('Manifold Analysis Dashboard', fontsize=18, fontweight='bold', y=0.98)
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Dashboard saved: {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_interactive_plotly_dashboard(results: Dict[str, Any], output_path: Optional[Path] = None):
    """
    Create interactive dashboard using plotly.
    
    Provides zoom, hover, and interactive exploration.
    """
    if not HAS_PLOTLY:
        print("Plotly not available. Skipping interactive dashboard.")
        return
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Eigenvalue Comparison', 'Manifold Distances',
                       'Principal Angles', 'Eigenvector Correlations'),
        specs=[[{'type': 'scatter'}, {'type': 'bar'}],
               [{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    # 1. Eigenvalue Comparison
    if 'true_eigenvalues' in results and 'sample_eigenvalues' in results:
        true_ev = results['true_eigenvalues']
        sample_ev = results['sample_eigenvalues']
        x = np.arange(1, len(true_ev) + 1)
        
        fig.add_trace(
            go.Scatter(x=x, y=true_ev, mode='lines+markers', name='True',
                      line=dict(color='blue', width=2), marker=dict(size=8)),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=sample_ev, mode='lines+markers', name='Sample',
                      line=dict(color='red', width=2), marker=dict(size=8, symbol='square')),
            row=1, col=1
        )
    
    # 2. Manifold Distances
    if all(k in results for k in ['dist_grassmannian', 'dist_procrustes', 'dist_chordal']):
        metrics = ['Grassmannian', 'Procrustes', 'Chordal']
        values = [
            results['dist_grassmannian'],
            results['dist_procrustes'],
            results['dist_chordal'],
        ]
        colors = ['green', 'blue', 'red']
        
        fig.add_trace(
            go.Bar(x=metrics, y=values, marker_color=colors,
                  text=[f'{v:.4f}' for v in values], textposition='outside'),
            row=1, col=2
        )
    
    # 3. Principal Angles
    if 'principal_angles' in results:
        angles = results['principal_angles']
        x = np.arange(1, len(angles) + 1)
        
        fig.add_trace(
            go.Bar(x=x, y=angles, marker_color='purple',
                  text=[f'{np.degrees(a):.1f}°' for a in angles], 
                  textposition='outside'),
            row=2, col=1
        )
    
    # 4. Eigenvector Correlations
    if 'vector_correlations' in results:
        corrs = results['vector_correlations']
        x = np.arange(1, len(corrs) + 1)
        colors = ['green' if c > 0.9 else 'orange' if c > 0.7 else 'red' for c in corrs]
        
        fig.add_trace(
            go.Bar(x=x, y=corrs, marker_color=colors,
                  text=[f'{c:.3f}' for c in corrs], textposition='outside'),
            row=2, col=2
        )
    
    fig.update_layout(
        title_text="Interactive Manifold Analysis Dashboard",
        showlegend=True,
        height=800,
        hovermode='closest'
    )
    
    if output_path:
        fig.write_html(str(output_path))
        print(f"✓ Interactive dashboard saved: {output_path}")
    else:
        fig.show()


def print_verbose_results(results: Dict[str, Any], title: str = "Analysis Results"):
    """
    Print verbose, formatted results matching Gemini's style.
    """
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)
    
    # Manifold Distances
    if 'dist_grassmannian' in results:
        print("\n📐 MANIFOLD DISTANCES")
        print("   ↳ Grassmannian (Subspace):  {:.6f}".format(results['dist_grassmannian']))
        if 'dist_procrustes' in results:
            print("   ↳ Procrustes (Aligned):     {:.6f}".format(results['dist_procrustes']))
        if 'dist_chordal' in results:
            print("   ↳ Chordal (Raw):            {:.6f}".format(results['dist_chordal']))
    
    # Principal Angles
    if 'principal_angles' in results:
        print("\n📊 PRINCIPAL ANGLES")
        angles = results['principal_angles']
        for i, angle in enumerate(angles):
            print(f"   Angle {i+1}: {angle:.6f} rad  ({np.degrees(angle):.2f}°)")
    
    # Eigenvalues
    if 'true_eigenvalues' in results and 'sample_eigenvalues' in results:
        print("\n🔢 EIGENVALUE COMPARISON")
        print("   Index | True (Implicit) | Sample (PCA) | Error    | Rel Error")
        print("   " + "-"*68)
        
        true_ev = results['true_eigenvalues']
        sample_ev = results['sample_eigenvalues']
        errors = results.get('eigenvalue_errors', true_ev - sample_ev)
        rel_errors = results.get('eigenvalue_relative_errors', errors / true_ev)
        
        for i in range(len(true_ev)):
            print(f"   {i+1:5d} | {true_ev[i]:15.6f} | {sample_ev[i]:12.6f} | "
                  f"{errors[i]:8.6f} | {rel_errors[i]:8.1%}")
        
        if 'eigenvalue_rmse' in results:
            print(f"\n   RMSE: {results['eigenvalue_rmse']:.6f}")
    
    # Eigenvector Correlations
    if 'vector_correlations' in results:
        print("\n🎯 EIGENVECTOR CORRELATIONS")
        corrs = results['vector_correlations']
        for i, corr in enumerate(corrs):
            status = "✓" if corr > 0.9 else "○" if corr > 0.7 else "✗"
            print(f"   Vector {i+1}: {corr:.4f}  {status}")
        
        if 'mean_correlation' in results:
            print(f"\n   Mean: {results['mean_correlation']:.4f}")
            print(f"   Min:  {results['min_correlation']:.4f}")
            print(f"   Max:  {results['max_correlation']:.4f}")
    
    # Summary Assessment
    print("\n💡 ASSESSMENT")
    if 'dist_grassmannian' in results:
        dist_g = results['dist_grassmannian']
        if dist_g < 0.1:
            print("   ✓ Excellent subspace recovery (Grassmannian < 0.1)")
        elif dist_g < 0.3:
            print("   ○ Good subspace recovery (Grassmannian < 0.3)")
        else:
            print("   ⚠ Moderate subspace recovery (Grassmannian ≥ 0.3)")
    
    if 'mean_correlation' in results:
        mean_corr = results['mean_correlation']
        if mean_corr > 0.95:
            print("   ✓ Excellent eigenvector recovery (Mean corr > 0.95)")
        elif mean_corr > 0.85:
            print("   ○ Good eigenvector recovery (Mean corr > 0.85)")
        else:
            print("   ⚠ Moderate eigenvector recovery (Mean corr ≤ 0.85)")
    
    print("\n" + "="*70 + "\n")
