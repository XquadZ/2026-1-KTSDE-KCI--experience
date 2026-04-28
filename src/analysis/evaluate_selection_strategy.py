import pandas as pd
from pathlib import Path
import os

# Configuration: Standard relative path for GitHub repository
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs" / "stats"

def format_float(val):
    """Format decimal numbers."""
    if pd.isna(val):
        return "NaN"
    return f"{val:.4f}"

def print_table4_preensemble():
    """Process and print Table 4: Score S indicators."""
    csv_path = OUTPUT_DIR / "table4_preensemble_indicators.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df_all = df[df['Scale'] == 'all'].sort_values('S', ascending=False)
    
    # Filter columns and rename for paper format
    cols = ['k', 'Combo', 'Gain_vs_Best', 'Avg_Disagreement', 'Avg_Comp', 'Avg_UFP', 'Avg_TFC', 'S']
    df_print = df_all[cols].copy()
    
    rename_map = {
        'Gain_vs_Best': 'Gain_miss',
        'Avg_Disagreement': 'Dis',
        'Avg_Comp': 'Comp',
        'Avg_UFP': 'UFP',
        'Avg_TFC': 'TFC',
        'S': 'Score S'
    }
    df_print.rename(columns=rename_map, inplace=True)
    
    print("\n" + "="*80)
    print(" TABLE 4: Pre-ensemble Indicators & Score S (Top 10)")
    print("="*80)
    print(df_print.head(10).to_markdown(index=False, floatfmt=".4f"))

def print_table5_actual_gains():
    """Process and print Table 5: Actual ensemble mAP performance."""
    csv_path = OUTPUT_DIR / "table5_actual_ensemble_gains.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    cols = ['k', 'Method', 'Combo', 'EnAP', 'GainAP', 'EnAR', 'GainAR']
    df_print = df[cols].sort_values(['k', 'Method', 'GainAP'], ascending=[True, True, False])
    
    print("\n" + "="*80)
    print(" TABLE 5: Actual Ensemble Performance & Gain (Top 5 per category)")
    print("="*80)
    
    # Group results by k and Method
    for k in [2, 3]:
        for method in ["WBF", "NMS"]:
            subset = df_print[(df_print['k'] == k) & (df_print['Method'] == method)]
            if not subset.empty:
                print(f"\n--- [ k={k} | Method={method} ] ---")
                print(subset.head(5).to_markdown(index=False, floatfmt=".4f"))

def print_table6_correlations():
    """Process and print Table 6: Spearman correlation analysis."""
    csv_path = OUTPUT_DIR / "table6_correlations.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df_all = df[df['Scale'] == 'all']
    
    cols = ['k', 'Method', 'Spearman_vs_GainAP', 'Spearman_vs_GainAR']
    df_print = df_all[cols].copy()
    
    print("\n" + "="*80)
    print(" TABLE 6: Spearman Correlation (Score S vs Actual Gain)")
    print("="*80)
    print(df_print.to_markdown(index=False, floatfmt=".4f"))

def print_table7_selection_regret():
    """Process and print Table 7: Comparative selection strategy performance."""
    csv_path = OUTPUT_DIR / "table7_selection_performance.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path)
    df_all = df[df['Scale'] == 'all']
    
    cols = ['k', 'Method', 'Selector', 'Selected_Combo', 'GainAP', 'RegretAP']
    df_print = df_all[cols].copy()
    
    print("\n" + "="*80)
    print(" TABLE 7: Selection Strategy Performance & Regret AP")
    print("="*80)
    
    for k in [2, 3]:
        for method in ["WBF", "NMS"]:
            subset = df_print[(df_print['k'] == k) & (df_print['Method'] == method)]
            if not subset.empty:
                print(f"\n--- [ k={k} | Method={method} ] ---")
                # Sort by Regret AP (Lower is better)
                subset_sorted = subset.sort_values('RegretAP', ascending=True)
                print(subset_sorted.to_markdown(index=False, floatfmt=".4f"))

def main():
    """Execute evaluation summary sequence."""
    if not OUTPUT_DIR.exists():
        print(f"Directory {OUTPUT_DIR} not found. Ensure outputs are generated in the standard directory.")
        return

    print("Extracting text results for paper formulation...\n")
    print_table4_preensemble()
    print_table5_actual_gains()
    print_table6_correlations()
    print_table7_selection_regret()
    print("\nExtraction complete.")

if __name__ == "__main__":
    main()