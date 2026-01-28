import os
import glob
import json
import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from omegaconf import OmegaConf 

# --- 1. HELPER FUNCTIONS ---

def get_policy_label(cfg):
    """
    Generates a readable label.
    Requested Mapping:
      - GMMPPO -> RSDR
      - PPO    -> UDR
    """
    policy = cfg.get('policy', 'unknown').lower()
    
    if policy == 'gmmppo':
        # Handle beta, ensure it handles string "-20.0" or int -20 safely
        raw_beta = cfg.get('beta', 0)
        try:
            beta = int(float(raw_beta))
        except:
            beta = raw_beta
        return f"RSDR (β={beta})"
        
    elif policy == 'epoptppo':
        eps = float(cfg.get('epsilon', 0))
        return f"EPOpt (ε={eps})"
        
    elif policy == 'adrppo':
        return "ADR-PPO"
        
    elif policy == 'ppo':
        # User requested PPO -> UDR
        return "UDR"
        
    elif policy == 'udr': 
        return "UDR"
        
    else:
        return policy.upper()

def get_dist_info(cfg):
    """
    Returns (Readable Label, Hyperparameter Name, Hyperparameter Value)
    """
    dist = cfg.get('dist_type', 'unknown')
    
    # 1. Langevin
    if 'langevin' in dist:
        grad = float(cfg.get('drift', 0.0))
        noise = float(cfg.get('random_walk_scale', 0.0))
        return f"Langevin (Noise={noise})", "Gradient Scale", grad
        
    # 2. Jump Diffusion
    elif 'jump' in dist or 'diffusion' in dist:
        diff = float(cfg.get('drift', 0.0))
        prob = float(cfg.get('jump_prob', 0.0))
        return f"Jump Diffusion (Drift={diff})", "Jump Probability", prob
        
    # 3. Beta Distribution
    elif 'beta' in dist:
        scale = float(cfg.get('beta_scale', 0.2))
        return "Beta Distribution", "Beta Scale", scale
        
    # 4. Random Walk
    elif 'walk' in dist or 'random' in dist:
        scale = float(cfg.get('random_walk_scale', 0.05))
        return "Random Walk", "Noise Scale", scale
        
    # 5. Shock
    elif 'shock' in dist:
        prob = float(cfg.get('random_walk_scale', 0.0))
        return "Shock Process", "Shock Probability", prob
        
    else:
        return dist, "Unknown", 0.0

def load_data(root_dir):
    search_pattern = os.path.join(root_dir, "**", "averaged_results", "avg_metrics_*.json")
    files = glob.glob(search_pattern, recursive=True)
    
    data_rows = []
    
    if not files:
        print(f"WARNING: No files found matching {search_pattern}")
        return pd.DataFrame()
    
    print(f"Found {len(files)} result files. Parsing...")
    
    for filepath in files:
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            cfg = data.get('config', {})
            
            policy_label = get_policy_label(cfg)
            dist_category, param_name, param_value = get_dist_info(cfg)
            
            dist_label_full = f"{dist_category} | {param_name}={param_value}"

            row = {
                "Task": cfg.get('task', 'Unknown'),
                "DistCategory": dist_category,
                "ParamName": param_name,
                "ParamValue": param_value,
                "Distribution": dist_label_full,
                "Policy": policy_label,
                
                # Metrics
                "Mean": data.get('reward_mean_avg', np.nan),
                "Mean_Std": data.get('reward_mean_std', 0.0),
                "CVaR10": data.get('reward_cvar10_avg', np.nan),
                "CVaR10_Std": data.get('reward_cvar10_std', 0.0),
                "Min": data.get('reward_min_avg', np.nan),
                "Min_Std": data.get('reward_min_std', 0.0),
            }
            data_rows.append(row)
            
        except Exception as e:
            print(f"Error reading {filepath}: {e}")
            
    return pd.DataFrame(data_rows)

# --- 2. COLOR MAPPING HELPER ---

def get_color_palette(policies):
    color_map = {}
    
    # Specific colors
    c_udr = "tab:green" # UDR is Green
    
    # Light Blue -> Medium Blue -> Dark Blue
    c_pessimistic = ["#99c2ff", "#3385ff", "#0000cc"] 
    
    policies = sorted(list(policies))
    
    # Identify RSDR/Beta policies
    pessimistic_pols = [p for p in policies if "RSDR" in p and "β=-" in p]
    # Sort by beta value descending (-10, -20, -30) -> Maps to Light, Medium, Dark
    pessimistic_pols.sort(key=lambda x: int(x.split('=')[1].replace(')','')), reverse=True) 
    
    udr_pols = [p for p in policies if "UDR" in p]
    
    remaining = [p for p in policies if p not in pessimistic_pols and p not in udr_pols]
    
    # Assign UDR
    for p in udr_pols:
        color_map[p] = c_udr
        
    # Assign RSDR (Blues)
    for i, p in enumerate(pessimistic_pols):
        color_map[p] = c_pessimistic[min(i, len(c_pessimistic)-1)]
        
    # Assign Others (Standard palette)
    standard_palette = sns.color_palette("tab10", n_colors=len(remaining))
    for i, p in enumerate(remaining):
        color_map[p] = standard_palette[i]
        
    return color_map

# --- 3. VALIDATION HELPER ---

def has_full_policy_suite(group_df):
    """
    Checks if the group contains the required policies.
    UPDATED NAMES: UDR, RSDR (β=-10), RSDR (β=-20), RSDR (β=-30)
    """
    required = {"UDR", "RSDR (β=-10)", "RSDR (β=-20)", "RSDR (β=-30)"}
    existing = set(group_df['Policy'].unique())
    
    missing = required - existing
    return len(missing) == 0, missing

# --- 4. LEGEND EXPORT HELPER ---

def export_legend(handles, labels, output_dir, filename="legend.png"):
    """
    Exports a legend as a separate image file in a single horizontal row.
    """
    if not handles:
        return

    # Create a dummy figure to draw the legend
    fig_leg = plt.figure(figsize=(len(labels) * 3, 1)) # Adjust width based on number of items
    ax_leg = fig_leg.add_subplot(111)
    ax_leg.axis('off')

    # Draw legend: ncol=len(labels) forces a single row
    legend = ax_leg.legend(handles, labels, loc='center', ncol=len(labels), 
                           frameon=False, fontsize=22, markerscale=1.5)
    
    save_path = os.path.join(output_dir, filename)
    
    # bbox_inches='tight' removes extra whitespace
    fig_leg.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig_leg)
    print(f"[Saved Legend]: {save_path}")

# --- 5. PLOTTING FUNCTION ---

def plot_and_print(df, output_dir, sweep_config_path):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_style("whitegrid")
    
    # --- Load Sweep Config ---
    try:
        sweep_cfg = OmegaConf.load(sweep_config_path)
        print(f"Loaded sweep config from {sweep_config_path}")
    except Exception as e:
        print(f"Error loading sweep config: {e}. Plotting ALL found data.")
        sweep_cfg = None

    metrics_to_plot = [
        ("Mean", "Episode Reward Mean"),
        ("CVaR10", "Episode Reward CVaR10"),
        ("Min", "Minimum Reward")
    ]

    tasks = df['Task'].unique()
    indices_to_drop = []
    
    # To store handles/labels for the global legend
    global_handles = []
    global_labels = []

    for task in tasks:
        task_df = df[df['Task'] == task]
        dist_cats = task_df['DistCategory'].unique()
        
        for dist_cat in dist_cats:
            scenario_df = task_df[task_df['DistCategory'] == dist_cat].copy()
            
            # --- FILTER X-AXIS BASED ON SWEEP CONFIG ---
            if sweep_cfg:
                allowed_values = []
                if "Random Walk" in dist_cat:
                    allowed_values = sweep_cfg.get('random_walk', {}).get('scales', [])
                elif "Jump" in dist_cat:
                    allowed_values = sweep_cfg.get('jump', {}).get('probs', [])
                elif "Langevin" in dist_cat:
                    allowed_values = sweep_cfg.get('langevin', {}).get('drifts', [])
                
                if allowed_values:
                    scenario_df = scenario_df[
                        scenario_df['ParamValue'].apply(lambda x: any(np.isclose(x, v, atol=1e-6) for v in allowed_values))
                    ]
            
            if scenario_df.empty: continue

            # --- CHECK: Full Suite Validation ---
            is_full, missing_pols = has_full_policy_suite(scenario_df)
            if not is_full:
                print(f"Skipping {task} | {dist_cat}")
                indices_to_drop.extend(scenario_df.index.tolist())
                continue
            
            scenario_df.sort_values(by="ParamValue", inplace=True)
            
            # --- EQUAL INTERVAL LOGIC ---
            all_x_values = sorted(scenario_df['ParamValue'].unique())
            x_map = {val: i for i, val in enumerate(all_x_values)}
            scenario_df['x_idx'] = scenario_df['ParamValue'].map(x_map)

            print(f"PROCESSING: {task} | {dist_cat}")
            
            policies = scenario_df['Policy'].unique()
            color_map = get_color_palette(policies)

            for metric_col, metric_label in metrics_to_plot:
                if metric_col not in scenario_df.columns or scenario_df[metric_col].isna().all():
                    continue

                plt.figure(figsize=(10, 6))
                
                x_label_txt = scenario_df['ParamName'].iloc[0]
                
                for policy in sorted(policies):
                    subset = scenario_df[scenario_df['Policy'] == policy]
                    if subset.empty: continue
                    
                    color = color_map.get(policy, 'black')
                    
                    # --- MARKER LOGIC ---
                    marker = 'D'
                    if policy == "UDR":
                        marker = 'x'
                    
                    # Plot
                    line, = plt.plot(subset['x_idx'], subset[metric_col], 
                            marker=marker, label=policy, color=color, linewidth=2, markersize=8)

                    # Store for Global Legend (only need to do this once per policy)
                    if policy not in global_labels:
                        global_labels.append(policy)
                        global_handles.append(line)

                    # Error Bars
                    std_col = f"{metric_col}_Std"
                    if std_col in subset.columns:
                        plt.fill_between(
                            subset['x_idx'], 
                            subset[metric_col] - subset[std_col], 
                            subset[metric_col] + subset[std_col], 
                            color=color, alpha=0.15
                        )
                
                plt.title(f"{dist_cat}", fontsize=30)
                plt.xlabel(x_label_txt, fontsize=22)
                plt.ylabel(metric_label, fontsize=22)
                plt.xticks(fontsize=20)
                plt.yticks(fontsize=20)
                ax = plt.gca()
                ax.xaxis.get_offset_text().set_fontsize(25)
                
                # --- NO LEGEND IN INDIVIDUAL PLOT ---
                # plt.legend(...)  <-- Removed
                
                plt.xticks(ticks=range(len(all_x_values)), labels=all_x_values)
                plt.tight_layout()
                
                safe_dist_name = dist_cat.replace("(", "").replace(")", "").replace("=", "").replace(" ", "_")
                filename = f"{task}_{safe_dist_name}_{metric_col}.png"
                save_path = os.path.join(output_dir, filename)
                
                plt.savefig(save_path, dpi=300)
                plt.close()

    # --- SAVE SEPARATE LEGEND ---
    # Sort handles/labels to ensure consistent order (e.g., UDR first, then RSDRs)
    if global_handles:
        # Zip, sort by label, unzip
        hl = sorted(zip(global_handles, global_labels), key=lambda x: x[1])
        handles_sorted, labels_sorted = zip(*hl)
        export_legend(handles_sorted, labels_sorted, output_dir, "global_legend_row.png")

    if indices_to_drop:
        df.drop(indices_to_drop, inplace=True)

    return df

# --- 6. MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=str, default="./logs", help="Path to logs directory")
    parser.add_argument("--output", type=str, default="final_leaderboard.csv", help="Output CSV file")
    parser.add_argument("--plot_dir", type=str, default="./plots", help="Directory to save plots")
    parser.add_argument("--sweep_config", type=str, default="sweep_config.yaml", help="Path to sweep config yaml")
    args = parser.parse_args()

    df = load_data(args.logs)
    
    if df.empty:
        print("No data found!")
        return

    df_clean = plot_and_print(df, args.plot_dir, args.sweep_config)
    
    if not df_clean.empty:
        df_clean.sort_values(by=["Task", "Distribution", "Mean"], ascending=[True, True, False], inplace=True)
        df_clean.to_csv(args.output, index=False)
        print(f"\n[Done] Data saved to {args.output}")
    else:
        print("\n[Warning] No complete data available (all filtered out). Check logs above.")

if __name__ == "__main__":
    main()