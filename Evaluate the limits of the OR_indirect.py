import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker  
import os

# Set random seed for reproducibility
np.random.seed(42)

# Set number of simulations (1 million)
N = 1000000

# Randomly sample parameters from uniform distributions
p_x = np.random.uniform(0.0001, 0.4, N)   
p_flu = np.random.uniform(0.0001, 0.4, N) 
ve = np.random.uniform(0.20, 0.80, N)     
theta = np.random.uniform(0.1, 3.0, N)    

# Constraints and data extraction
valid_mask_1 = (theta * p_x) <= 1.0

A = 1 + p_flu * (1 - ve) * (theta - 1)
B = 1 + p_flu * (theta - 1)

p_x_vax = p_x * A
p_x_unvax = p_x * B

valid_mask_2 = (p_x_vax > 0) & (p_x_vax < 1) & (p_x_unvax > 0) & (p_x_unvax < 1)
valid_mask = valid_mask_1 & valid_mask_2

ve_valid = ve[valid_mask]
theta_valid = theta[valid_mask]
p_x_valid = p_x[valid_mask]
p_flu_valid = p_flu[valid_mask] 
p_x_vax_valid = p_x_vax[valid_mask]
p_x_unvax_valid = p_x_unvax[valid_mask]

# Calculate OR_indirect values and output statistics
odds_vax = p_x_vax_valid / (1 - p_x_vax_valid)
odds_unvax = p_x_unvax_valid / (1 - p_x_unvax_valid)
OR_sim = odds_vax / odds_unvax 

summary_stats = {
    "Min": np.min(OR_sim),
    "1st Percentile": np.percentile(OR_sim, 1),
    "2.5th Percentile": np.percentile(OR_sim, 2.5),
    "5th Percentile": np.percentile(OR_sim, 5),
    "Median": np.median(OR_sim),
    "Mean": np.mean(OR_sim),
    "95th Percentile": np.percentile(OR_sim, 95),
    "97.5th Percentile": np.percentile(OR_sim, 97.5),
    "99th Percentile": np.percentile(OR_sim, 99),
    "Max": np.max(OR_sim),
    "Valid Samples": len(OR_sim),
    "Total Samples": N
}

print("=== Simulation OR Value Distribution Statistics ===")
for k, v in summary_stats.items():
    if "Samples" in k:
        print(f"{k}: {v:,}")
    else:
        print(f"{k}: {v:.4f}")

# Load observed OR dataset for various pathogens
file_name = 'Loop_results_include_coinfection_change_match.csv'

# Define mapping between original names and display names
name_map = {
    'Mp': 'MP',
    'HADV': 'HAdV',
    'HMPV': 'HMPV',
    'HCOV': 'HCoV',
    'HRSV': 'RSV'
}

# Target pathogen list using display names
target_pathogens = list(name_map.values())

# Load and clean data
if os.path.exists(file_name):
    try:
        df_pathogen = pd.read_csv(file_name)
        
        # Keep only pathogens present in the mapping keys
        df_pathogen = df_pathogen[df_pathogen['Pathogen'].isin(name_map.keys())].copy()
        
        # Map to display names
        df_pathogen['Pathogen'] = df_pathogen['Pathogen'].map(name_map)
        
        # Convert to categorical type to preserve specific order
        df_pathogen['Pathogen'] = pd.Categorical(
            df_pathogen['Pathogen'], 
            categories=target_pathogens, 
            ordered=True
        )
        
        print(f"Dataset loaded and transformed successfully. Current records: {len(df_pathogen)}")
        
    except Exception as e:
        print(f"Error processing file: {e}")
        df_pathogen = pd.DataFrame() 
else:
    print(f"Error: File {file_name} not found.")
    df_pathogen = pd.DataFrame()

# Data Visualization settings
cmap_choice = 'viridis'
sns.set_theme(style="whitegrid")

# ---- Figure 1: Combined Distribution Plot ----
median_or = np.median(OR_sim)
p025 = np.percentile(OR_sim, 2.5)   
p975 = np.percentile(OR_sim, 97.5)  

plt.figure(figsize=(14, 7))
ax1 = plt.gca()

sns.histplot(OR_sim, bins=150, color='#3498db', stat='density', alpha=0.3, edgecolor=None, ax=ax1)
sns.kdeplot(OR_sim, color='#2980b9', linewidth=2.5, ax=ax1)

ax1.axvline(1.0, color='#e74c3c', linestyle='--', linewidth=1.5)
ax1.axvline(median_or, color='black', linestyle='-', linewidth=2)
ax1.axvline(p025, color='#27ae60', linestyle=':', linewidth=2.5)
ax1.axvline(p975, color='#27ae60', linestyle=':', linewidth=2.5)

xmin, xmax = ax1.get_xlim()
ymin, ymax_orig = ax1.get_ylim()
ymax = max(ymax_orig, 8.5) 

if xmin < p025:
    ax1.axvspan(xmin, p025, color='#F0F0F0', alpha=0.7)
    ax1.text(xmin + (p025 - xmin) / 2, ymax * 0.55, 
             'Rare Event\n(< 2.5th Percentile)', 
             color='#d35400', fontsize=18, ha='center', fontweight='bold')

if xmax > p975:
    ax1.axvspan(p975, xmax, color='#F0F0F0', alpha=0.7)
    ax1.text(p975 + (xmax - p975) / 2, ymax * 0.55, 
             'Rare Event\n(> 97.5th Percentile)', 
             color='#d35400', fontsize=18, ha='center', fontweight='bold')

# Plot pathogen boxplots directly on the main coordinate system
if df_pathogen is not None:
    colors = sns.color_palette("husl", len(target_pathogens))
    
    for i, path in enumerate(target_pathogens):
        path_data = df_pathogen[df_pathogen['Pathogen'] == path]['OR'].dropna().values
        if len(path_data) == 0: continue
        
        # Vertical positions set at Density levels 2, 3, 4, 5, 6
        y_pos = 2 + i 
        
        # Draw horizontal boxplots
        bp = ax1.boxplot(path_data, positions=[y_pos], vert=False, widths=0.2, 
                         patch_artist=True, showfliers=False, whis=(2.5, 97.5), zorder=3)
        
        dark_color = tuple([max(0, c * 0.7) for c in colors[i]])
        
        # Beautify boxplot styles
        for patch in bp['boxes']:
            patch.set_facecolor((1, 1, 1, 0.6))  
            patch.set_edgecolor(colors[i])
            patch.set_linewidth(1.5)
        for median in bp['medians']:
            median.set(color=dark_color, linewidth=2) 
        for whisker in bp['whiskers']:
            whisker.set(color=colors[i], linewidth=1.5)
        for cap in bp['caps']:
            cap.set(color=colors[i], linewidth=1.5)
            
        # Add pathogen name labels just above the boxplots
        x_center = np.median(path_data)
        ax1.text(x_center, y_pos + 0.3, path, ha='center', va='bottom', 
                 fontsize=14, fontweight='bold', color=dark_color,
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

ax1.set_ylim(0, ymax)
ax1.set_xlim(xmin, xmax) 
ax1.yaxis.set_major_locator(ticker.MultipleLocator(1))
ax1.yaxis.set_major_formatter(ticker.ScalarFormatter())

# Custom Legend
custom_lines = [
    Line2D([0], [0], color='#e74c3c', lw=1.5, linestyle='--'),
    Line2D([0], [0], color='black', lw=2),
    Line2D([0], [0], color='#27ae60', lw=2.5, linestyle=':'),
    plt.Rectangle((0, 0), 1, 1, fc='#f39c12', alpha=0.25)
]
new_labels = [
    'OR = 1.0 (No Effect)', 
    f'Sim. Median ({median_or:.3f})', 
    f'2.5th & 97.5th Percentiles', 
    'Rare Event Region ($\leq$2.5% or $\geq$97.5%)'
]
ax1.legend(custom_lines, new_labels, loc='upper right', frameon=True, fontsize=18)

ax1.set_xlabel(r'$\mathit{OR}_{indirect}$', fontsize=22, fontweight='bold')
ax1.set_ylabel('Density', fontsize=22, fontweight='bold')

ax1.tick_params(axis='both', which='major', labelsize=20)

plt.tight_layout()

# ---- Figure 2-4: Combine into a single figure (1 row, 3 columns) ----

# Randomly sample 15,000 points for the scatter plots
sample_idx = np.random.choice(len(OR_sim), min(15000, len(OR_sim)), replace=False)

# Create a 1x3 grid of subplots, sharing the y-axis (sharey=True)
fig, axes = plt.subplots(1, 3, figsize=(24, 7), sharey=True)

# Prepare the data and labels to be mapped onto the three subplots
c_data_list = [ve_valid[sample_idx], p_x_valid[sample_idx], p_flu_valid[sample_idx]]
cbar_labels = ['Vaccine Effectiveness (VE)', 'Pathogen X Prevalence ($P_X$)', 'Influenza Prevalence ($P_{Flu}$)']

for i, ax in enumerate(axes):
    # Plot the scatter plot
    scatter = ax.scatter(theta_valid[sample_idx], OR_sim[sample_idx], 
                         c=c_data_list[i], cmap=cmap_choice, alpha=0.6, s=15)
    
    # Add reference lines
    ax.axhline(1.0, color='#e74c3c', linestyle='--', linewidth=2, label='OR = 1.0')
    ax.axvline(1.0, color='gray', linestyle=':', linewidth=2, label=r'$\theta = 1.0$ (No interaction)')
    
    # Grid, aesthetics, and tick parameters
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=22)
    ax.legend(fontsize=20, loc='upper right')
    
    # Add a horizontal colorbar inside the bottom-left corner of each subplot
    cbaxes = ax.inset_axes([0.03, 0.04, 0.35, 0.03]) 
    cbar = fig.colorbar(scatter, cax=cbaxes, orientation='horizontal')
    
    # Set the colorbar label and adjust its styling
    cbar.set_label(cbar_labels[i], fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=18)
    cbaxes.xaxis.set_ticks_position('top')
    cbaxes.xaxis.set_label_position('top')

# Set the y-axis label only on the leftmost subplot
axes[0].set_ylabel(r'$\mathit{OR}_{indirect}$', fontsize=24, fontweight='bold')

# Set a common x-axis label for the entire figure
fig.supxlabel(r'Interaction Factor ($\theta$)', fontsize=24, fontweight='bold', y=0.02)

# Adjust layout to prevent overlap
plt.tight_layout()
# Fine-tune the bottom margin to ensure enough space for the supxlabel
plt.subplots_adjust(bottom=0.12, wspace=0.05) 

plt.show()