# MODULES
# Data science
import pandas as pd # Arrays
import numpy as np # Tables
import math

# ML and data processing
from sklearn.decomposition import PCA # Principle Component Analysis
from sklearn.preprocessing import StandardScaler # Data normalization

# Data visualization
import matplotlib.pyplot as plt # Typical plotting
import seaborn as sns # Better plotting

# Others
from IPython.display import display # To show pandas tables
import argparse


# ==================================== 
# INPUT DATA in the CLI
# ==================================== 
parser = argparse.ArgumentParser(description="PCA analysis of RNA-seq data")
parser.add_argument("--input", "-i", type=str, required=True, help="Path to the input TSV file")
args = parser.parse_args()


# ==================================== 
# FILE FORMAT  
# ==================================== 
# Divide the file in single genes, breaking pairs. 
# access to the command line input
df = pd.read_csv(args.input, delimiter="\t")

# 1. Separate the human data
human_cols = [c for c in df.columns if c.endswith("_human") or c == "gene_name"]
df_human = df[human_cols].rename(columns=lambda c: c.replace("_tpm_human", "").replace("_id_human", "_id"))
df_human["species"] = "human"

# 2. Separate the mouse data
mouse_cols = [c for c in df.columns if c.endswith("_mouse") or c == "gene_name"]
df_mouse = df[mouse_cols].rename(columns=lambda c: c.replace("_tpm_mouse", "").replace("_id_mouse", "_id"))
df_mouse["species"] = "mouse"

# 3. Combine them into one long-format DataFrame and rearrange columns
df_long = pd.concat([df_human, df_mouse], ignore_index=True)
id_cols = ["gene_id", "gene_name", "species"]
tissue_cols = [c for c in df_long.columns if c not in id_cols]

df_final = df_long[id_cols + tissue_cols]
all_genes_df = df_final



# ==================================== 
# PCA
# ==================================== 
id_cols = ["gene_id", "gene_name", "species"]
features = [c for c in all_genes_df.columns if c not in id_cols]

# Log-transform, drop NaNs & Standardize
df_pca_clean = all_genes_df.copy()
df_pca_clean[features] = np.log2(df_pca_clean[features] + 1)
df_pca_clean = df_pca_clean.dropna().copy()
x = StandardScaler().fit_transform(df_pca_clean[features])

# Run PCA
pca = PCA(n_components=2)
components = pca.fit_transform(x)

# Extract percentage of variance explained by each PC
var_explained = pca.explained_variance_ratio_ * 100
pc1_var = var_explained[0]
pc2_var = var_explained[1]

pc1_col = f"PC1 ({pc1_var:.2f}%)"
pc2_col = f"PC2 ({pc2_var:.2f}%)"

pca_res = pd.DataFrame(data=components, columns=[pc1_col, pc2_col])
for col in all_genes_df.columns:
    pca_res[col] = df_pca_clean[col].values


# ====================================
# PLOT PCA
# ====================================
dot_size = 38

def plot_combined_pca_megaplot(pca_res, tissue_features, pc1_label, pc2_label, n_cols=2):
    n_panels = len(tissue_features) + 1
    n_rows = math.ceil(n_panels / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 3, n_rows * 2),
        sharex=True,
        sharey=True,
        squeeze=False
    )

    axes = axes.flatten()

    sns.scatterplot(
        data=pca_res,
        x=pc1_label,
        y=pc2_label,
        hue="species",
        palette="Set1",
        alpha=0.8,
        edgecolor="white",
        linewidth=0.3,
        s=dot_size,
        legend=False,
        ax=axes[0]
    )

    axes[0].set_title("species", fontsize=14)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(labelsize=14)

    vmin = pca_res[tissue_features].min().min()
    vmax = pca_res[tissue_features].max().max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    for i, color_var in enumerate(tissue_features, start=1):
        ax = axes[i]

        sc = ax.scatter(
            pca_res[pc1_label],
            pca_res[pc2_label],
            c=pca_res[color_var],
            cmap="viridis",
            norm=norm,
            alpha=0.8,
            edgecolors="white",
            linewidth=0.2,
            s=dot_size
        )

        ax.set_title(color_var, fontsize=14)
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.tick_params(labelsize=14)

    for j in range(n_panels, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle("PCA colored by species and tissue expression", fontsize=18)
    fig.supxlabel(pc1_label, fontsize=14)
    fig.supylabel(pc2_label, fontsize=14)

    fig.subplots_adjust(
        left=0.17,
        right=0.82,
        bottom=0.12,
        top=0.88,
        wspace=0.12,
        hspace=0.35
    )

    cbar_ax = fig.add_axes([0.90, 0.18, 0.025, 0.65])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Expression", fontsize=14, labelpad=10)
    cbar.ax.tick_params(labelsize=14)

    plt.savefig("PCA_combined_megaplot.png", dpi=300)
    plt.show()


plot_combined_pca_megaplot(pca_res, features, pc1_col, pc2_col, n_cols=2)

# Save results (TSV and plot)
pca_res.to_csv("PCA_results.tsv", sep="\t", index=False)



# ====================================
# PCA LOADINGS
# ====================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ====================================================== #
# PCA BIPLOT (GENE EXPRESSION SPACE WITH TISSUE VECTORS) #
# ====================================================== #

dot_size = 38

pc1_label = pca_res.columns[0]
pc2_label = pca_res.columns[1]

loadings = pd.DataFrame(
    pca.components_.T,
    columns=[pc1_label, pc2_label],
    index=features
)

print("PCA Loadings (weights per tissue):")
display(loadings)

fig, ax = plt.subplots(figsize=(4.4, 3.6))

# Gene points without species coloring
ax.scatter(
    pca_res[pc1_label],
    pca_res[pc2_label],
    alpha=0.3,
    edgecolors="white",
    linewidth=0.3,
    color="gray",
    s=dot_size
)

scale_factor = 3.5
colors = plt.cm.Set1(np.linspace(0, 1, len(loadings.index)))
colors = ["orange", "magenta", "green", "blue", "red",]

legend_handles = []

for color, tissue in zip(colors, loadings.index):
    x_arrow = loadings.loc[tissue, pc1_label] * scale_factor
    y_arrow = loadings.loc[tissue, pc2_label] * scale_factor

    ax.arrow(
        0,
        0,
        x_arrow,
        y_arrow,
        color=color,
        alpha=1,
        head_width=0.12,
        linewidth=2.5,
        length_includes_head=True
    )

    legend_handles.append(
        Line2D([0], [0], color=color, lw=2, label=tissue)
    )

ax.set_title("PCA biplot", fontsize=14)
ax.set_xlabel(pc1_label, fontsize=12)
ax.set_ylabel(pc2_label, fontsize=12)
ax.tick_params(labelsize=11)

ax.axvline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)
ax.axhline(0, color="black", linestyle="--", linewidth=0.5, alpha=0.5)

ax.legend(
    handles=legend_handles,
    title="Tissue",
    fontsize=10,
    title_fontsize=10,
    loc="best",
    frameon=True,
)

plt.tight_layout()
plt.savefig("PCA_biplot.png", dpi=300)
plt.show()



# ====================================
# FEATURE DISTRIBUTIONS
# ====================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

# EXPRESSION DISTRIBUTION BY TISSUE AND SPECIES         
id_cols = ["gene_id", "gene_name", "species"]
tissue_features = [c for c in all_genes_df.columns if c not in id_cols]

plot_df = all_genes_df.melt(
    id_vars=id_cols,
    value_vars=tissue_features,
    var_name="tissue",
    value_name="expression"
)

species_colors = {
    "human": "blue",
    "mouse": "red"
}

hue_order = ["human", "mouse"]

fig, ax = plt.subplots(figsize=(5.6, 3.9))

sns.violinplot(
    data=plot_df,
    x="tissue",
    y="expression",
    hue="species",
    order=tissue_features,
    hue_order=hue_order,
    palette=species_colors,
    split=True,
    inner=None,
    linewidth=1.1,
    edgecolor="black",
    alpha=0.9,
    ax=ax
)

# MARK MEAN OF EACH DISTRIBUTION                         

mean_df = (
    plot_df
    .groupby(["tissue", "species"], as_index=False)["expression"]
    .mean()
)

species_offsets = {
    "human": -0.18,
    "mouse": 0.18
}

mean_line_half_width = 0.11

for tissue_index, tissue in enumerate(tissue_features):
    for species in hue_order:
        mean_value = mean_df.loc[
            (mean_df["tissue"] == tissue) &
            (mean_df["species"] == species),
            "expression"
        ]

        if mean_value.empty:
            continue

        x_center = tissue_index + species_offsets[species]

       
        # White mean line
        ax.hlines(
            y=mean_value.iloc[0],
            xmin=x_center - mean_line_half_width,
            xmax=x_center + mean_line_half_width,
            color="white",
            linewidth=1,
            edgecolor="black",
            zorder=11
        )

# FORMAT                                                
ax.set_title("Expression distribution by tissue and species", fontsize=18)
ax.set_xlabel("Tissue", fontsize=14)
ax.set_ylabel("Expression value", fontsize=14)

ax.tick_params(axis="x", labelsize=14, rotation=25)
ax.tick_params(axis="y", labelsize=14)

ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.5, alpha=0.7)
ax.grid(True, which="major", axis="x", linestyle="--", linewidth=0.5, alpha=0.7)

for spine in ax.spines.values():
    spine.set_visible(True)

legend_handles = [
    Line2D([], [], linestyle="None", label="human"),
    Line2D([], [], linestyle="None", label="mouse")
]

legend = ax.legend(
    handles=legend_handles,
    title="Species",
    fontsize=12,
    title_fontsize=12,
    loc="best",
    frameon=False,
    handlelength=0,
    handletextpad=0
)

for text, color in zip(legend.get_texts(), [species_colors["human"], species_colors["mouse"]]):
    text.set_color(color)

plt.tight_layout()
plt.savefig("expression_distribution_by_tissue_species.png", dpi=300)
plt.show()