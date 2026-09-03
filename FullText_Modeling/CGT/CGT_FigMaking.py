# MAKING FIGURES for COMPARISON PAPER THAT COMPARE HUMAN CODES WITH CLUSTERS
# SAME/SIMILAR FUNCTIONS AS Abstract_Modeling/CGT/DisciplineData/DisseminationFigs/HumanLoopPaper/humanloopfig.py


# The team has chosen Pink as there final/best set 15_0.2_5_3


import pandas as pd 
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
from matplotlib.patches import RegularPolygon
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from matplotlib.legend_handler import HandlerBase
import matplotlib.patches as mpatches


import re

# import plotly.graph_objects as go
# from plotly.colors import sample_colorscale
# import plotly.express as px
# plt.rc('text', usetex=True)



# please note all code related to triangle hatching was heavily suggested by Claude :D, rest of the code is mine

## FOR making the hatched triangles on the legend
class HandlerHatchTriangle(HandlerBase):
    def __init__(self, direction="left", hatch="/////"):
        self.direction = direction
        self.hatch = hatch
        super().__init__()

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Use height as the base so all triangles are the same size
        side = height * 1.2        # slightly larger than the box height
        eq_height = np.sqrt(3) / 2 * side
    
        # Center horizontally and vertically in the legend box
        cx = xdescent + width / 2
        cy = ydescent + height / 2
    
        if self.direction == "left":
            verts = [
                (cx - eq_height / 2,  cy),                  # tip (left)
                (cx + eq_height / 2,  cy + side / 2),       # top right
                (cx + eq_height / 2,  cy - side / 2),       # bottom right
            ]
        else:
            verts = [
                (cx + eq_height / 2,  cy),                  # tip (right)
                (cx - eq_height / 2,  cy + side / 2),       # top left
                (cx - eq_height / 2,  cy - side / 2),       # bottom left
            ]
    
        triangle = Polygon(verts, closed=True,
                           facecolor='white',
                           edgecolor='black',
                           linewidth=1.5,
                           hatch=self.hatch,
                           transform=trans)
        return [triangle]
    


# Convert ["thing1"] to "thing1" and ["thing1" , "thing2"] to Multiple
def simplify_value(value):
    if isinstance(value, list):
        if len(value) == 1:
            return value[0]        # unwrap single-item list to string
        elif len(value) == 0:
            return None            # no match found
        else:
            return "Multiple"      # collapse multi-item list
    return value


def make_centroid_dict(data):
    data_uniq = data["labels"].unique()
    cent_dict = {}

    for val in data_uniq:
        centroid = data[data["labels"] == val]["cent_dimred"].tolist()[:1]
        cent_dict[val] = (centroid[0][0], centroid[0][1]) # this feels slightly convoluted, but also somehow easier than doing a bunch of checks lol
    return cent_dict


# REGULAR PLOT WITH FINAL CLUSTERING AND LABELS
def plot_clusters_wcent(sorteddata_wrep, cent_dimred, num_clust,  savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add=""):
     # Create label-to-color mapping
    cmap = cm.get_cmap('tab20')
    label_to_color = {label: cmap(i % 20) for i, label in enumerate(num_clust)} # this is the random version 
    print(label_to_color)
        # Need to make colors by each dataset since sizes are different - this is getting messy :DDDD


     # Get Subsets of data - pull out representative ones and plot with marker, plot everything else as normal and remove -1 data
    other_data = sorteddata_wrep[sorteddata_wrep["most_rep"]==0]
    other_data_minus1 = other_data[other_data["labels"]==-1]
    other_data_excminus1 = other_data[other_data["labels"]!=-1]
    o_d_m1_col = [label_to_color[row["labels"]] for idx,row in other_data_excminus1.iterrows()]

    rep_data = sorteddata_wrep[sorteddata_wrep["most_rep"]==1]
    # print(rep_data["labels"])
    rep_data = rep_data[rep_data["labels"]!=-1] # don't want rep for noise on plot
    rep_col = [label_to_color[row["labels"]] for idx, row in rep_data.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(other_data_minus1["x"], other_data_minus1["y"], color = "black")
    # plt.scatter(other_data_excminus1["x"], other_data_excminus1["y"], color = "black")
    ax.scatter(other_data_excminus1["x"], other_data_excminus1["y"], c = o_d_m1_col, alpha = other_data_excminus1["probs"], marker = "o", label = "Regular")
    ax.scatter(rep_data["x"], rep_data["y"], c = rep_col, alpha = rep_data["probs"],marker = "*", s = 150, edgecolors = "black", label = "Representative")
    
    if label == True:
        for clust in cent_dimred.keys(): #skip noise ie -1
            if clust != -1:
                ax.annotate(text = clust, 
                            xy = cent_dimred[clust],
                            xytext=(12, 12),
                            textcoords='offset points',
                            # horizontalalignment='center',
                            # verticalalignment='center',
                            size=12, fontweight='bold',
                            color='black', 
                            backgroundcolor='white')
        
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "_wlab_wmostrep_wcolorlab" + savepath[1] + ".png"
    else:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "wmostrep" + savepath[1] + ".png"


    # LEGEND
        # add labels to color legend
    cluster_labs = [ "Developmental Classes / Pathways", "Pedagogical Practices to Promote Equity and Inclusion", "Applied Engineering Educational Approaches", "Faculty Roles for Institutional Change", "TA Training", "Educational Technologies", "Active Learning Strategies in Biology", "Evaluating Active Learning Strategies", "Assessment and Outcomes", "Scaling Evidence-Based Pedagogies"]

    cl_list = [ Line2D( [], [],  color=label_to_color[num], marker='o', ls='',label=str(num)+": "+cluster_labs[num]) for num in label_to_color if num != -1]
    legend1 = ax.legend( handles=cl_list, title="Cluster Labels", edgecolor='black', loc="upper left", bbox_to_anchor=(1.00, 0.95), prop={'size': 13}, title_fontsize=15)
    ax.add_artist(legend1)      
    
    other_mar = Line2D([], [], color='black', marker='o', ls='', label='Regular')
    rep_mar = Line2D([], [], color='black', marker='*', markersize=10, ls='', label='Representative')
    ax.legend( handles=[other_mar, rep_mar], title="Type", edgecolor='black', loc="upper left", bbox_to_anchor=(1.0, 0.44), prop={'size': 13}, title_fontsize=15)

    # Custom Text
    legend_text = legend1.get_texts()[0]
    fontprops = legend_text.get_fontproperties()
    params_text = 'UMAP Parameters' + "\n" + "n_neighbors = " + str(umap_params[0]) + "\n" + "min_dist = " + str(umap_params[1]) + "\n\n" + 'HDBSCAN Parameters' + "\n" + "min_cluster_size = " + str(hdbscanparams[0]) + "\n" + "min_samples = " + str(hdbscanparams[1]) 

    box = {'facecolor': 'none', 'edgecolor': 'black','boxstyle': 'round'}
    # xmin, xmax, ymin, ymax = plt.axis()
    # ax.text(xmax+0.1, (ymax+ymin)/2, params_text, fontsize= "medium", bbox=box)
    ax.text(1.015, 0.05, params_text, fontsize= "large", transform=ax.transAxes,fontproperties=fontprops, bbox=box)
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Unsupervised Clustering of Full Text Data" + "\n" + title_add, fontsize=20)
    
    plt.savefig(savefig, bbox_inches='tight', bbox_extra_artists=[legend1], dpi = 600)
    plt.close()

# REGULAR PLOT WITH FINAL CLUSTERING AND LABELS with copyright
def plot_clusters_wcent_wcopyright(sorteddata_wrep, cent_dimred, num_clust,  savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add=""):
     # Create label-to-color mapping
    cmap = cm.get_cmap('tab20')
    label_to_color = {label: cmap(i % 20) for i, label in enumerate(num_clust)} # this is the random version 
    print(label_to_color)
        # Need to make colors by each dataset since sizes are different - this is getting messy :DDDD


     # Get Subsets of data - pull out representative ones and plot with marker, plot everything else as normal and remove -1 data
    other_data = sorteddata_wrep[sorteddata_wrep["most_rep"]==0]
    other_data_minus1 = other_data[other_data["labels"]==-1]
    other_data_excminus1 = other_data[other_data["labels"]!=-1]
    o_d_m1_col = [label_to_color[row["labels"]] for idx,row in other_data_excminus1.iterrows()]

    rep_data = sorteddata_wrep[sorteddata_wrep["most_rep"]==1]
    # print(rep_data["labels"])
    rep_data = rep_data[rep_data["labels"]!=-1] # don't want rep for noise on plot
    rep_col = [label_to_color[row["labels"]] for idx, row in rep_data.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(other_data_minus1["x"], other_data_minus1["y"], color = "black")
    # plt.scatter(other_data_excminus1["x"], other_data_excminus1["y"], color = "black")
    ax.scatter(other_data_excminus1["x"], other_data_excminus1["y"], c = o_d_m1_col, alpha = other_data_excminus1["probs"], marker = "o", label = "Regular")
    ax.scatter(rep_data["x"], rep_data["y"], c = rep_col, alpha = rep_data["probs"],marker = "*", s = 150, edgecolors = "black", label = "Representative")
    
    if label == True:
        for clust in cent_dimred.keys(): #skip noise ie -1
            if clust != -1:
                ax.annotate(text = clust, 
                            xy = cent_dimred[clust],
                            xytext=(12, 12),
                            textcoords='offset points',
                            # horizontalalignment='center',
                            # verticalalignment='center',
                            size=12, fontweight='bold',
                            color='black', 
                            backgroundcolor='white')
        
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "_wlab_wmostrep_wcolorlab" + savepath[1] + ".png"
    else:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "wmostrep" + savepath[1] + ".png"


    # LEGEND
        # add labels to color legend
    cluster_labs = [ "Developmental Classes / Pathways", "Pedagogical Practices to Promote Equity and Inclusion", "Applied Engineering Educational Approaches", "Faculty Roles for Institutional Change", "TA Training", "Educational Technologies", "Active Learning Strategies in Biology", "Evaluating Active Learning Strategies", "Assessment and Outcomes", "Scaling Evidence-Based Pedagogies"]

    cl_list = [ Line2D( [], [],  color=label_to_color[num], marker='o', ls='',label=str(num)+": "+cluster_labs[num]) for num in label_to_color if num != -1]
    legend1 = ax.legend( handles=cl_list, title="Cluster Labels", edgecolor='black', loc="upper left", bbox_to_anchor=(1.00, 0.95), prop={'size': 13}, title_fontsize=15)
    ax.add_artist(legend1)      
    
    other_mar = Line2D([], [], color='black', marker='o', ls='', label='Regular')
    rep_mar = Line2D([], [], color='black', marker='*', markersize=10, ls='', label='Representative')
    ax.legend( handles=[other_mar, rep_mar], title="Type", edgecolor='black', loc="upper left", bbox_to_anchor=(1.0, 0.44), prop={'size': 13}, title_fontsize=15)

    # Custom Text
    legend_text = legend1.get_texts()[0]
    fontprops = legend_text.get_fontproperties()
    params_text = 'UMAP Parameters' + "\n" + "n_neighbors = " + str(umap_params[0]) + "\n" + "min_dist = " + str(umap_params[1]) + "\n\n" + 'HDBSCAN Parameters' + "\n" + "min_cluster_size = " + str(hdbscanparams[0]) + "\n" + "min_samples = " + str(hdbscanparams[1]) 

    box = {'facecolor': 'none', 'edgecolor': 'black','boxstyle': 'round'}
    # xmin, xmax, ymin, ymax = plt.axis()
    # ax.text(xmax+0.1, (ymax+ymin)/2, params_text, fontsize= "medium", bbox=box)
    ax.text(1.015, 0.05, params_text, fontsize= "large", transform=ax.transAxes,fontproperties=fontprops, bbox=box)

    ax.text(0.5, 0.5, 'Copyright', 
        transform=ax.transAxes,
        fontsize=60, 
        color='gray', 
        alpha=0.3,          # Controls transparency (0 = invisible, 1 = solid)
        ha='center',        # Horizontal alignment
        va='center',        # Vertical alignment
        rotation=30)  
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Unsupervised Clustering of Full Text Data" + "\n" + title_add, fontsize=20)
    
    plt.savefig(savefig, bbox_inches='tight', bbox_extra_artists=[legend1], dpi = 600)
    plt.close()

    

# PLOTTING THE BASE CODE -- COLOR MAP, BASIC LEGEND, NOISE POINTS, ETC, ETC
def plot_basecode(sorteddata_wrep, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add=""):
    fig, ax = plt.subplots(figsize=(10, 8))
     # Create label-to-color mapping
    cmap = cm.get_cmap('tab20')
    label_to_color = {label: cmap(i % 20) for i, label in enumerate(num_clust)} # this is the random version 
    print(label_to_color)
        # Need to make colors by each dataset since sizes are different - this is getting messy :DDDD

     # Get Subsets of data - pull out representative ones and plot with marker, plot everything else as normal and remove -1 data
    other_data = sorteddata_wrep[sorteddata_wrep["most_rep"]==0]
    other_data_minus1 = other_data[other_data["labels"]==-1]
    other_data_excminus1 = other_data[other_data["labels"]!=-1]
    o_d_m1_col = [label_to_color[row["labels"]] for idx,row in other_data_excminus1.iterrows()]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(other_data_minus1["x"], other_data_minus1["y"], color = "black", facecolors='black' )
    
    if label == True:
        for clust in cent_dimred.keys(): #skip noise ie -1
            if clust != -1:
                plt.annotate(text = clust, 
                            xy = cent_dimred[clust],
                            xytext=(12, 12),
                            textcoords='offset points',
                            # horizontalalignment='center',
                            # verticalalignment='center',
                            size=12, fontweight='bold',
                            color='black', 
                            backgroundcolor='white')
     # LEGEND
        # add labels to color legend
    cluster_labs = [ "Developmental Classes / Pathways", "Pedagogical Practices to Promote Equity and Inclusion", "Applied Engineering Educational Approaches", "Faculty Roles for Institutional Change", "TA Training", "Educational Technologies", "Active Learning Strategies in Biology", "Evaluating Active Learning Strategies", "Assessment and Outcomes", "Scaling Evidence-Based Pedagogies"]

    cl_list = [ Line2D( [], [],  color=label_to_color[num], marker='o', ls='',label=str(num)+": "+cluster_labs[num]) for num in label_to_color if num != -1]
    legend1 = ax.legend( handles=cl_list, title="Cluster Labels", edgecolor='black', loc="upper left", bbox_to_anchor=(1.00, 1.00), prop={'size': 12}, title_fontsize=14)
    ax.add_artist(legend1)  

    legend_text = legend1.get_texts()[0]
    fontprops = legend_text.get_fontproperties()
    params_text = 'UMAP Parameters' + "\n" + "n_neighbors = " + str(umap_params[0]) + "\n" + "min_dist = " + str(umap_params[1]) + "\n\n" + 'HDBSCAN Parameters' + "\n" + "min_cluster_size = " + str(hdbscanparams[0]) + "\n" + "min_samples = " + str(hdbscanparams[1]) 
    box = {'facecolor': 'none', 'edgecolor': 'black','boxstyle': 'round'}
    ax.text(1.015, 0.015, params_text, fontsize= "medium", transform=ax.transAxes,fontproperties=fontprops, bbox=box)

    return label_to_color, fig, ax, legend1, other_data_excminus1

# PLOT WITH QUALITY CODES
def plot_qualcodes(sorteddata_wrep, qual_col, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add=""):
    label_to_color, fig, ax, legend1, non_noise_data = plot_basecode(sorteddata_wrep, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add="")
    
    low_points = non_noise_data[(non_noise_data[qual_col] == "Low")]
    med_points = non_noise_data[(non_noise_data[qual_col] == "Medium")]
    high_points = non_noise_data[(non_noise_data[qual_col] == "High")]

    low_col = [label_to_color[row["labels"]] for idx, row in low_points.iterrows()]
    med_col = [label_to_color[row["labels"]] for idx, row in med_points.iterrows()]
    high_col = [label_to_color[row["labels"]] for idx, row in high_points.iterrows()]

    ax.scatter(low_points["x"], low_points["y"], c = low_col, alpha = low_points["probs"], marker = "s", s = 125, edgecolors = "black", label = "Low")
    ax.scatter(med_points["x"], med_points["y"], c = med_col, alpha = med_points["probs"], marker = "P", s = 125, edgecolors = "black", label = "Medium")
    ax.scatter(high_points["x"], high_points["y"], c = high_col, alpha = high_points["probs"], marker = "d", s = 125, edgecolors = "black", linewidth = 1, label = "High")
    
    other_mar = Line2D([], [], color='black', marker='o', ls='', label='Noise Points')
    low_mar = Line2D([], [], color='black', marker='s', markersize=10, ls='', label='Low')
    mid_mar = Line2D([], [], color='black', marker='P', markersize=10, ls='', label='Medium')
    high_mar = Line2D([], [], color='black', marker='d', markersize=10, ls='', label='High')
    ax.legend( handles=[other_mar, low_mar, mid_mar, high_mar], title=str(qual_col) + " Quality Level", edgecolor='black', loc="upper left", bbox_to_anchor=(1.0, 0.49), prop={'size': 13}, title_fontsize=15)

    if label==True:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "_wlab_wmostrep_hc_" + qual_col + ".png"
    else:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "wmostrep_hc" + qual_col + ".png"

    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Unsupervised Clustering of Full Text Data" + "\n" + title_add, fontsize=20)
    
    plt.savefig(savefig, bbox_inches='tight', bbox_extra_artists=[legend1], dpi = 600)
    plt.close()


# PLOTS DISCIPLINE OF JOURNAL
    # there are no double codes in this one
def plot_journaldiscip(sorteddata_wrep, journaldiscip_col, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add=""):
    label_to_color, fig, ax, legend1, non_noise_data = plot_basecode(sorteddata_wrep, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add="")

    ser_points = non_noise_data[(non_noise_data[journaldiscip_col] == "SER")]
    her_points = non_noise_data[(non_noise_data[journaldiscip_col] == "HER")]
    fdr_points = non_noise_data[(non_noise_data[journaldiscip_col] == "FDR")]
    other_points = non_noise_data[(non_noise_data[journaldiscip_col] == "Other")]

    ser_col = [label_to_color[row["labels"]] for idx, row in ser_points.iterrows()]
    her_col = [label_to_color[row["labels"]] for idx, row in her_points.iterrows()]
    fdr_col = [label_to_color[row["labels"]] for idx, row in fdr_points.iterrows()]
    other_col = [label_to_color[row["labels"]] for idx, row in other_points.iterrows()]

    ax.scatter(ser_points["x"], ser_points["y"], c = ser_col, alpha = ser_points["probs"], marker = "s", s = 125, edgecolors = "black", label = "SER")
    ax.scatter(her_points["x"], her_points["y"], c = her_col, alpha = her_points["probs"], marker = "P", s = 125, edgecolors = "black", label = "HER")
    ax.scatter(fdr_points["x"], fdr_points["y"], c = fdr_col, alpha = fdr_points["probs"], marker = "d", s = 125, edgecolors = "black", linewidth = 1, label = "FDR")
    ax.scatter(other_points["x"], other_points["y"], c = other_col, alpha = other_points["probs"], marker = "p", s = 125, edgecolors = "black", linewidth = 1, label = "Other")
    
    other_mar = Line2D([], [], color='black', marker='o', ls='', label='Noise Points')
    ser_mar = Line2D([], [], color='black', marker='s', markersize=10, ls='', label='SER')
    her_mar = Line2D([], [], color='black', marker='P', markersize=10, ls='', label='HER')
    fdr_mar = Line2D([], [], color='black', marker='d', markersize=10, ls='', label='FDR')
    otherjour_mar = Line2D([], [], color='black', marker='p', markersize=10, ls='', label='Other')
    ax.legend( handles=[other_mar, ser_mar, her_mar, fdr_mar, otherjour_mar], title="Discipline of Journal", edgecolor='black', loc="upper left", bbox_to_anchor=(1.0, 0.495), prop={'size': 13}, title_fontsize=15)

    if label==True:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "_wlab_wmostrep_hc_journaldiscip.png"
    else:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "wmostrep_hc_journaldiscip.png"
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Unsupervised Clustering of Full Text Data" + "\n" + title_add, fontsize=20)
    plt.savefig(savefig, bbox_inches='tight', bbox_extra_artists=[legend1], dpi = 600)
    plt.close()

# PLOT WITH TECHNOLOGY RELATED CODES
def plot_techcodes(sorteddata_wrep, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add=""):
    label_to_color, fig, ax, legend1,non_noise_data = plot_basecode(sorteddata_wrep, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add="")

    tech_pts = non_noise_data[(non_noise_data["Technology"] == "Yes") & (non_noise_data["Online Instruction"] == "No")] 
    online_pts = non_noise_data[(non_noise_data["Technology"] == "No") & (non_noise_data["Online Instruction"] == "Yes")] 
    both_pts = non_noise_data[(non_noise_data["Technology"] == "Yes") & (non_noise_data["Online Instruction"] == "Yes")] 
    neither_pts = non_noise_data[(non_noise_data["Technology"] == "No") & (non_noise_data["Online Instruction"] == "No")] 
    techonlineunsure_pts = non_noise_data[(non_noise_data["Technology"] == "Unsure") | (non_noise_data["Technology"].isna() ) | (non_noise_data["Online Instruction"] == "Unsure") | (non_noise_data["Online Instruction"].isna() )] 

    tech_col = [label_to_color[row["labels"]] for idx, row in tech_pts.iterrows()]
    onl_col = [label_to_color[row["labels"]] for idx, row in online_pts.iterrows()]
    both_col = [label_to_color[row["labels"]] for idx, row in both_pts.iterrows()]
    nei_col = [label_to_color[row["labels"]] for idx, row in neither_pts.iterrows()]
    techonlineunsure_col = [label_to_color[row["labels"]] for idx, row in techonlineunsure_pts.iterrows()]
    # onlineunsure_col = [label_to_color[row["labels"]] for idx, row in onlineunsure_pts.iterrows()]

    ax.scatter(neither_pts["x"], neither_pts["y"], c = nei_col, alpha = neither_pts["probs"], marker = "X", s = 100, edgecolors = "black", linewidth = 1, label = "Neither")
    ax.scatter(tech_pts["x"], tech_pts["y"], c = tech_col, alpha = tech_pts["probs"], marker = "<", s = 150, edgecolors = "black", label = "Technology")
    ax.scatter(online_pts["x"], online_pts["y"], c = onl_col, alpha = online_pts["probs"], marker = ">", s = 150, edgecolors = "black", label = "Online Instruction")
    ax.scatter(both_pts["x"], both_pts["y"], c = both_col, alpha = both_pts["probs"], marker = "s", s = 100, edgecolors = "black", linewidth = 1, label = "Tech & Online")
    ax.scatter(techonlineunsure_pts["x"], techonlineunsure_pts["y"], c = techonlineunsure_col, alpha = techonlineunsure_pts["probs"], marker = "d", s = 100, edgecolors = "black", linewidth = 1)
    # ax.scatter(onlineunsure_pts["x"], onlineunsure_pts["y"], c = onlineunsure_col, alpha = onlineunsure_pts["probs"], marker = "8", s = 100, edgecolors = "black", linewidth = 1)

    other_mar = Line2D([], [], color='black', marker='o', ls='', label='Noise Points')
    tech_mar = Line2D([], [], color='black', marker='<', markersize=10, ls='', label='Technology')
    onl_mar = Line2D([], [], color='black', marker='>', markersize=10, ls='', label='Online Instruction')
    to_mar = Line2D([], [], color='black', marker='s', markersize=10, ls='', label='Technology \& Online Instruction')
    nei_mar = Line2D([], [], color='black', marker='X', markersize=10, ls='', label='Neither Technology nor Online Instruction')
    unsure_mar = Line2D([], [], color='black', marker='d', markersize=10, ls='', label='Unsure')
    ax.legend( handles=[other_mar, tech_mar, onl_mar, to_mar, nei_mar, unsure_mar], title="Type", edgecolor='black', loc="upper left", bbox_to_anchor=(1.0, 0.52),  prop={'size': 12}, title_fontsize=14)

    if label==True:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "_wlab_wmostrep_hc_tech.png"
    else:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "wmostrep_hc_tech.png"
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Unsupervised Clustering of Full Text Data" + "\n" + title_add, fontsize=20)
    plt.savefig(savefig, bbox_inches='tight', bbox_extra_artists=[legend1], dpi = 600)
    plt.close()




# Make hatched triangles for plot:D
def add_hatched_triangles(ax, x_vals, y_vals, colors, alpha_vals=None, 
                          direction="left", size=0.08, hatch="///"):
    if alpha_vals is None:
        alpha_vals = [1.0] * len(x_vals)

    patches = []
    rgba_colors = []
    for x, y, color, alpha in zip(x_vals, y_vals, colors, alpha_vals):
        if direction == "left":
            verts = [
                (x - size, y),
                (x + size, y - size),
                (x + size, y + size),
            ]
        else:
            verts = [
                (x + size, y),
                (x - size, y - size),
                (x - size, y + size),
            ]
        patches.append(Polygon(verts, closed=True))
        rgba_colors.append(mcolors.to_rgba(color, alpha=alpha))

    collection = PatchCollection(
        patches,
        match_original=False,   # ← must be False so edgecolors/linewidths apply
        linewidths=1,
        edgecolors="black",     # ← black outline on each triangle
        hatch=hatch,
    )
    collection.set_facecolor(rgba_colors)  # ← per-patch color + alpha
    ax.add_collection(collection)

   
# PLOT WITH FOUR SQUARE CODES
def plot_foursquare(sorteddata_wrep, foursquare_col, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add=""):
    label_to_color, fig, ax, legend1, non_noise_data = plot_basecode(sorteddata_wrep, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add="")

    # dcp_points = sorteddata_wrep[(sorteddata_wrep[foursquare_col] == "Disseminating Curriculum/Pedagogy")]
    dcp_points = non_noise_data[(non_noise_data[foursquare_col] == "Disseminating Curriculum/Pedagogy")]
    dpol_points = non_noise_data[(non_noise_data[foursquare_col] == "Developing Policy")]
    dsv_points = non_noise_data[(non_noise_data[foursquare_col] == "Developing Shared Vision")]
    drt_points = non_noise_data[(non_noise_data[foursquare_col] == "Developing Reflective Teachers")]
    multi_points = non_noise_data[(non_noise_data[foursquare_col] == "Multiple")]
    
    dcp_col = [label_to_color[row["labels"]] for idx, row in dcp_points.iterrows()]
    dpol_col = [label_to_color[row["labels"]] for idx, row in dpol_points.iterrows()]
    dsv_col = [label_to_color[row["labels"]] for idx, row in dsv_points.iterrows()]
    drt_col = [label_to_color[row["labels"]] for idx, row in drt_points.iterrows()]
    multi_col = [label_to_color[row["labels"]] for idx, row in multi_points.iterrows()]

    x_range = sorteddata_wrep["x"].max() - sorteddata_wrep["x"].min()
    y_range = sorteddata_wrep["y"].max() - sorteddata_wrep["y"].min()
    tri_size = min(x_range, y_range) / 80   # tune the divisor to adjust marker size

    # ax.scatter(dcp_points["x"], dcp_points["y"], c = dcp_col, alpha = dcp_points["probs"], marker = "<", s = 100, edgecolors = "black", linewidth = 1, label = "Disseminating Curriculum/Pedagogy")
    add_hatched_triangles(ax, dcp_points["x"], dcp_points["y"], colors=dcp_col, alpha_vals=list(dcp_points["probs"]),  direction="left", size=tri_size, hatch="/////" )
    ax.scatter(dpol_points["x"], dpol_points["y"], c = dpol_col, alpha = dpol_points["probs"], marker = ">", s = 100, edgecolors = "black", linewidth = 1, label = "Developing Policy")
    ax.scatter(dsv_points["x"], dsv_points["y"], c = dsv_col, alpha = dsv_points["probs"], marker = "d", s = 100, edgecolors = "black", linewidth = 1, label = "Developing Shared Vision")    
    ax.scatter(drt_points["x"], drt_points["y"], c = drt_col, alpha = drt_points["probs"], marker = "P", s = 100, linewidth = 1, edgecolors = "black", label = "Developing Reflective Teachers")
    ax.scatter(multi_points["x"], multi_points["y"], c = multi_col, alpha = multi_points["probs"], marker = "s", s = 100, edgecolors = "black", linewidth = 1, label = "Multiple Change Strategies")
    
    
    other_mar = Line2D([], [], color='black', marker='o', ls='', label='Noise Points')
    # dcp_mar = Line2D([], [], color='black', marker='<', markersize=10, ls='', label='Disseminating Curriculum/Pedagogy')
    dcp_mar   = mpatches.Patch(label='Disseminating Curriculum/Pedagogy')   # dummy handle
    dpol_mar = Line2D([], [], color='black', marker='>', markersize=10, ls='', label='Developing Policy')
    dsv_mar = Line2D([], [], color='black', marker='d', markersize=10, ls='', label='Developing Shared Vision')
    drt_mar = Line2D([], [], color='black', marker='P', markersize=10, ls='', label='Developing Reflective Teachers')
    multi_mar = Line2D([], [], color='black', marker='s', markersize=10, ls='', label='Multiple Change Stratgies')
    ax.legend( handles=[other_mar, dcp_mar, dpol_mar, dsv_mar, drt_mar, multi_mar], title="Type", edgecolor='black', loc="upper left", bbox_to_anchor=(1.0, 0.525), prop={'size': 12}, title_fontsize=14, handler_map={dcp_mar: HandlerHatchTriangle(direction="left", hatch="/////") })
    
    if label==True:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "_wlab_wmostrep_hc_foursq.png"
    else:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "wmostrep_hc_foursq.png"
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Unsupervised Clustering of Full Text Data" + "\n" + title_add, fontsize=20)
    plt.savefig(savefig, bbox_inches='tight', bbox_extra_artists=[legend1], dpi = 600)
    plt.close()

# PLOT WITH METHODOLOGY TYPE
def plot_methodology(sorteddata_wrep, method_col, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add=""):
    label_to_color, fig, ax, legend1, non_noise_data = plot_basecode(sorteddata_wrep, cent_dimred, num_clust, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add="")
    # ["Quantitative", "Qualitative", "Qualitative Quantitative", "Mixed Method (capital M)", "Review (Systematic & meta-analysis & metasynthesis", "Absent/ None"]

    quant_points = non_noise_data[(non_noise_data[method_col] == "Quantitative")]
    qual_points = non_noise_data[(non_noise_data[method_col] == "Qualitative")]
    quantqual_points = non_noise_data[(non_noise_data[method_col] == "Qualitative Quantitative")]
    mixed_points = non_noise_data[(non_noise_data[method_col] == "Mixed Method (capital M)")]
    slr_points = non_noise_data[(non_noise_data[method_col] == "Review (Systematic & meta-analysis & metasynthesis")]
    absent_points = non_noise_data[(non_noise_data[method_col] == "Absent/ None")]
    
    quant_col = [label_to_color[row["labels"]] for idx, row in quant_points.iterrows()]
    qual_col = [label_to_color[row["labels"]] for idx, row in qual_points.iterrows()]
    quantqual_col = [label_to_color[row["labels"]] for idx, row in quantqual_points.iterrows()]
    mixed_col = [label_to_color[row["labels"]] for idx, row in mixed_points.iterrows()]
    slr_col = [label_to_color[row["labels"]] for idx, row in slr_points.iterrows()]
    abs_col = [label_to_color[row["labels"]] for idx, row in absent_points.iterrows()] 

    ax.scatter(quant_points["x"], quant_points["y"], c = quant_col, alpha = quant_points["probs"], marker = "<", s = 100, edgecolors = "black", linewidth = 1, label = "Quantitative")
    ax.scatter(qual_points["x"], qual_points["y"], c = qual_col, alpha = qual_points["probs"], marker = ">", s = 100, edgecolors = "black", linewidth = 1, label = "Qualitative")
    ax.scatter(quantqual_points["x"], quantqual_points["y"], c = quantqual_col, alpha = quantqual_points["probs"], marker = "d", s = 100, edgecolors = "black", linewidth = 1, label = "Qualitative & Quantitative")  
    ax.scatter(mixed_points["x"], mixed_points["y"], c = mixed_col, alpha = mixed_points["probs"], marker = "P", s = 100, edgecolors = "black", linewidth = 1, label = "Mixed Method")    
    ax.scatter(slr_points["x"], slr_points["y"], c = slr_col, alpha = slr_points["probs"], marker = "s", s = 100, linewidth = 1, edgecolors = "black", label = "Systematic Review")
    ax.scatter(absent_points["x"], absent_points["y"], c = abs_col, alpha = absent_points["probs"], marker = "X", s = 100, edgecolors = "black", linewidth = 1, label = "Absent/ None")
    
    
    other_mar = Line2D([], [], color='black', marker='o', ls='', label='Noise Points')
    quant_mar = Line2D([], [], color='black', marker='<', markersize=10, ls='', label='Quantitative')
    qual_mar = Line2D([], [], color='black', marker='>', markersize=10, ls='', label='Qualitative')
    quantqual_mar = Line2D([], [], color='black', marker='d', markersize=10, ls='', label='Qualitative & Quantitative')
    mixed_mar = Line2D([], [], color='black', marker='P', markersize=10, ls='', label='Mixed Method')
    slr_mar = Line2D([], [], color='black', marker='s', markersize=10, ls='', label='Systematic Review')
    abs_mar = Line2D([], [], color='black', marker='X', markersize=10, ls='', label='Absent/ None')
    ax.legend( handles=[other_mar, quant_mar, qual_mar, quantqual_mar, mixed_mar, slr_mar, abs_mar], title="Type", edgecolor='black', loc="upper left", bbox_to_anchor=(1.0, 0.53),  prop={'size': 11}, title_fontsize=13)

    if label==True:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "_wlab_wmostrep_hc_method.png"
    else:
        savefig = savepath[0] + "FT_finalclustering" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "wmostrep_hc_method.png"
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Unsupervised Clustering of Full Text Data" + "\n" + title_add, fontsize=20)
    plt.savefig(savefig, bbox_inches='tight', bbox_extra_artists=[legend1], dpi = 600)
    plt.close()
   


def main():
    umap_par = [15,0.2] 
    hdbscan_par = [5, 3] 
    data_path = "/mnt/research/NLP-Lit-Review/bolger/metasyn/data/datafilesforCGT/removed/finalset_Pink/"     # DATA PATH - has cluster data and human codes data
    fig_path = "/mnt/research/NLP-Lit-Review/bolger/metasyn/figures/removed/finalset_Pink/"     # FIGURE PATH

    # Read in cluster data 
    data_file = pd.read_parquet(data_path + "ftremoved_embedding15_0.2_5_3.parquet")
    print(data_file.head())
    num_clust = data_file["labels"].unique().tolist() # number of cluster labels
    num_clust.sort()
    print(num_clust)

    # PLOT OF FINAL CLUSTER SET WITH LABELS ADDED
        # Make centroid dictionary
    cent_dimred = make_centroid_dict(data_file)
        # Plot data with centroids and associated labels in a nice format
    # plot_clusters_wcent(data_file, cent_dimred, num_clust, [ fig_path, "removed"], hdbscan_par, umap_par,  label=True, title_add = "Removed Data")

    # # FINAL CLUSTER PLOT WITH QUALITY CODES
    # plot_qualcodes(data_file, "Relevance", cent_dimred, num_clust, [ fig_path, "removed"], hdbscan_par, umap_par,  label=True, title_add = "Relevance")
    # plot_qualcodes(data_file, "Richness", cent_dimred, num_clust, [ fig_path, "removed"], hdbscan_par, umap_par,  label=True, title_add = "Richness")
    # data_file['Rigor'] = data_file['Rigor/Credibility of empirical studies'].combine_first(data_file['Rigor/Credibility of theory/lit reviews'])
    # plot_qualcodes(data_file, "Rigor", cent_dimred, num_clust, [ fig_path, "removed"], hdbscan_par, umap_par,  label=True, title_add = "Rigor")

    # # # FINAL CLUSTER PLOT WITH DISCIPLINE OF JOURNAL 
    # plot_journaldiscip(data_file, "Discipline of Journal", cent_dimred, num_clust, [ fig_path, "removed"], hdbscan_par, umap_par,  label=True, title_add = "Discipline of Journal")
    
    # # # FINAL CLUSTER PLOT WITH TECH CODES
    # plot_techcodes(data_file, cent_dimred, num_clust, [ fig_path, "removed"], hdbscanparams = [5, 5], umap_params=[15,0.1], label=True, title_add="Technology Related Codes")

    # # # # FINAL CLUSTER PLOT WITH FOUR SQUARE
    # #     # Separate data columns
    # foursq_categories = [ "Disseminating Curriculum/Pedagogy", "Developing Policy", "Developing Shared Vision", "Developing Reflective Teachers", "Other or N/A"]
    # pattern = "|".join(re.escape(c) for c in foursq_categories)
    # extracted = data_file["Change Strategy"].str.findall(f"({pattern})")
    # data_file["Change Strategy Cleaned"] = extracted.apply(simplify_value)    
    # # print(data_file[data_file["labels"] == -1])
    # plot_foursquare(data_file, "Change Strategy Cleaned", cent_dimred, num_clust, [ fig_path, "removed"], hdbscan_par, umap_par,  label=True, title_add = "Change Strategy")


    # # FINAL CLUSTER PLOT WITH METHODOLOGY TYPE
    # no cleaning needed bc these are the only types :D
    # methods_types = ["Quantitative", "Qualitative", "Qualitative Quantitative", "Mixed Method (capital M)", "Review (Systematic & meta-analysis & metasynthesis", "Absent/ None"]
    # ##### mixed methods meant they had to explicitly name mixed methods whereas qual + quant meant they used both types but it wasn't combining/sequencing as they should with proper mixed methods
    # plot_methodology(data_file, "Methodology", cent_dimred, num_clust, [ fig_path, "removed"], hdbscan_par, umap_par,  label=True, title_add = "Methodology")


    # with watermark
    plot_clusters_wcent_wcopyright(data_file, cent_dimred, num_clust, [ fig_path, "removed_watermark"], hdbscan_par, umap_par,  label=True)



                    


  
  





if __name__ == '__main__':
    main()