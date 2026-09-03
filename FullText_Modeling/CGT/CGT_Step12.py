# implement relative validity and silohouette score for comparison of clustering outputs 

#### !!!! TODO 
## set up the parameter testing - run together should only include the functions you need for parameter testing. rest of the code can go in another function 

######## Libraries #########
import argparse
import time
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import colors as mcolors
import umap
import hdbscan
from collections import Counter
import contractions # https://github.com/kootenpv/contractions
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import yake
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')


######## Code #########

def run_umap(data, embedcol, num_dim, umap_params=[15,0.1]): 
    print("Running UMAP")
    # transform embedding column (each value a list) to matrix of values
         # dim (num of entries x dim of embedding (768))
    embedded_vec = pd.DataFrame(data=[data.loc[i,embedcol] for i in range(len(data))], index=None, columns=None)
    # print(embedded_vec.head())

    # UMAP Parameters - default in function are default in function
    n_neighbors, min_dist = umap_params[0], umap_params[1] 
            # size of the local neighborhood, how tightly UMAP is allowed to pack points together

    reducer = umap.UMAP(random_state=3317, n_components=num_dim, n_neighbors = n_neighbors, min_dist = min_dist)
    dimred_data = reducer.fit_transform(embedded_vec)
    # print(dimred_data.shape) # puts in 2D space
    print(dimred_data[0:5,])

    # Add dimension reduced data as column in original dataset
    data["x"] = dimred_data.T[0]
    data["y"] = dimred_data.T[1]
    
    return data


def run_hdbscan(data, hdbscan_params = [5, 5]):
    print("Running HDSCAN")
    # Reshape data from columsn in data set to 2D numpy array 
        # This is the same as dimred_data in `run_UMAP`, but it felt easier to just reshape here than track as return value throughout
    dimred_data = data[['x','y']].to_numpy()

    # Parameters
    min_cluster_size = hdbscan_params[0]
    min_samples = hdbscan_params[1]

    # Run HDBSCAN
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, gen_min_span_tree=True)
    clusterer.fit(dimred_data)

    # Add cluster labels and probabilites to dataframe
    data["labels"] = clusterer.labels_
    data["probs"] = clusterer.probabilities_

    # Relative Validity
    rv = clusterer.relative_validity_

    return data, rv

##### CLUSTERING FUNCTIONS ######
def get_cluster_df(df, cluster_number):
    return df[df["labels"] == cluster_number].reset_index()

def calc_centroids(sorted_data, num_clust, embed_col):
    centriod_labs_dimred = {} # centroid for dim reduced data
    centriod_labs_fullembeds = {} # centroids for full embeddings

    for clust in num_clust: # for each cluster
        cent_data = get_cluster_df(sorted_data, clust) # get the cluster specific data
        centriod_labs_dimred[clust] = (np.mean(cent_data["x"]), np.mean(cent_data["y"]))
        centriod_labs_fullembeds[clust] = np.mean(cent_data[embed_col])
    
    return(centriod_labs_dimred, centriod_labs_fullembeds)

# gets most representative abstract in each cluster, ie shortest euclidean distance between the centroid of cluster and each point in cluster
def get_mostrepresent_abst(sorted_data, centriod_dict, num_clust): 
    sorted_data["most_rep"] = 0 # add new column in dataframe
    print("IN MOST REP")

    for clust in num_clust:
        cent_data = get_cluster_df(sorted_data, clust) # get data per cluster
        centroid_dimred = centriod_dict[clust] # get centroid for cluster the dim reduced one
        print("The centroid for Cluster", clust,"is: ", centroid_dimred)
        
        # Calculate dictionary of original point index and distances to centroid
        dist_to_centr = {}
        for idx in range(len(cent_data)): # for other points in cluster
             # calc Euclid distance bewteen centroid of cluster and all other points in cluster
            tempx = centroid_dimred[0]-cent_data.loc[idx, "x"]
            tempy = centroid_dimred[1]-cent_data.loc[idx, "y"]
            dist_to_centr[idx] = np.sqrt(tempx**2 + tempy**2) 

        # Get minimum distance
        minval = min(dist_to_centr.values())
        # Find associated key
        minval_idx = [key for key, val in dist_to_centr.items() if val==minval]
        og_idx = cent_data.loc[minval_idx, "index"]

        sorted_data.loc[og_idx, "most_rep"] = 1
    print("Number of Representative is Same as Number of Clusters:", sum(sorted_data["most_rep"]) == len(num_clust))
    return sorted_data

# Clean full text string
def clean_str(ft_string, rem_list):
    stop_words = list(stopwords.words('english'))
    remove_list = rem_list + stop_words

    # not perfect, but gets the job done pretty well
    ft_string = ft_string.replace("-\n", "").replace("\n-",""). replace("\n"," ")  
    ft_string = ft_string.replace("-", " ")
    ft_string = ft_string.lower()
    pattern = r'\b[a-zA-Z]\b|\b(?:' + '|'.join(map(re.escape, remove_list)) + r')\b|\d+|[^\w\s]|\t|\r' 
            # removes all single letters, digits, chars, etc
    clean_str = re.sub(pattern, '', ft_string, flags=re.IGNORECASE)
    
    clean_str = " ".join(clean_str.split())
    clean_list = clean_str.split()
    return clean_str, clean_list

# GET most frequent words for cluster, USING the full text data 
def get_freq_values(clean_str, clean_list):
    # Top 10 most frequent words in cluster
    word_freq = Counter(clean_list).most_common()[:10]
    topten_words = [f for f, s in word_freq]
    topten_freq = [s for f, s in word_freq]
    # print(topten_words, topten_freq)
    
    # Calculate YAKE keyword identifier while I'm here - https://www.markovml.com/blog/yake-keyword-extraction, https://liaad.github.io/yake/
    keyword_extr = yake.KeywordExtractor(top=10, n=2, stopwords=None) # leaving rest parameters as default
    kw_ext = keyword_extr.extract_keywords(clean_str) # lower score = higher importance
    yake_words = [kw for kw, score in kw_ext]
    yake_values = [score for kw, score in kw_ext]
    # print(yake_words, yake_values)

    return topten_words, topten_freq, yake_words, yake_values


# GET combined data for full text data within a specific cluster
def combine_cluster_data(sorted_data, num_clust, rem_list, ft_file):
    cluster_data = pd.DataFrame(index = range(len(num_clust)-1), columns = ["clust_num", "top_10_words", "top_10_values", "top10_yake", "top10_yakevalues"])
    
    
    # Read in json file with full text
    with open(ft_file, 'r') as f:
        ft_data = json.load(f) # list of dictionaries

    combined_ft_list = [] # for tdidf
    # Get data by cluster
    for clust in num_clust:
        if clust == -1:
            continue
        clust_data = sorted_data[sorted_data["labels"] == clust].reset_index() 

        combined_ft = ""
        # For item_id in specific cluster, get the specific full text
        for id_idx in range(len(clust_data)):
            curr_id = clust_data.loc[id_idx, "item_id"] # get current id for data in cluster
            for entry in ft_data: # get the ft data
                if entry["item_id"] == curr_id:
                     curr_ft = entry["text"]
        
            # Keep adding individual fulle texts to combined string
            combined_ft += curr_ft
        
        combined_ft_list.append(combined_ft)
        # Do data cleaning
        clean_data_str, clean_data_list = clean_str(combined_ft, rem_list)

        # Call to function that calculates top ten, tfidf, and yake
        topten_words, topten_freq, yake_words, yake_freq = get_freq_values(clean_data_str, clean_data_list)
        cluster_data.loc[clust,"top_10_words"] = topten_words
        cluster_data.loc[clust, "top_10_values"] = topten_freq
        cluster_data.loc[clust, "top10_yake"] = yake_words
        cluster_data.loc[clust, "top10_yakevalues"] = yake_freq
    cluster_data["clust_num"] = [ n for n in num_clust if n != -1]
        
    # Calculate TF-IDF data
        # We calculate a TF-IDF for the the whole cluster. IE concat all documents from one cluster to a single document (like we do for the top 10 words), then calculate TF-IDF as usual the cluster of documents turned single document - https://arxiv.org/pdf/2203.05794v1
    # We want the top 10 highest Tf-IDF scores for each document 
    tf_vect = TfidfVectorizer()
    combined_ft_pd = pd.Series(combined_ft_list)
    tf_matrix = tf_vect.fit_transform(combined_ft_pd).toarray()      # get tf-idf matrix
        # give it column with each entry as combined cleaned data (type pandas series), shape (numcolumns, xxx )
    get_words = tf_vect.get_feature_names_out() # each words refers to a column of tf_matrix

    tf_df = pd.DataFrame(columns=["top10_tfidf", "top10_tfidfvalues"], index=range(tf_matrix.shape[0]) )
    for doc_i in range(tf_matrix.shape[0]): # tf_matrix maintans document order
        doc = tf_matrix[doc_i, ]
        top10_indices = np.argsort(doc)[::-1][:10]
        tf_df.loc[doc_i, "top10_tfidf"] =  get_words[top10_indices].tolist()
        tf_df.loc[doc_i, "top10_tfidfvalues"] = doc[top10_indices].tolist()

    cluster_data = pd.concat([cluster_data, tf_df], axis=1)
    return cluster_data  

        
# get the papers with the top 5 probabilities in the cluster, somewhat arbitrary
    # Should be static bc embeddings are static and way pandas orders should be static
def get_top5_probability(sorted_data, num_clust, umap_params=[15,0.1],  hdbscanparams = [5, 5]): 
    print("Getting top 5")

    # Get Cluster Data
    combined_top5_cols = list(sorted_data.columns)
    combined_top5 = pd.DataFrame(columns = combined_top5_cols)

    for clust in num_clust:
        clust_df = get_cluster_df(sorted_data, clust) # get data per cluster
        # for clust_df, sort by highest probability 
        clust_df_sort = clust_df.sort_values(by=['probs'], ascending = False)

        # extract top 5
            # note there can be more than 5 with prob = 1 (and the ordering is arbitrary, I am just picking 5)
        clust_df_top5 = clust_df_sort.iloc[0:5, :]

        # Combine back to one dataframe 
        combined_top5 = pd.concat([combined_top5, clust_df_top5])
    return combined_top5
  

##### PLOTTING FUNCTIONS ######
## Just plotting data after UMAP

def just_plot_postUMAP(data, dim1, dim2, savepath, umap_params=[15,0.1], title_add = ""):
      # Abstract embedded data (no replicates)
    plt.scatter(data[dim1], data[dim2], color="black")
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    
    params_text = "n_neighbors = " + str(umap_params[0]) + "\n" + "min_dist = " + str(umap_params[1])
    box = {'facecolor': 'none', 'edgecolor': 'black','boxstyle': 'round'}
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(xmax-1, ymax+1, params_text, bbox=box)
    plt.title("Vectorized Abstract Data Reduced to Two Dimensions" + "\n"+ title_add)
    plt.savefig(savepath[0] + "ft_embedded" + dim1 + dim2 + "_" + str(umap_params[0]) + "_" + str(umap_params[1]) + savepath[1] +".png", bbox_inches='tight')
    plt.close()



    
## Plot without final labels - full color, representative point, etc
def plot_clusters_wcent(sorteddata_wrepresentative, cent_dimred, savepath, hdbscanparams = [5, 5], umap_params=[15,0.1], title_add=""):
     # Get Subsets of data - pull out representative ones and plot with marker, plot everything else as normal and remove -1 data
    other_data = sorteddata_wrepresentative[sorteddata_wrepresentative["most_rep"]==0]
    other_data_minus1 = other_data[other_data["labels"]==-1]
    other_data_excminus1 = other_data[other_data["labels"]!=-1]

    rep_data = sorteddata_wrepresentative[sorteddata_wrepresentative["most_rep"]==1]
    # print(rep_data["labels"])
    rep_data = rep_data[rep_data["labels"]!=-1] # don't want rep for noise on plot

    fig, ax = plt.subplots(1,1)
    plt.figure(figsize=(10,8))
    plt.scatter(other_data_minus1["x"], other_data_minus1["y"], color = "black") # plot noise points
    plt.scatter(other_data_excminus1["x"], other_data_excminus1["y"], c = other_data_excminus1["labels"], alpha = other_data_excminus1["probs"], cmap = "tab20", marker = "o", label = "Regular") # plot cluster points
    plt.scatter(rep_data["x"], rep_data["y"], c = rep_data["labels"], alpha = rep_data["probs"], cmap = "tab20", marker = "*", s = 150, edgecolors = "black", label = "Representative") # plot rep points
  
    for clust in cent_dimred.keys(): #skip noise ie -1
        if clust != -1:
            plt.annotate(text = clust, 
                        xy = cent_dimred[clust],
                        xytext=(8, 8),
                        textcoords='offset points',
                        size=7, weight='bold',
                        color='black', 
                        backgroundcolor='white')
    savefig = savepath[0] + "ft_hbscan_" + str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscanparams[0]) + "_" + str(hdbscanparams[1]) + "_wlab_wmostrep" + savepath[1] + ".png"

    params_text = 'UMAP Parameters' + "\n" + "n_neighbors = " + str(umap_params[0]) + "\n" + "min_dist = " + str(umap_params[1]) + "\n\n" + 'HDBSCAN Parameters' + "\n" + "min_cluster_size = " + str(hdbscanparams[0]) + "\n" + "min_samples = " + str(hdbscanparams[1])   
    
    box = {'facecolor': 'none', 'edgecolor': 'black','boxstyle': 'round'}
    xmin, xmax, ymin, ymax = plt.axis()
    plt.text(xmax+0.1, (ymax+ymin)/2, params_text, fontsize= "medium", bbox=box)
    
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.title("Unsupervised Clustering of Full Text Data with Centroids" + "\n" + title_add)
    other_mar = Line2D([], [], color='black', marker='o', ls='', label='Regular')
    rep_mar = Line2D([], [], color='black', marker='*', markersize=10, ls='', label='Representative')
    plt.legend(handles=[other_mar, rep_mar])
    plt.savefig(savefig, bbox_inches='tight')
    plt.close()
    
    
#! title add
# Runs UMAP, HDBSCAN, Plotting, and output files for a specific dataset 
def run_all(data, embedcol, num_dim, umap_params, hdbscan_params, ft_file, files_directory, save_files_directory, save_fig_directory, which_type, param_search, plot_title):
    # pd.set_option('display.max_columns', None)
    # Run UMAP & HBSCAN 
    data_postumap = run_umap(data, embedcol, num_dim, umap_params)
    data_posthdbscan, rel_val = run_hdbscan(data_postumap, hdbscan_params)

    # CALCULATE Num Clusters
    num_clust = data_posthdbscan["labels"].unique().tolist() # number of cluster labels
    num_clust.sort()
    data_sorted = data_posthdbscan.sort_values(by=['labels'])
    print(data_sorted.head())

    ### DO Calculations on the data 
    # Calculate the centriods # use dim red for plotting, use embed for other calculations in step 2
    cent_dimred, cent_fullembed = calc_centroids(data_sorted, num_clust, embedcol)
    cent_df = pd.DataFrame(list(cent_dimred.items()), columns=["clust_num", "cent_dimred"])
    cent_fullembed_list = list(cent_fullembed.values())
    cent_df["cent_fullembed"] = cent_fullembed_list
        # merge in centroids
    data_sorted = pd.merge(data_sorted, cent_df, left_on='labels', right_on='clust_num', how='outer')
    data_sorted = data_sorted.drop(columns='clust_num')
    # print(data_sorted.head())
    # print(data_sorted.tail())
    

     # Get most representative abstract with full embed centroids
    data_sorted_wrepres = get_mostrepresent_abst(data_sorted, cent_dimred, num_clust)
    # print(data_sorted_wrepres.head())

    # Get top 10 words for each cluster
    removal_list = ["student", "students", "university", "college", "education", "teach", "teaching", "undergraduate", "course", "learning", "learn", "research", "design", "study", "practice", "et", "al", "also", "additionally", "further", "furthermore"]        
    top_words = combine_cluster_data(data_sorted_wrepres, num_clust, removal_list, ft_file)

    # Combine embed_df and top_df
    embedtop_df = pd.merge(data_sorted_wrepres, top_words, left_on='labels', right_on='clust_num', how='outer')
    embedtop_df = embedtop_df.drop(columns='clust_num')
    
    # Add in the bib_df
    bib_df = pd.read_pickle(files_directory + 'hc_bib_metadata.pkl')
    complete_df = pd.merge(embedtop_df, bib_df, left_on='item_id', right_on='Item ID', how='left')
    complete_df = complete_df.drop(columns='Item ID')
    # print(complete_df.tail())
    
    complete_df = complete_df[["item_id", "filename", "Short Title", "Title", "abv_author", "full_author", "title", "abstract", "year", "journal", "Discipline of Journal", "First Author's Affiliation", "Funding Mechanism", "Institution Type", "Primary Stated Purpose", "Change Strategy", "Target of Change", "Agent of Change", "Online Instruction", "COVID", "Technology", "DEI","Adjunct Faculty", "Relevance", "Richness","Rigor/Credibility of empirical studies","Rigor/Credibility of theory/lit reviews","Success Claimed around systemic change", "Paper Type", "Methodology", embedcol, "x", "y", "labels", "probs", "cent_dimred","cent_fullembed", "most_rep", "top_10_words", "top10_yake", "top10_tfidf", "top_10_values","top10_yakevalues", "top10_tfidfvalues", "title_lower"]]
    
    # Write to separate df, has all original columns + most_rep + centroids (TOTAL from other py file) + top words (repeated for each row
        # This is full dataset
    param_path =  str(umap_params[0]) + "_" + str(umap_params[1]) + "_" +str(hdbscan_params[0]) + "_" + str(hdbscan_params[1])
    complete_df.to_csv(save_files_directory + embedcol + param_path + ".csv")
    complete_df.to_parquet(save_files_directory  + embedcol + param_path + '.parquet')


     # Get top 5 papers by highest probability in the cluster 
        # Write to separate df, same info as full dataset, just top 5 from each cluster + most_rep + top_words + centroids
    complete_top5 = get_top5_probability(complete_df, num_clust, umap_params, hdbscan_params) 
    complete_top5.to_csv(save_files_directory + embedcol + "_top5_" + param_path + ".csv")
    complete_top5.to_parquet(save_files_directory + embedcol + "_top5_" + param_path + '.parquet')

     # Make the plot 
    plot_clusters_wcent(complete_df, cent_dimred, [ save_fig_directory, which_type], hdbscanparams = hdbscan_params, umap_params=umap_params, title_add=plot_title)
    
    if param_search == False:
        return None 
    elif param_search == True:
        return complete_df, rel_val


def main(args):
    ######## Directories #########
    files_dir = "/mnt/research/NLP-Lit-Review/bolger/metasyn/data/datafilesforCGT/"
    figs_files_dir = "/mnt/research/NLP-Lit-Review/bolger/metasyn/figures/"

    # GET DATA 
    # just use this if to get the data 
    if args.data == "ft_complete":
        print("--------- USING COMPLETE DATA ---------------")
        # Load data
        data = pd.read_parquet(files_dir + 'complete/embedding_files/ft_embedding.parquet')
        file = files_dir + 'complete/singlestr_FinalIncludedArticles_fulltext.json' # main ft file
        save_data_dir = files_dir + "complete/" 
        save_figures_dir = figs_files_dir + 'complete/'
        embed_col = "ft_embedding"
        plot_title_add = "Complete Data"
        umap_path_title = "complete"
        
        
        print(data.head())

    if args.data == "ft_removed":
        print("--------- USING REMOVED DATA ---------------")
        # KEEP VARIABLE NAMES, CHANGE FILE DIRECTORIES
       # Copy from above
        data = pd.read_parquet(files_dir + 'removed/embedding_files/ftremoved_embedding.parquet')
        file = files_dir + 'removed/remove_singlestr_FinalIncludedArticles_fulltext.json' # main ft file
        save_data_dir = files_dir + "removed/" 
        save_figures_dir = figs_files_dir + 'removed/'
        embed_col = "ftremoved_embedding"
        plot_title_add = "Removed Data"
        umap_path_title = "removed"
        
        print(data.head())




    if args.type == "single":
        print("--------- RUNNING SINGLE MODEL---------------")
     # PARAMETERS
        umap_par = [10,0.1] #default is set in function, also [15,0.1]
        hdbscan_par = [10,5] #default is set in function, also [5,5]
         # Run umap, hdbscan, metrics, plot
        run_all(data, embed_col, 2, umap_par, hdbscan_par, file, files_dir, save_data_dir, save_figures_dir, umap_path_title, False, plot_title = plot_title_add)
                
        
    
    # Run with one dataset and one set of parameters
    if args.type == "paramtest":
        print("--------- PERFORMING PARAMETER TEST---------------")
        # Running parameter test for UMAP and HDBSCAN together, just change data based on which i want to run
        # data; file; embed_col;data_dir, fig_dir variables are same as decided above

        # Defining parameters
        umap_nneigh = [3, 5, 10, 15, 20] # capturing structure with components
        umap_mindist = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.30] # how clumpy the points are

        if args.param == "umap":
            umap_save_figures_dir = save_figures_dir + 'paramtesting/umap_only/'
            for n in umap_nneigh:
                for d in umap_mindist:
                    print("RUNNING UMAP - n_neighbors: ", n, ", min_dist: ", d)
                    umap_params = [n, d]
                    # Run UMAP
                    data_postumap = run_umap(data, embed_col, 2, umap_params)
                    # Plot UMAP data 
                    just_plot_postUMAP(data_postumap, "x", "y", [umap_save_figures_dir, umap_path_title], umap_params, title_add = plot_title_add)

        if args.param == "umap_and_hdbscan":
            umap_totalparams = [[10, 0.15], [10, 0.2], [10, 0.25], [15, 0.15], [15, 0.20], [15, 0.25], [20, 0.10], [20, 0.15], [20, 0.25]] # from umap above for complete data
            # umap_totalparams = [[5, 0.15], [5, 0.2], [5, 0.25], [10, 0.1], [10, 0.15], [10, 0.2], [10, 0.25], [15, 0.05], [15, 0.15], [15, 0.2], [15, 0.25], [20, 0.05], [20, 0.15], [20, 0.2], [20, 0.25]] # from umap above for removed data
            
            hdbscan_minclust = [5, 10, 15] # smallest group to be cluster
            hdbscan_minsamp = [3, 5, 10] # conservativeness of clusters 

            params_save_data_dir = save_data_dir + "paramtesting/" 
            params_save_figures_dir = save_figures_dir + 'paramtesting/'
           
            hdbscan_outputs = open(params_save_data_dir + 'hdbscan_outputs_removed.txt', 'w')
            hdbscan_outputs.write("n_neighbors, min_dist, min_cluster_size, min_samples, noise_percentage, relative_validity \n")

            start_time_over = time.perf_counter()
            # Loop through UMAP Parameters
            for u in umap_totalparams:
                # Loop through HDBSCAN Parameters
                    for c in hdbscan_minclust:
                        for s in hdbscan_minsamp:
                            start_time = time.perf_counter()
                            
                            print("RUNNING UMAP - n_neighbors: ", u[0], ", min_dist: ", u[1])
                            print("RUNNING HDBSCAN - min_clust: ", c, ", min_samp: ", s)
                            # Run_all command - making sure it outputs to the right file directory

                            complete_df, rel_val = run_all(data, embed_col, 2, u, [c,s], file, files_dir, params_save_data_dir, params_save_figures_dir, umap_path_title, True, plot_title_add)    
                            noise_perc = sum(complete_df["labels"] > -1)/ len(complete_df["labels"]) # Count percentage of non noise
                            hdbscan_outputs.write(str(u[0]) + "," + str(u[1]) + "," + str(c) + "," + str(s) + "," + str(noise_perc) + "," + str(rel_val) + "\n")
                            
                            end_time = time.perf_counter()
                            print("TIMING: ", end_time - start_time) 
            
            end_time_over = time.perf_counter()
            print("TIMING TOTAL: ", end_time_over - start_time_over) 

            hdbscan_outputs.close()


if __name__ == '__main__':
    # Set up for parameter testing AND to run with specified dataset and parameters

    parser = argparse.ArgumentParser(description="CGT abstract", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-t", "--type", choices = ["single", "paramtest" ],  help="Single, Parameter Test")
    parser.add_argument("-d", "--data", choices = ["ft_complete", "ft_removed", "ft_replaced" ],  help="FT Complete, FT Discipline Data Removed, FT Discipline Data Replaced, Parameter Testing")
    parser.add_argument("-p", "--param", choices = ["umap", "umap_and_hdbscan" ],  help="UMAP, UMAP+HDBSCAN (after running just UMAP first)")
    
    args = parser.parse_args()
    config = vars(args)
    print(config)
    main(args)




