# Similar vibe to Bib_to_MLData.py
## Goal: dataframe with the filename/identifier, title, full author, author abbreviated, year, jounral, and human codes. 
    # This looks like combining "Finalized022226_ML_LibraryDatabaseExport.csvt" and "Final included articles (Library database).bib"
    # Taking "Finalized022226_ML_LibraryDatabaseExport.csv" column, matching based on Full Title, adding missing columns from .bib file
# NOTE: Finalized022226_ML_LibraryDatabaseExport.csv is updated and finalized version of ML_LibraryDatabaseExport.csv from July 2025

## Inputs: "data/human_codes/Finalized022226_ML_LibraryDatabaseExport.csv"; "data/Final included articles (Library database).bib" 
## Outputs: 


######## Libraries #########
import json
import pandas as pd
import bibtexparser
from bibtexparser.bparser import BibTexParser

######## Directories #########
files_dir = "/mnt/research/NLP-Lit-Review/bolger/metasyn/data/"

######### Functions #########

# Takes bibtex file and converts to dataframe
def create_fulltext_df(library):
    # Create DF from bibtext
    abs_data = pd.DataFrame(None, columns = ['filename', 'full_author', 'abv_author', 'title', 'abstract', 'year', 'journal'], index = range(len(library.entries)))
    # print(library.entries)
    # print(f"\n\t{len(library.entries)} entries")
    
    # Loop through bibtex entries
        # add individual entry to data set
        # add to abstract combined entry, if applicable
    idx = 0
    for paper in library.entries: # loop through papers
        for entry in paper.keys(): # loop through keys in dictionary
            
            if entry == "title":
                title = paper[entry]
                title =  title.replace("{", "").replace("}", "")
                abs_data.loc[idx, "title"] = title

            if entry == "author":
                author = paper[entry]
                author =  author.replace("{", "").replace("}", "")
                abs_data.loc[idx, "full_author"] = author
            
            if entry == "journal":
                journal = paper[entry]
                journal =  journal.replace("{", "").replace("}", "")
                abs_data.loc[idx, "journal"] = journal

            # Year is not bibtext attribute, getting from file name of pdf I added in Zotero
            if entry == "file":
                filename = paper[entry]
                abs_data.loc[idx, "filename"] = filename
                abv_author, year = process_filename(paper[entry])
                abs_data.loc[idx, "abv_author"] = abv_author
                abs_data.loc[idx, "year"] = year

            if entry == "abstract":
                abstract = paper[entry]
                abs_data.loc[idx, "abstract"] = abstract
        
        idx += 1
        
    return abs_data

# Looks like 
    # file = {Hollowell et al. - 2017 - Course Design, Quality Matters Training, and Stude.pdf:files/25453/Hollowell et al. - 2017 - Course Design, Quality Matters Training, and Stude.pdf:application/pdf}
def process_filename(field_value):
    # Split by '-', return first entry as author and second as year
    value_split = field_value.split(" - ")
    abv_author, year = value_split[0], value_split[1]

    return (abv_author, year)

# Processes str for title text comparison
        # NOTE: you did a version with just casefold(). it left 4 from human codes that were unmatched. you checked them all and it was becasuse of punctuation. this helps that. 
        # titles were: use of virtual labs to support demand-oriented..., Vision & Change: Why It Matters.,  Program Assessment for an Undergraduate Statis..., The myths and misconceptions of change for STE...
            # the last one is a duplicate in the human code.... and there are two version. 
# THIS FIXED LEFT ONLY ONES - IE ONES IN HUMAN CODES THAT WEREN'T IMMEDIATELY RECOGNIZED IN BIBTEX
def process_str(series):
    return (
        series
        .str.strip()
        .str.casefold()
        .str.replace(r'[^a-z0-9]', '', regex=True)
    )

# Compares bibtext dataframe and matches by full title to match human codes data base
 # Start with ML_HumanCodes and add in bibtext one
    # loop through ml_humancodes, find title -- loop through bib file and compare titles 
        # if match, add in rest of row
def match_dataframes(humancodes_df, bib_df):
    # Get columns of bib_df and add to humancodes_df
    new_colnames = humancodes_df.columns.tolist() + bib_df.columns.tolist()

    # Create new df
    combined_df = pd.DataFrame(columns=new_colnames)

    # Lowercasing columns so can match 
    humancodes_df['title_lower'] = process_str(humancodes_df['Title'])
    bib_df['title_lower'] = process_str(bib_df['title'])
    
    # Match dfs
        # e.g. why row count may not equal (mis)match count - does matter now bc updated set
        # df1 keys:  A B C D E F G H
        # df2 keys:  A B C D E F X Y Z W
    combined_df = pd.merge(humancodes_df, bib_df, on='title_lower', how='outer' , indicator=True)
    print(combined_df[combined_df['_merge'] == 'left_only']) # humancodes no match in bib_df - 0
    print(combined_df[combined_df['_merge'] == 'right_only']) # bib_df no match in human_codes - 0
    print("Amount in 'both' in merged df", len(combined_df[combined_df['_merge'] == 'both']))
      
    return combined_df

def reorder_cols(combined_df):
    com_cols_names = combined_df.columns.tolist()
    # col_order = [0, 24, 1, 2, 26, 25, 27, 28, 29, 3:23, 30]
    
    misc_combined = combined_df.iloc[:, [0, 24, 1, 2, 26, 25, 27, 28, 29]]
    print(misc_combined)
    slice_combined = combined_df.iloc[:, 3:23]
    print(slice_combined)
    final_combined = combined_df.iloc[:, [23, 30]]
    print(final_combined)

    final_df = pd.concat([misc_combined, slice_combined, final_combined], axis=1)
    # print(final_df)
    # print(final_df.columns.tolist())
    return final_df
    


def main():
    # Import human codes csv file - from Ying on 2/27/26 as finalized set
    ml_lib_exp = pd.read_csv(files_dir + "human_codes/Finalized022226_ML_LibraryDatabaseExport.csv")
    ml_lib_exp  = ml_lib_exp.replace(r'\n', '', regex=True)
    print(ml_lib_exp.head())
    print("Length of Human Codes:", len(ml_lib_exp)) #204

    # Import .bib file, use bib parser, and pull information -- Match title to add to new df 
    with open(files_dir + "Final included articles (Library database).bib") as bibtex_file:
       final_bib = bibtexparser.load(bibtex_file)
    final_bib_data = create_fulltext_df(final_bib)
    
    # Remove duplicate Cavanagh entry
    # for idx in range(len(final_bib_data)):
    #     if "Cavanagh" in final_bib_data.loc[idx, "filename"]:
    #         print(final_bib_data.loc[idx])
    #         print(idx) #193
    final_bib_data = final_bib_data.drop(index=193).reset_index(drop=True)
    
    print(final_bib_data[0:5])
    print("Length of Bibtex:", len(final_bib_data)) #204

    # Import file with full text data just to compare numbe rof entries
    with open(files_dir + 'singlestr_FinalIncludedArticles_fulltext.json', 'r') as f:
        data = json.load(f)
    print(f"Number of entries: {len(data)}") # also 204, thank god


    # Create combined dataframe with human codes and meta data from bibtex
    combined_df = match_dataframes(ml_lib_exp, final_bib_data)
    print(combined_df)
    print("Length of combined_df:", len(combined_df))  # also, 204 again, woot woot 
    
    # Reorganize the columns of the dataset so it makes more sense to read 
    hc_bib_df = reorder_cols(combined_df)
    print(hc_bib_df.head())

    # Export file in csv and pkl 
    hc_bib_df.to_csv(files_dir + 'datafilesforCGT/hc_bib_metadata.csv')
    hc_bib_df.to_pickle(files_dir + 'datafilesforCGT/hc_bib_metadata.pkl')

   
    
    
if __name__ == "__main__":
    main()