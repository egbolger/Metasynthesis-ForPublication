## GOAL: We are taking the Full Text Json file that Anthony created from scrapping the PDFs ("data/FinalIncludedArticles_fulltext.json") and reformatting it. The data for each pdf are separated by page (e.g. the 'text' key holds a list where each element is the text from the page). This file will create a new json file where each entry will have a title and the compiled text for the paper in one string (table and figures included). 

## Inputs: "data/FinalIncludedArticles_fulltext.json"
## Outputs: "data_processing/FinalIncludedArticles_fulltext_compiled.json" and a pkl file with column of unique identifiers. 



######## Libraries #########
import json
import pandas as pd

######## Directories #########
files_dir = "/mnt/research/NLP-Lit-Review/bolger/metasyn/data/"


######## Data reorganizing #########
# Old Json is a list of dictionaries. Each dictionary corresponds to one pdf
## Each entry in the list for text, tables, and images corresponds to one page of text. For Text, the entries are strings. For tables and images, the entries are None or sublist - where each subentry is related to an image/table on that page.      

def collect_text(val_list):
    remove_none = [x for x in val_list if x != None] 
    return "".join(remove_none)
    

def collect_tablesimages(val_list):    
    comb_val = ""
    remove_none = [x for x in val_list if x != None] 
    
    for entry in remove_none:
        if isinstance(entry, list): 
            comb_val += "".join(entry)         
        else:
            comb_val += entry
    return comb_val


###### Processing string and filenames to match to get ids

def process_str(string):
    return (string.strip().casefold().replace(r'[^a-z0-9]', '') )

def process_filename(field_value):
    # Split by '-', return first entry as author and second as year
    value_split = field_value.split(" - ")
    author_year_pair = value_split[0] + " - " + value_split[1]
    return author_year_pair

def process_filename_borrego(field_value):
    # Split by '-', return first entry as author and second as year
    value_split = field_value.split(" - ")
    author_year_title_pair = value_split[0] + " - " + value_split[1] + " - " + value_split[2][0:50]
    return author_year_title_pair


# Matching by filename with final df from "full_text_getbibinfo.py" to add the unique id to this dataset. 
def match_filename(final_df, pdf_title):
    # loop through final_df. If pdf_title matches a title in the final_df dataset, return the ID
        # by title, we really mean the file name 
    paper_id = 0
    # I'm sure there's a simpler way to do this, but rn idc 
    if "Borrego et al. - 2013" in pdf_title or "BORREGO et al. - 2013" in pdf_title:
        pdf_title = process_str(process_filename_borrego(pdf_title))
        for idx in range(len(final_df)):        
            curr_title = process_filename_borrego(process_str(final_df.loc[idx, "filename"]))
            if curr_title == pdf_title:
                paper_id = final_df.loc[idx, "Item ID"]
                # print(pdf_title)
                # print(curr_title)
                # print(paper_id) 
    elif "Zemliansky" in pdf_title: # bc the file name for this is messy and has IEEE Xplore Abstract Record, bc ofc it does 
        # print(pdf_title)
        for idx in range(len(final_df)):
            if "Zemliansky ( )" in final_df.loc[idx, "Short Title"]:
                paper_id = final_df.loc[idx, "Item ID"]
                # print(paper_id)
                
    else: # for all the rest
        # Loop through final_df and find filename match
        pdf_title = process_str(process_filename(pdf_title))
        for idx in range(len(final_df)):        
            curr_title = process_filename(process_str(final_df.loc[idx, "filename"]))
            if curr_title == pdf_title:
                paper_id = final_df.loc[idx, "Item ID"]
                # print(curr_title)
                # print(pdf_title)
                # print(paper_id)
   
    # since json doesnt do numpy
    paper_id = int(paper_id)
    return paper_id
    
# New json is also a list of dictionaries, with each dictionary corresponding to a pdf. The two key, value pairs are title and text which has  text, tables, images combined into one string.
def create_combined_json(old_pdf, final_df):
    new_pdf = {}
    for key in old_pdf:
        if key == "title":
            title_str = old_pdf[key]
        if key == "text":
            text_str = collect_text(old_pdf[key])
            # try:
                
            # except Exception as e:
            #     print(key, old_pdf)
            #     print(e)
        if key == "tables":
            tab_str = collect_tablesimages(old_pdf[key])
        if key == "images":
            im_str = collect_tablesimages(old_pdf[key])
            
    # add text, table, and images to one string
    fullpaper_text = text_str + tab_str + im_str

    new_pdf["title"] = title_str
    new_pdf["item_id"] = match_filename(final_df, title_str)
    new_pdf["text"] = fullpaper_text
    return new_pdf



def main():
    # Read in "data/FinalIncludedArticles_fulltext.json"
    with open(files_dir +'compiled_FinalIncludedArticles_fulltext_removechars.json', 'r') as file:
        original_json = json.load(file, strict=False)

    # Read in finaldf from "full_text_getbibinfo.py"
    hc_bib_df = pd.read_csv(files_dir + 'datafilesforCGT/hc_bib_metadata.csv')
    # print(hc_bib_df)
    # print(hc_bib_df['Item ID'].nunique()) # There's 204, woot woot, should be unique


    # CODE THE SHOWS THERE IS A UNIQUE ABOUT OF AUTHOR - YEAR PAIRS in the set, so this is what we match on when building the json 
        # OOPSS except for Borrego et al. - 2013, doing a special case for this one. 
    # unique_aupairs = []
    # for idx in range(len(hc_bib_df)):
    #     field_value =  hc_bib_df.loc[idx, "filename"]
    #     au_pair = process_str(process_filename(field_value))
    #     unique_aupairs.append(au_pair)
    #     if au_pair == "Borrego et al. - 2013":
    #         print(au_pair)
    # # print(unique_aupairs)
    # print(len(set(unique_aupairs)))
    

    new_json = [create_combined_json(original_json[article_idx], hc_bib_df) for article_idx in range(len(original_json))]
    print(len(new_json))

    # Write out new json file 
    json_str = json.dumps(new_json)
    
    with open(files_dir + "datafilesforCGT/complete/singlestr_FinalIncludedArticles_fulltext.json", "w") as f:
        f.write(json_str)

    # NOTE: emily has checked the ids found here by hand with the humans codes dataset. so it should be right. 






if __name__ == "__main__":
    main()
