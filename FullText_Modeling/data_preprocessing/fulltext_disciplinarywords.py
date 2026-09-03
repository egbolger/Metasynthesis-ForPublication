### USE THIS FILE FOR DISCIPLINARY REMOVE
# read in json, remove words, then write out to json for passing to docembedding.py
# Read in "singlestr_FinalIncludedArticles_fulltext.json" from "fulltext_jsonreformatting.py" in "datafilesforCGT/complete" so the file has the id attached to it

# Write out to removed folder - file "remove_singlestr_FinalIncludedArticles_fulltext.json"


######## Libraries #########
import json
import pandas as pd
import re

######## Directories #########
complete_filedir = "/mnt/research/NLP-Lit-Review/bolger/metasyn/data/datafilesforCGT/complete/"
removed_filedir = "/mnt/research/NLP-Lit-Review/bolger/metasyn/data/datafilesforCGT/removed/"


### REMOVE DISCIPLINARY WORDS from the text
# Loop through each entry in json file
# get text datat
# do removing words
# write out to new json with same title and id information 


# Read in data to remove and set up
def set_word_data(word_path, two_words = False):
    with open(word_path) as f:
        word_list = [line.rstrip() for line in f]
        
    if two_words == True:
        # Create sublist for two words
        subword_list = []
        for word in word_list:
            if " " in word:
                split_word = word.split(" ")
                subword_list.append(split_word)
            else:
                subword_list.append(word)
        return subword_list
    else:
        return word_list
    

# Remove disciplinary words
# Find two words
def find_twowords(single_paper, two_data):    
    twoword_indices = [] # list of indices to remove
    for word_idx in range(0, len(single_paper)-2): # abs in list form with words and punctuation
       for twoword in two_data:
            if single_paper[word_idx].lower() == twoword[0]: # if the first words match, check second word
                if (single_paper[word_idx+1] == ' ' or single_paper[word_idx+1] == '-' ) and single_paper[word_idx+2].lower() == twoword[1]: 
                    if single_paper[word_idx+3] == ' ': # next word is space, remove it, else keep whatever other character
                        twoword_indices.append([word_idx, word_idx+1, word_idx+2, word_idx+3]) # append word, space, word, and space
                    else:
                        twoword_indices.append([word_idx, word_idx+1, word_idx+2]) # append word, space, word
    return twoword_indices
 
# Find single words
def find_oneword(single_abs, one_data):
    noone_indices = []
    for word_idx in range(len(single_abs)): # abs in list form with words and punctuation
        for oneword in one_data:
            if single_abs[word_idx].lower() == oneword: # if words match
                if single_abs[word_idx+1] == " ": 
                    noone_indices.append([word_idx, word_idx+1]) # append word, space
                else:
                    noone_indices.append([word_idx]) # append word

    return noone_indices

def remove_words(single_paper, two_data, one_data):
    # Remove two words
    twoword_indices = find_twowords(single_paper, two_data)
    twoword_indices_unzipped = [idx for idxlist in twoword_indices for idx in idxlist] # unzip 
    notwo_words = [single_paper[i] for i in range(len(single_paper)) if i not in twoword_indices_unzipped]

    # Remove one word
    oneword_indices = find_oneword(notwo_words, one_data)
    oneword_indices_unzipped = [idx for idxlist in oneword_indices for idx in idxlist] # unzip 
    noone_words = [notwo_words[i] for i in range(len(notwo_words)) if i not in oneword_indices_unzipped]
    return noone_words


# Make str from the old_pdf[text] into a list (sep by punctuation) for removing words
def makestr_list(old_pdf):
    return re.split('(\\W)', old_pdf)


# takes lists of words from removing and recreates the string 
def makelist_str(list_old_pdf):
    pap_str = ""
    for entry in list_old_pdf:
        pap_str += entry
    return pap_str


# rewrites entry in the data frame
def create_removediscp_json(old_pdf, two_data, one_data):
    # print("in new json")
    new_pdf = {}
    
    for key in old_pdf:
        if key == "title":
            title_str = old_pdf[key]
        if key == "item_id":
            item_id_str = old_pdf[key]
        if key == "text": # remove disciplinary words
            text_to_list = makestr_list(old_pdf[key])
            remove_list = remove_words(text_to_list, two_data, one_data)
            remove_str = makelist_str(remove_list)
            # print(remove_str)
            
    new_pdf["title"] = title_str
    new_pdf["item_id"] = item_id_str
    new_pdf["text"] = remove_str
    return new_pdf


def main():
    # Read in "singlestr_FinalIncludedArticles_fulltext.json" from "fulltext_jsonreformatting.py", which reformats the scrapped data to id, title, and text all in one string
        # reading from "datafilesforCGT/complete" so the file has the id attached to it

    with open(complete_filedir + "singlestr_FinalIncludedArticles_fulltext.json", 'r') as file:
        onestr_json = json.load(file, strict=False)

    # Read in Disciplinary Data
    oneword_data = set_word_data(removed_filedir+"science_words_oneword.txt")
    twoword_data = set_word_data(removed_filedir+"science_words_twowords.txt", two_words=True)

    # for each pdf, remove disciplinary words, create new json
    new_json = [ create_removediscp_json(onestr_json[article_idx], twoword_data, oneword_data) for article_idx in range(len(onestr_json))]
    print(len(new_json)) # 204 woot!

     # Write out new json file 
    json_str = json.dumps(new_json)
    
    with open(removed_filedir + "remove_singlestr_FinalIncludedArticles_fulltext.json", "w") as f:
        f.write(json_str)



if __name__ == "__main__":
    main()
