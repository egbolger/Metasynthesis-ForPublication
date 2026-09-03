#### Document Embeddings 
#### Embedding full text documents with jinav5-small model

######## Libraries #########
import json
import pandas as pd
import pyarrow
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
import time

######## Directories #########
files_dir = "/mnt/research/NLP-Lit-Review/bolger/metasyn/data/datafilesforCGT/"

######## Code #########

# Calculate total number of tokens
    # UPDATE: all total number of tokens are max length of jina -> so no sliding window 
def token_overunder(article_list, model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
    # Max length = 32768
    over, under = 0,0
    
    for pap_idx in range(0, len(article_list)):
        paper = article_list[pap_idx]["text"]
        inputs = tokenizer(paper, padding=True, truncation=True, return_tensors='pt') # tokenize text
        num_tokens = inputs["input_ids"].shape[1] # Number of input tokens

        print(num_tokens)
        if num_tokens == 32768: # likely over max level
            over+=1
        else:
            under+=1
    print("Number of Over:", over, "; Number of Under:", under)
    print("Percentage of Over:", over/len(article_list), "; Percentage of Under:", under/len(article_list))


# CHANGE THIS BACK TO A FUNCTION so easy to write out for the different data set (complete, remove, replace)
def run_embed(data, model_id, embed_col_name, filefolder, filename):
    start_time1 = time.time()

    device = 'cuda' if torch.cuda.is_available() else 'cpu' # running with gpus
    model = SentenceTransformer(model_id, trust_remote_code=True, device=device)
    
    ids = [data[pap_idx]["item_id"] for pap_idx in range(0, len(data))  ]
    papers = [data[pap_idx]["text"] for pap_idx in range(0, len(data))  ]
    # print(papers)
    embeddings = model.encode(sentences=papers, task="clustering", batch_size=1, show_progress_bar=True, convert_to_numpy = True)
    print(embeddings)
    print(embeddings.shape)

    end_time1 = time.time()
    elapsed_time1 = end_time1 - start_time1
    print(f"Elapsed time: {elapsed_time1} seconds")

    # Create DataFrame
    embed_papers = list(zip(ids, embeddings))
    embed_df = pd.DataFrame(embed_papers, columns=['item_id', embed_col_name])
    print(embed_df)

    # Export Data
    embed_df.to_csv(files_dir + filefolder + "/embedding_files/" + filename + '.csv') 
    embed_df.to_pickle(files_dir + filefolder + "/embedding_files/" + filename + '.pkl')
    embed_df.to_parquet(files_dir + filefolder + "/embedding_files/" + filename + '.parquet')


def main():
    # Load jina model and tokenizer
    model_id = 'jinaai/jina-embeddings-v5-text-small' 

    # Read json file with Full Text Data
    with open(files_dir +'complete/singlestr_FinalIncludedArticles_fulltext.json', 'r') as file:
        FT_articles = json.load(file, strict=False)
        
    with open(files_dir +'removed/remove_singlestr_FinalIncludedArticles_fulltext.json', 'r') as file:
        FTremoved_articles = json.load(file, strict=False)
    
    # Calculate the number of over and under the total token count - All are under :D
    # token_overunder(FT_articles, model_id)

    # we dont need any of this, all in context window and using GPU
    # We use batch_size to determine how many to send to each processor/gpus
        # chunk_size is splitting text, we don't need/want that
        # https://sbert.net/examples/sentence_transformer/applications/computing-embeddings/README.html
        # https://milvus.io/ai-quick-reference/how-can-you-use-a-gpu-to-speed-up-the-embedding-generation-with-sentence-transformers-and-what-changes-are-needed-in-code-to-do-so

    # Create the embeddings and write them out to a csv, pkl, parquet
        # Full text
    # run_embed(FT_articles, model_id, "ft_embedding", "complete", "ft_embedding")

        # with remove disciplinary words
    run_embed(FTremoved_articles, model_id, "ftremoved_embedding", "removed", "ftremoved_embedding")

    
   

    
    

if __name__ == "__main__":
    main()

# Different version of model.encode line
    # This is doing it by hand, model.encode line is doing it based on jina's optimized model
#    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
#    model = AutoModel.from_pretrained(model_id, trust_remote_code=True)
 # inputs = tokenizer(paper, padding=True, truncation=True, return_tensors='pt')  # Tokenize
        # with torch.no_grad():
        #     embeddings = model(**inputs).last_hidden_state.mean(dim=1) 
                # gets input token ids, pads and truncs to max length, returns torch objects - 
                # https://medium.com/axinc-ai/how-tokenizer-parameters-impact-transformers-behavior-8be8030637c6
