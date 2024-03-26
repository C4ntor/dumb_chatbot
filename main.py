import pandas as pd
import numpy as np
#import tensorflow as tf
#import tensorflow_hub as hub
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import sys

data_set = pd.read_csv('train.csv').dropna()
data_set['user_prompt'] = data_set['user_prompt'].str.lower()
documents = data_set["user_prompt"].to_list()



"""

def initialize_use():
    #USE (Universal Sentence Encoder)
    module_url = "https://tfhub.dev/google/universal-sentence-encoder-large/5"
    embed = hub.load(module_url)
    sentence_embeddings = embed(documents)
    return sentence_embeddings


def ms_univ(query, embed, sentence_embeddings):
    input_embedding = embed([query])
    # Calculate cosine similarity between the input sentence embedding and each sentence embedding
    similarity_scores = np.inner(input_embedding, sentence_embeddings)
    # Find the index of the most similar sentence
    most_similar_index = np.argmax(similarity_scores)   
    # Return the most similar sentence and its similarity score
    return documents[most_similar_index]

"""


#TFIDF: used to compute the data matrix
tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(documents)
tfidf_vectors = tfidf_vectorizer.transform(documents)

def ms_tfidf(query):
    #TFIDF matrix is computed independently from the user_prompt query, useful to compute it once
    tfidf_vector_query = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(tfidf_vector_query, tfidf_vectors)
    # Find the index of the most similar document
    most_similar_index = similarities.argmax()
    return documents[most_similar_index]




def most_sim_prompt(query_user_prompt, type):
    """Given the user_prompt STRING from the query_set returns the row INDEX of the most similar user_prompt in data_set (based on the type arg)"""
    if type==0:
        ms_prompt = ms_tfidf(query_user_prompt)
    else:
        ms_prompt=""
        
    return data_set[data_set["user_prompt"]==ms_prompt].index[0]
   


def reply(user_prompt, type):
    ms_prompt = data_set.iloc[most_sim_prompt(user_prompt, type)]
    assert ms_prompt.empty == False
    return ms_prompt['model_response']



def main(type):
    while True:
        print('Please enter your question, to quit enter: exit')
        u_input = input("> ")
        if u_input=="exit":
            exit()
        else:
            print(reply(u_input, type))





if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == '-o':
            #offline usage
            main(0)
        else:
            #online usage
            pass


