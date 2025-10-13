# here we will use the local json file as a vector database to create a chatbot with rag

import json

with open('vector_database.json', 'r') as f:
    vector_database = json.load(f)


print(len(vector_database))
# for key, value in vector_database.items() :
#     print(key)

first_value = list(vector_database.values())[0]
# print(first_value['id'])
# print(first_value['embedding'])
# print(first_value['chunk'])

second_value = list(vector_database.values())[1]
# print(second_value['id'])
# print(second_value['embedding'])
# print(second_value['chunk'])

input("Press Enter to continue...")
# cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
# print(cos_sim(first_value['embedding'], second_value['embedding']))

# input("Press Enter to continue...")

# import torch
# import torch.nn.functional as F

# tensor1 = torch.tensor(first_value['embedding'])
# tensor2 = torch.tensor(second_value['embedding'])
# # Method 1: Using F.cosine_similarity
# cos_sim_1 = F.cosine_similarity(tensor1, tensor2)
# print(cos_sim_1)

## numpy method 
import numpy as np

def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    magnitude_vec1 = np.linalg.norm(vec1)
    magnitude_vec2 = np.linalg.norm(vec2)
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0  # Handle cases with zero vectors to avoid division by zero
    
    return dot_product / (magnitude_vec1 * magnitude_vec2)

vec1 = first_value['embedding']
vec2 = second_value['embedding']
# print(calculate_cosine_similarity(vec1, vec2))


#Dot product
score = np.dot(vec1, vec2)
print("Dot product score:", score)

# Euclidean distance
distance = np.linalg.norm(vec1 - vec2)
print("Euclidean distance:", distance)

#BM25 
from rank_bm25 import BM25Okapi
tokenized_corpus = [doc.split(" ") for doc in chunks]
bm25 = BM25Okapi(tokenized_corpus)
tokenized_query = query.split(" ")
scores = bm25.get_scores(tokenized_query)
print("BM25 scores:", scores)

# Cross-encoder
from sentence_transformers import CrossEncoder
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L6-v2')
score = cross_encoder.predict([query, chunk])
print("Cross-encoder score:", score)



# Now we have to find the most similar chunk to the query

# calculate the cosine similarity for all the chunks in the vector database
# !pip install transformers
from transformers import AutoModel
from numpy.linalg import norm

# cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
# embeddings = model.encode(['How is the weather today?', 'What is the current weather like today?'])
# print(cos_sim(embeddings[0], embeddings[1]))
# print(len(embeddings))

def find_most_similar_chunk(query, vector_database, model, k = 3):
    query_embedding = model.encode(query)
    similarities = []
   #  max_similarity = 0 
   # most_similar_chunk = ""
    for key, value in vector_database.items():
        vector = value['embedding']
        similarity = calculate_cosine_similarity(query_embedding, vector)
        similarities.append((similarity, value['chunk']))

    similarities.sort(reverse=True, key=lambda x: x[0])
    return similarities[:k]
   #      if similarity > max_similarity:

    #     if similarity > max_similarity:
    #         max_similarity = similarity
    #         most_similar_chunk = value['chunk']
    # return most_similar_chunk, max_similarity

    # return 0


# query = "What is the unique move in kings gambit opening?"
# similar_chunk, similarity = find_most_similar_chunk(query, vector_database, model)
# print(f"Most similar chunk: {similar_chunk}")
# print(f"Similarity: {similarity}")


# Now we use it to create a chatbot with rag
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os
client = OpenAI()
chat_model = "gpt-5"
api_key = os.getenv("OPENAI_API_KEY")
client.api_key = api_key

# completion = client.chat.completions.create(
#   model="gpt-5",
#   messages=[
#     {"role": "developer", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message)
def chatbot(query, chat_model, k = 3):
    similar_chunk= find_most_similar_chunk(query, vector_database, model,k)
    combined_chunks = " ".join([chunk for _, chunk in similar_chunk])
    improved_query = f"The query is: {query}. The chunk of text is: {similar_chunk}. You need to answer the query based on the chunk of text."
    response = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "developer", "content": "You are a helpful assistant. You are given a chunk of text and a query. You need to answer the query based on the chunk of text."},
            {"role": "user", "content": improved_query}
        ]
    )
    return response.choices[0].message

print(chatbot("What is the unique move in kings gambit opening?",  chat_model))