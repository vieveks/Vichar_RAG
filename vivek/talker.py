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

def find_most_similar_chunk(query, vector_database, model):
    query_embedding = model.encode(query)
    max_similarity = 0 
    most_similar_chunk = ""
    for key, value in vector_database.items():
        vector = value['embedding']
        similarity = calculate_cosine_similarity(query_embedding, vector)
        if similarity > max_similarity:
            max_similarity = similarity
            most_similar_chunk = value['chunk']
    return most_similar_chunk, max_similarity

    return 0


input("Press Enter to continue... to find top k similar chunks")
def find_top_k_similar_chunks(query, vector_database, model, k):
    query_embedding = model.encode(query)
    similarity_scores = [0.0]
    similarity_scores_ids = []
    similarity_scores_chunks = []
    similarity_scores_embeddings = []

    max_similarity = similarity_scores[0]

    for key, value in vector_database.items():
        vector = value['embedding']
        similarity = calculate_cosine_similarity(query_embedding, vector)
        if len(similarity_scores) < k+1:
            similarity_scores.append(similarity)
            similarity_scores_ids.append(key)
            similarity_scores_chunks.append(value['chunk'])
            similarity_scores_embeddings.append(vector)
            similarity_scores.sort(reverse=True)
        else:
            if similarity > similarity_scores[-1]:
                similarity_scores[-1] = similarity
                similarity_scores_ids[-1] = key
                similarity_scores_chunks[-1] = value['chunk']
                similarity_scores_embeddings[-1] = vector
                similarity_scores.sort(reverse=True)
            else:
                pass

    return similarity_scores_ids, similarity_scores_chunks, similarity_scores_embeddings, similarity_scores

ids, chunks, embeddings, similarity_scores = find_top_k_similar_chunks("what is prajeets height?", vector_database, model, 3)
print(ids)
print(chunks)
print(len(embeddings))
print(similarity_scores)
input("Press Enter to continue...")

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
def chatbot(query, chat_model):
    similar_chunk, similarity = find_most_similar_chunk(query, vector_database, model)
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


'''
solved top k

def find_top_k_similar_chunks(query, vector_database, model, k):
    query_embedding = model.encode(query)

    # start empty
    similarity_scores = []
    similarity_scores_ids = []
    similarity_scores_chunks = []
    similarity_scores_embeddings = []

    for key, value in vector_database.items():
        vector = value['embedding']
        similarity = calculate_cosine_similarity(query_embedding, vector)

        # append the first k items directly
        if len(similarity_scores) < k:
            similarity_scores.append(similarity)
            similarity_scores_ids.append(key)
            similarity_scores_chunks.append(value['chunk'])
            similarity_scores_embeddings.append(vector)

        else:
            # check if this similarity is higher than the smallest in the list
            min_index = similarity_scores.index(min(similarity_scores))
            if similarity > similarity_scores[min_index]:
                # replace the lowest score and its corresponding info
                similarity_scores[min_index] = similarity
                similarity_scores_ids[min_index] = key
                similarity_scores_chunks[min_index] = value['chunk']
                similarity_scores_embeddings[min_index] = vector

    # finally, sort everything by descending similarity to return in order
    combined = sorted(
        zip(similarity_scores, similarity_scores_ids, similarity_scores_chunks, similarity_scores_embeddings),
        reverse=True
    )

    similarity_scores, similarity_scores_ids, similarity_scores_chunks, similarity_scores_embeddings = zip(*combined)

    return list(similarity_scores_ids), list(similarity_scores_chunks), list(similarity_scores_embeddings), list(similarity_scores)

'''