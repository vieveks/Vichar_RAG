# lets read a file one directory behind this folder also read a pdf

import os
import PyPDF2

cwd = os.getcwd() 
print("Current working directory:", cwd) 



print(cwd)
os.chdir("../")
os.chdir("Books")
pdf_dir = os.getcwd()
os.chdir(cwd)
# print(f"{pdf_dir}\\Books\\book1.pdf")

full_text = ""
# extracting text from all the pdfs in the folder for all pages
temp_lst = os.listdir(pdf_dir)

for file in temp_lst[:2]:
    file_path = f"{pdf_dir}\\{file}"
    with open(file_path, "rb") as text_file:
        reader = PyPDF2.PdfReader(text_file)
        reader_text = ""
        for page in reader.pages:
            reader_text += page.extract_text()
        full_text+=reader_text
    print(f"Extracted text from {file}")
    print(f"Total pages: {len(reader.pages)}")
    print(f"Total text: {len(reader_text)}")
    print(f"Text: {reader_text[0:50]}")
    print(f"Full text: length {len(full_text)}")
    print("--------------------------------")
    # input("Press Enter to continue...")



print(f"Full text length: {len(full_text)}")

# Now let's create a json where we create chunks of text from the full text and then also use embeddings to create a vector database

def create_chunks(text, chunk_size=5000):

    chunks = []
    for i in range(0, len(text), chunk_size): 
        chunks.append(text[i:i+chunk_size])

    return chunks

# using jina embeddings

input("Press Enter to continue to embeddings...")

# !pip install transformers
from transformers import AutoModel
from numpy.linalg import norm

cos_sim = lambda a,b: (a @ b.T) / (norm(a)*norm(b))
model = AutoModel.from_pretrained('jinaai/jina-embeddings-v2-base-en', trust_remote_code=True) # trust_remote_code is needed to use the encode method
embeddings = model.encode(['How is the weather today?', 'What is the current weather like today?'])
print(cos_sim(embeddings[0], embeddings[1]))
print(len(embeddings))


# if dictonary doesnt have a key, it will create it
# we need to save it also in a json file

import json
def create_vector_database(chunks):
    vector_database = {}
    for i, chunk in enumerate(chunks):
        tmp = {}
        tmp['embedding'] = model.encode(chunk).tolist()
        tmp['chunk'] = chunk
        tmp['id'] = i 
        vector_database[i] = tmp
    with open('vector_database.json', 'w') as f:
        json.dump(vector_database, f)
    return vector_database

chunks = create_chunks(full_text)
vector_database = create_vector_database(chunks)
# print(vector_database)



    