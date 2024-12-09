# %%
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import EmbeddingRetriever
from haystack.schema import Document
import ollama
import os
import pickle

# %%
DOCUMENT_STORE_PKL = "document_store.pkl"

# %% [markdown]
# ## Prepare key terms embeddings

# %%
if (os.path.exists(DOCUMENT_STORE_PKL)):
    with open(DOCUMENT_STORE_PKL, 'rb') as f:
        document_store = pickle.load(f)
        retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=False 
        )
else:
    document_store = InMemoryDocumentStore(embedding_dim=384)
    key_terms = open('terms.csv').read().split('\n')
    documents = [Document(content=term) for term in key_terms]
    document_store.write_documents(documents)
    retriever = EmbeddingRetriever(
            document_store=document_store,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            use_gpu=False 
        )
    document_store.update_embeddings(retriever)
    with open(DOCUMENT_STORE_PKL, 'wb') as f:
        pickle.dump(document_store, f)

# %% [markdown]
# ## Prepare question embedding and question for Llama3.2

# %%
def get_relevant_key_terms(question, top_k=20):
    results = retriever.retrieve(query=question, top_k=top_k)
    return [doc.content for doc in results]

os.popen('ollama serve')
client = ollama.Client()

def ask_llama(question, relevant_key_terms):
    prompt = f"""
        Here is a search prompt by a user of the UN digital library:
        Question: {question}
        Here is a list of key terms. Each key term is in square brackets.
        Key Terms: {relevant_key_terms}
        Select ideally one, if necessary two key terms that are most relevant to the question.
        Output only the selected key terms as they are presented to you: each key term within square brackets, each set of square brackets separated by a semi-colon and no space.
        """

    return client.chat(model="llama3.2", messages=[{"role": "user", "content": prompt}])['message']['content']


# %% [markdown]
# ## Ask question

# %%
#question = "I would like to know more about the impact of world war 1 on the economy of Germany."
question = "world war ii"

# %%
# Get relevant key terms
key_terms = get_relevant_key_terms(question)

key_terms_array = ["[" + term.replace('"', '') + "]" for term in key_terms]
print("Key Terms:", key_terms_array)

# Join the key terms array into a single string separated by semi-colons
key_terms_string = ";".join(key_terms_array)
print("Relevant Key Terms:", key_terms_string)

# %%
is_answer_valid = False
while not is_answer_valid:
    answer = ask_llama(question, key_terms_string)
    print("Answer:", answer)

    answer_split = [term.strip() for term in answer.split(';')]
    is_answer_valid = isinstance(answer, str) and len(answer_split) <= 2
    is_answer_valid = is_answer_valid and all([term.strip() in key_terms_array for term in answer.split(';')])
    print(is_answer_valid)

# %%
print(answer_split)
output = ""
for term in answer_split:
    output += f"subjectheading:{term}"
print(output)

# %%



