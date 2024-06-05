# Notes

## RAG Architecture

![rag architecture](images/rag.png)

On a high level, the architecture of a RAG system can be distilled down to the pipelines shown in the figure above:

- A recurring pipeline of document pre-processing, ingestion and embedding generation.
- An inference pipeline with a user query and response generation.

Let us understand the first pipeline in detail.

### Document Ingestion:

First, raw data from diverse sources, such as databases, documents, or live feeds, is ingested into the RAG system. To pre-process this data, LangChain provides a variety of document loaders that load data of many forms from many different sources.

The term document loader is used loosely. Source documents do not necessarily need to be what you might think of as standard documents (PDFs, text files, and so on). For example, LangChain supports loading data from Confluence, CSV files, Outlook emails, and more. LlamaIndex also provides a variety of loaders, which can be viewed in LlamaHub.

### Document Pre-processing

After documents have been loaded, they are often transformed. One transformation is text-splitting, which breaks down long texts into smaller segments. This is necessary for fitting the texts into the embedding model, like e5-large-v2 which has a maximum token length of 512. While splitting the text sounds simple, this can be a nuanced process. Remember tokenizers!!

### Generate Embeddings

When data is ingested, it must be transformed into a format that the system can efficiently process. So we convert data into high-dimensional vectores, which represent text in a numerical format.

### Storing Embeddings in a Vector Database

The generated embeddings are now stored in vector dbs. These dbs enable rapid search and retrieval operations on vectors.
The architecture of RAG consists of:

- Retriever:
- Generator:

### Training RAG

Training involves two main stages:

- Retriever Training:
- Joint Training

### Key Challenges

- Computational Resources
- Quality of Retrieval
- Scalability

RAG represents a significant step forward in making AI models more informative and context-aware by combining the strengths of retrieval and generative approaches.

## How does it work?

In a Retrieval-Augmented Generation (RAG) model, the generative model (such as GPT-3, BART, or T5) uses the input query along with the retrieved documents to generate the final output. Here’s a more detailed breakdown of this process:

1. Input Query: The process begins with an initial input query from the user.
2. Retrieval Phase:
   - The input query is used to search a large corpus of documents
   - The retrieval model identifies and retrieves the most relevant documents or passages related to the query
3. Combination of Inputs:
   - The retrieved info is combined with the original input query.
   - This combination typically involves appending the retrieved documents to the original query, creating a new input that includes both the query and the context provided by the retrieved documents.
4. Generation Phase: The combined input(query + retrieved info) is then fed into the generative model.

### Good RAG Frameworks

- LLama Index
- LangChain

### Retriever Models

- BM25
- DPR
