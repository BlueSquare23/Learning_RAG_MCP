# Learning RAG & MCP Fundamentals

## Overview

This document contains some notes I've taken while learning MCP and RAG
fundamentals.

I'm following along with [this freecodecamp.org Youtube video](https://youtu.be/I7_WXKhyGms).

## Table of Contents

What this tutorial covers.

* A Simple RAG Explanation 
* When Not to use RAG?
* What is RAG?
* Vector Search vs Semantic Search
* Embedding Models
* VectorDB
* Chunking
* RAG Architecture
* Caching, Monitoring, Error Handling
* RAG in Production

There are also hands-on free labs in the browser after each section to practice
what we've learned.

## A Simple Explanation of RAG 

RAG is about giving providing context for LLM models.

In a simple form, RAG takes your internal knowledge and makes it accessible to
an LLM. It does so by searching your knowledge store for relevant sections of
information and then incorporating that into the response the LLM synthesises.

**Retrieval Augmented Generation**

1. We retrieve only the relevant information.

2. We augment the prompt with the retrieved information.

3. We generate a response using the additional information.

## When Not to Use RAG

Rag is not the solution to all problems. 

There are different ways to get better responses out of an LLM. 

* We can prompt better (prompt engineering)

* We can fine tune models

* RAG

### Building a Policy Bot

Say we notice employees copying and pasting internal policy documents into
ChatGPT to get answers. So we decide to build an internal bot to help with
this.

The bot should be aware of all of our internal policy documents and should be
able to generate accurate responses from them.

We also want to limit type of responses the bot can give. There are certain
topics that we'd rather have employees talking with HR about directly (salary
increases, sensitive topics).

Lastly, we want our chatbot to have a warm friendly tone and a particular way
of speaking.

### When to use Prompt Engineering

Prompt engineering is a good solution to the limitations we put on the model.
We can use it to set guardrails like "never reveal personal employee
information" or "if someone asks about sensitive topics, politely redirect them
to HR."

### Voice Style & Language - Fine Tuning

Fine tuning is a good solution for creating a model with a custom voice and
style of speaking. Its good because it automatically applies the voice to all
responses and is embedded in the model.

Fine tuning is great for stable unchanging things like communication style, but
bad for dynamic factual information.

### When to use Rag

Lastly, RAG is great for that dynamic factual information.

## What is RAG?

We discussed in short how rag works with the retrieve, augment, and generate
steps. But we'll now look at that in more detail.

### Retrieval - Vector Search

How do we find the right document with context relevant to the user? And what
do you search for within these files?

We start by identifying a few keywords from the users question.

> What is the **reimbursement** policy for **home-office** setup?

We could do a literal "grep" like search. However, that would only match the
literal string we're searching for.

Nonetheless, it is still a popular method of searching documents. We just count
the occurrences of keywords and whatever docs have the highest occurrence get
the highest scores.

There are two popular keyword search algorithms:

* TF-IDF
* BM25

#### TF-IDF Example

We can use these via the sidekit learn python module.

> [!NOTE]
> **PSUEDO CODE**

```python
from sklearn... import TfidfVectorizer

# Sample Docs
docs = [ "Office equipment policy",
    "Office furniture guidelines",
    "Office travel policy" ]

# Create TF-IDF analyzer
analiyzer = TfidfVectorizer()

# Find word importance scores
word_scores = analyzer.fit_transform(docs)

# Print importances
print(word_scores.toarray())
```

The wordscores show a bi-dimensional array with the word in each sentence.

Now we can run a query on the analyzer transform to query the docs for a word.

```python
# Test search
query = "furniture"
query_scores = analyzer.transform([query])

print(f"Query scores: {query_scores.toarray()}")
print(f"Query: {query} => Document {get_document_id(scores) + 1}")
```

This would print:

```
Query: Scores: [[0. 1. 0. 0. 0. 0.]]
Query: 'furniture' -> Best match: Document 2
```

It returns a score of 1 indicating the second document is the right document.

#### BM25 Example

Now lets look at an example using the BM25 algo.

We use the `rank_bm25` library which is a popular module with a BM25 algo.

```python
from rank_bm25 import BM25Okapi

# Sample Docs
docs = [ "Office equipment policy",
    "Office furniture guidelines",
    "Office travel policy" ]

# Create BM25 index
bm25 = BM25Okapi([doc.split() for doc in docs])

word_scores = bm25.get_scores()

# Print importences 
print(word_scores)
```

In this output we can see the wordscores are slightly different. For example,
the word "Office" has been given a zero because it appears everywhere, so bm25
scores it lower.

And again we run a query and print scores array:

```
query = "furniture"
query_scores = bm25.get_scores(query.split())

print(f"Query scores: {query_scores}")
print(f"Query: {query} => Document {get_document_id(scores) + 1}")
```

Which again still identifies the second document as being the right document.

```
Query: Scores: [[0. 1. 0. 0. 0. 0.]]
Query: 'furniture' -> Best match: Document 2
```

## Lab 1 - Vector Search

Labs linked here:

https://kode.wiki/4nzgBbW

Full Link:

https://learn.kodekloud.com/user/courses/rag-mcp-fundamentals-a-hands-on-crash-course?utm_source=youtube&utm_medium=link&utm_campaign=fcc_rag_mcp&utm_term=&utm_content=

You have to create a free account.

I scp'ed the lab files to `lab1` dir.

Basically, its just showing the same thing as we did above but with a hybrid
approach too.

```
cd /home/lab-user/techcorp-docs
grep -r -i "holiday" . > /home/lab-user/extracted-content.txt
```

To setup lab1:

```
Task 3: Environment Setup

    Navigate to the project:

cd /home/lab-user/rag-project
ls -la
cat requirements.txt

    Create virtual environment:

uv venv venv
source venv/bin/activate

    Install packages:

uv pip install -r requirements.txt

    Verify installation:

uv run python -c "import sklearn, pandas, numpy; print('All packages available')"
```

Then to run the scripts:

```
cd /home/lab-user/rag-project
uv run python tfidf_search.py

uv run python bm25_search.py
```

Task 6: Hybrid Search Analysis

Hybrid search combines TF-IDF and BM25 scores to get the best of both methods. It takes a weighted average:

```python
hybrid_score = (tfidf_weight * tfidf_score) + (bm25_weight * bm25_score)
```

For example, with equal weights (50% each):

```python
hybrid_score = 0.5 * tfidf_score + 0.5 * bm25_score
```

Run the hybrid search script:

```
cd /home/lab-user/rag-project
uv run python hybrid_search.py
```

Look for the "Equal weights" section in the output. What is the hybrid score for "remote work policy"?

## Semantic Search

One problem with keyword search is it only matched for literal matches and not
synonyms. It doesn't know anything about the meaning of the text, just literal
string matching.

That's the area of _Semantic Search_ or searching by meaning of the keywords
used, not just literal strings.

To do this we convert the strings to "embeddings." Embeddings are just
tokenized words in multidimensional vector space. Once a document has been
broken down to its embeddings, words that are similar in meaning will appear
nearer to each other in this vector space.

## Embedding Models

We all know there are different kinds of machine learning models such as
Computer Vision or Text To Speech. Within the Natural Language Processing (NLP)
space there's a category of specialized model for "Sentence Similarity."

If we look at the examples on huggingface.co we can see a few.

https://huggingface.co/models?pipeline_tag=sentence-similarity&sort=trending

A good one in particular is the `all-MiniLM-L6-v2`

https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2 

> This is a sentence-transformers model: It maps sentences & paragraphs to a
> 384 dimensional dense vector space and can be used for tasks like clustering
> or semantic search.

We can see its a 22.7M parameter model. A parameter is just a number learned by
the model during training to help it better produce natural language. So for
this model there are 22 million weights that the model learned. You can think
of parameters as roughly representing the brain power of the model.

Since this model has a relatively small parameter size compared to models like
GPT-3.5 (175B) or GPT-4.0 (1.8T), its only really useful for creating these
embeddings, not text generation or "agentic reasons."

What these embedding models do is convert a text sentence into a vector space
representation of it, which contains semantic information about meanings.

### Dot Product

We can use the dot product to measure distances between two vectors in our
multidimensional vector space. That's how the similarity of words is found.

Let's look at an extremely oversimplified example in two dimensions.

We have three sentences:

* Dogs are allowed in the office [1, 5]

* Pets can come to work on Furry Fridays [2, 4]

* Remote work policy allows three days from home [6, 1]

First we multiply the values in the arrays:

```
[1, 5] * [2, 4] = [2, 20]
[2, 4] * [6, 1] = [12, 4]
[1, 5] * [6, 1] = [6, 5]
```

We then add the multiplied numbers together.

```
2 + 20 = 22
6 + 5 = 11
12 + 4 = 16
```

Finally, we apply a normalization function to convert the sums to floats.
Numbers closer to 1 are more similar.

Of course we don't have to do this by hand. We can use `numpy` to compute the
dot products for us.

```python
import numpy as np

# Calculate Similarity 
similarity = np.dot(vector1, vector2)
print(similarity)  # Prints a number between 0 and 1
```

Or to compare and compute the dot products using Sentence Transformer.

```
pip install sentence-transformers numpy
```

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

# test documents
sentences = [
    "Dogs are allowed in the office on Fridays",
    "Pets can come to work on Furry Fridays",
    "Remote work policy allows 3 days from home"
]

embeddings = model.encode(sentences)

# Use numpy's dot product to compare similarity
sim_1_2 = np.dot(embeddings[0], embeddings[1])
sim_1_3 = np.dot(embeddings[0], embeddings[2])
sim_2_3 = np.dot(embeddings[1], embeddings[2])

print(f" Dogs vs Pets: {sim_1_2*100:.1f}% similar")
print(f" Dogs vs Remote: {sim_1_3*100:.1f}% similar")
print(f" Pets vs Remote: {sim_2_3*100:.1f}% similar")
```

Outputs:

```
 Dogs vs Pets: 73.3% similar
 Dogs vs Remote: 36.2% similar
 Pets vs Remote: 33.8% similar
```

### Lab 2 - Embeddings

Task 1: Keyword Search Limitations

First navigate to the project, activate the already created virtual environment
venv and install the requirements:

```shell
cd /home/lab-user/rag-project
source venv/bin/activate
uv pip install -r requirements.txt
```

Then, run this command to see keyword search limitations:

```shell
cd /home/lab-user/rag-project
uv run python keyword_limitation_demo.py
```

Why did the search for "distributed workforce policies" fail to find relevant documents?


Task 2: Install Embedding Dependencies

* Navigate to the project:

```
cd /home/lab-user/rag-project
```

* Install embedding packages:

```
uv pip install "sentence-transformers==2.2.2" "huggingface_hub<0.20" openai
```

* Verify installation:

```
uv run python -c "import sentence_transformers, openai; print('Embedding packages available')
```

Task 3: Local Embeddings Demo

Run the local embeddings script:

```
cd /home/lab-user/rag-project
uv run python semantic_search_demo.py
```

## Vector Databases

The thing is sequentially searching all of our documents and comparing
embeddings for each one against our query _does Not scale well._

So instead we use vector databases to store our embeddings. Think of vector
databases as having a smart librarian that can retrieve relevant results
instantly. They do this by using smart indexing algorithms.

But what does _"indexing"_ mean?

### Indexing Algorithms

Indexing works by pre-organizing our vectors into related neighborhoods.

So all the "animal" related vectors would be grouped together on one axis, and
all "remote work" policies would be grouped together on another, etc.

There are a few common algorithms used to accomplish this:

* Hierarchical Navigable Small World (HNSW)
  - Works by creating a graph where each vector is connected to its most
    similar neighbor. So when searching it just drops in at a random point on
    the graph and navigates the graph to find the closest matches.

* Inverted File (IVF)

* Locality Sensitive Hashing (LSH)


### Vector Database Implementations

* Chroma: Open Source and Python Friendly

* Pinecone: Managed Service, Takes care of infra for you

* Weaviate: Has a graph tool api

For most users you'll likely start with Chroma for learning, small projects and
then for production use cases might scale up to using something like Pinecone.

### Using Chroma DB

First install required libraries:

```
pip install chromadb
```

Then we import the chromadb library, connect to the client, and create a
collection. We can then add policy documents to the collection. 

```python
import chromadb

client = chromadb.Client()

collection = client.create_collection("policies")

# Our "policy docs"
policies = [
    "Dogs are allowed in the office on Fridays",
    "Pets can come to work on Furry Fridays",
    "Remote work policy allows 3 days from home"
]

# Add policy docs to db
for i, policy in enumerate(policies):
    collection.add(
        documents=[policy],
        ids=[f"policy_{i}"]
    )

# Query
query = "Can I bring my dog to work?"
results = collection.query(query_texts=[query], n_results=2)

print(results)
```

* Output:

```
{'ids': [['policy_0', 'policy_1']], 'embeddings': None, 'documents': [['Dogs are allowed in the office on Fridays', 'Pets can come to work on Furry Fridays']], 'uris': None, 'included': ['metadatas', 'documents', 'distances'], 'data': None, 'metadatas': [[None, None]], 'distances': [[0.8606346249580383, 0.8716706037521362]]}
```

There are some additional things we'll want to do when working with chromadb.

First off, out of the box chromadb client is not persistent. So we'll want to
use the `PersistentClient` method instead when getting a client.

```python
# Option 1: Specify a directory for persistence
client = chromadb.PersistentClient(path="/path/to/db/files/")

# Option 2: Use a specific database file
client = chromadb.PersistentClient(path="./chroma.db")
```

### Lab 3 - ChromaDB and Documents

🗄️ Vector Databases Lab

Learn how to scale semantic search with vector databases - the foundation of
enterprise RAG systems.

Task 1: Vector Database Concepts

Before we start building, let's understand what vector databases are and why we need them.

Vector databases are specialized databases designed to store and search
high-dimensional vectors (embeddings). Unlike traditional databases that store
text, numbers, or structured data, vector databases are optimized for
similarity search.

Key benefits of vector databases:

* Store millions of embeddings efficiently
* Fast similarity search across all vectors
* Persistent storage that survives restarts
* Can be shared across multiple applications

What is the primary advantage of using a vector database over storing embeddings in memory?

Task 2: Install Vector Database Dependencies

Navigate to the project:

```
cd /home/lab-user/rag-project
```

~Activate~ Create virtual environment:

```
#source .venv/bin/activate
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt
```

Install embedding model package:

```
uv pip install sentence-transformers
```

Install vector database package:

```
uv pip install chromadb
```

Verify installation:

```
uv run python -c "import chromadb; print('ChromaDB available')"
```

Task 3: Initialize ChromaDB Vector Database

Now let's set up our vector database! The initialization script will:

* Create a ChromaDB client - our connection to the vector database
* Create a collection - a container for our documents (like a table in SQL)
* Load the embedding model - converts text to 384-dimensional vectors
* Test with sample data - verify everything works correctly

Install the embedding model package:

```
cd /home/lab-user/rag-project
uv pip install sentence-transformers
```

Run the ChromaDB initialization script:

```
uv run python init_vectordb.py
```

Task 4: Store Documents in Vector Database

Now let's populate our vector database with real documents! The storage script will:

* Load TechCorp documents - Read all markdown files from the document repository
* Generate embeddings - Convert each document to 384-dimensional vectors
* Store in ChromaDB - Save documents and their embeddings in the collection
* Verify storage - Confirm all documents were successfully added

Store TechCorp documents in the vector database:

```
cd /home/lab-user/rag-project
uv run python store_documents.py
```

Task 5: Perform Vector Search

Let's see vector search in action! The demo script will:

* Create a ChromaDB collection - Set up a new vector database
* Add sample documents - Insert 4 TechCorp policy documents
* Perform semantic search - Query "Can I work from home?" and find relevant docs
* Show similarity scores - Display how well each document matches the query

💡 Before running, take a moment to read the script:

```
cat vector_search_demo.py
```

Then run the vector search demo:

```
cd /home/lab-user/rag-project
uv run python vector_search_demo.py
```

Task 6: Save Vector Database to File

Let's make our vector database persistent! The save script will:

* Create sample documents - Add 4 TechCorp policy documents to the collection
* Persist the database to disk - Use ChromaDB's built-in persistence at ./chroma_db
* Export to JSON file - Save all documents and metadata to vectordb_backup.json
* Verify file creation - Check file size and confirm successful save
* Show persistence benefits - Data survives restarts, can be shared between apps

Save the vector database to a file for persistence:

```
cd /home/lab-user/rag-project
uv run python save_vectordb.py
```

Task 6: Save Vector Database to File

Let's make our vector database persistent! The save script will:

* Create sample documents - Add 4 TechCorp policy documents to the collection
* Persist the database to disk - Use ChromaDB's built-in persistence at ./chroma_db
* Export to JSON file - Save all documents and metadata to vectordb_backup.json
* Verify file creation - Check file size and confirm successful save
* Show persistence benefits - Data survives restarts, can be shared between apps

Save the vector database to a file for persistence:

```
cd /home/lab-user/rag-project
uv run python save_vectordb.py
```

Task 7: Test Database Persistence

Let's test the persistence of our vector database! The check script will:

* Not add anything - Will not add any new entry to the database

* Reads from the persistent database - Read the entry done in the database using the save script in the previous task

Test the vector database contents to demonstrate persistence:

```
cd /home/lab-user/rag-project
uv run python check_persistence.py
```

### Chunking & The Precision Probem

Now if the document is small enough, we could vectorize and store the whole
thing in one chunk in the database. However, in reality most documents are full
of a bunch of different sections with different information. If our model just
returned the whole document everytime, it wouldn't be very useful.

So instead what we do is split the documents up into chunks. 

One way to do this is with fixed size chunking. Where we just break the
document up every say 500 words and store those as unique chunks. But that has
the downside of clipping sentences and ideas midway.

For example, we could end up with "Dogs are allowed" and "on Fridays" in two
different chunks, which would be bad if our bot though dogs were allowed
always because it didn't see the chunk about only on fridays.

One way to fix this is to add a 50 character overlap between the chunks. That
has better results for the particular "Dogs allowed on fridays" case. But there
are still other issues with fixed size chunking.

There are other methods of chunking as well like sentence based chunking. Or
paragraph based chunking. There's even semantic and agentic chunking, those are
out of scope.

So chunking is actually a somewhat complicated problem. If we have too big of a
chunk, then end user get's overwhelmed with information. Whereas if we have too
small of chunks, we could be missing vital context.

> [!NOTE]
> **PSUEDO CODE**

Let's look at an example chunking function:

```python
def chunk_document(text, chunk_size=500, overlap=50):
    """Split docs into overlapping chunks"""
    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        # Try to break at sentence boundary
        if end < len(text):
            last_period = chunk.rfind('.')
            if last_period > chunk_size * 0.7:
                chunk = chunk[:last_period + 1]
                end = start + last_period + 1

        chunks.append(chunk.strip())
        start = end - overlap  # Overlap for context

    return chunks 
```

This function takes our document and splits it into overlapping chunks. It also
tries to break on sentences and handle the end of the document properly.

So then in our code from earlier, we'd include each chunk in our chroma db instead.

```python
chunks = chunk_document(large_policy_doc_text, chunk_size=500, overlap=50)

# Add chunks to vector database
for i, chunk in enumerate(chunks):
    collection.add(
        documents=[chunk],
        ids=[f"chunk_{i}"]
    )

# Query our db
...
```

Good guidelines for chunking are:

* Size: 200-500 characters, 50-100 character overlap.
* Boundary Rules: Split at sentences, Avoid mid-word breaks
* Quality Checks: Test with real queries, Verify context preservation, Monitory search results

In production we can use libraries like LangChain's `RecursiveCharacterTextSplitter` 
and spaCy's `SpacyTextSplitter` to take care of chunking for us.

### Lab 4 - Document Chunking

Document Chunking for RAG

Why Chunking Matters
Large documents are hard to search effectively. Chunking breaks them into smaller, focused pieces that improve retrieval accuracy and make your RAG system smarter.

What You'll Learn:

* Setup: Verify environment & install packages
* Task 1: Understand why chunking matters
* Task 2: Basic character-based chunking
* Task 3: Overlap chunking for context
* Task 4: Sentence-aware chunking with spaCy
* Task 5: Chunked vector search comparison
* Task 6: Agentic chunking with AI

Chunking Methods Covered:

* Basic - Split by character count
* Overlap - Preserve context at boundaries
* Sentence-Aware - Break at natural language points
* Agentic - AI-powered semantic splitting

Duration: ~20-30 minutes

Setup: Verify Environment

Before starting, let's set up your environment.

Navigate to the project and ~activate~ setup the virtual environment:

```
cd lab4/rag-project
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt
```

Run the setup script (auto-installs all packages):

```
python verify_environment.py
```

The script will automatically:

* ✅ Check virtual environment is active
* ✅ Install all missing packages
* ✅ Verify modules can be imported
* ✅ Download spaCy model if needed
* ✅ Check API configuration

Task 1: Why Document Chunking Matters

📄 View the script - chunking_problem_demo.py to see how it works

What this demo shows:
This script demonstrates the core problem with searching large documents in RAG systems. It creates a sample employee handbook and shows how searching for specific information (like 'internet speed requirements') returns the entire document instead of just the relevant section.

What you'll see:

* A large document stored as a single chunk
* Search queries that should find specific sections
* Results that return the entire document (the problem!)
* Clear explanation of why this is problematic for RAG

Run this command to see the chunking problem:

```
cd /home/lab-user/rag-project
uv run python chunking_problem_demo.py
```

What is the main problem with searching large documents without chunking?

Task 2: Basic Document Chunking

📄 View the script - basic_chunking.py to see how it works

What this demo shows:
This script demonstrates basic document chunking using LangChain's RecursiveCharacterTextSplitter. It takes a sample document and breaks it into smaller, manageable chunks based on character count and separators.

What you'll see:

* Original document length and content preview
* LangChain text splitter configuration
* Document split into multiple chunks (6 chunks total)
* Each chunk's length, content, and separator used
* Benefits of basic chunking approach

Run the basic chunking demo:

```
cd /home/lab-user/rag-project
uv run python basic_chunking.py
```

Task 3: Chunking with Overlap

📄 View the script - overlap_chunking.py to see how it works

What this demo shows:
This script demonstrates the importance of chunk overlap in RAG systems. It compares chunking with and without overlap, showing how overlap preserves context across chunk boundaries and prevents loss of important information.

What you'll see:

* Same document chunked without overlap (7 chunks)
* Same document chunked with overlap (7 chunks)
* Side-by-side comparison of chunk boundaries
* Analysis of context preservation
* Clear demonstration of why overlap matters

Run the overlap chunking demo:

```
cd /home/lab-user/rag-project
uv run python overlap_chunking.py
```

Task 4: Sentence-Aware Chunking

📄 View the script - sentence_chunking.py to see how it works

What this demo shows:
This script demonstrates advanced chunking using spaCy for sentence-aware text splitting. It compares basic character-based chunking with intelligent sentence-boundary chunking, showing how breaking at natural language boundaries improves semantic coherence.

What you'll see:

* Basic character-based chunking (may break mid-sentence)
* spaCy-powered sentence-aware chunking
* Side-by-side comparison of chunk quality
* Analysis of sentence boundary preservation
* Benefits of natural language processing for chunking

Run the sentence-aware chunking demo:

```
cd /home/lab-user/rag-project
uv run python sentence_chunking.py
```

Task 5: Chunked Vector Search

📄 View the script - chunked_search.py to see how it works

What this demo shows:
This script demonstrates the complete RAG search improvement with chunking. It compares search performance between a single large document and properly chunked documents, showing how chunking leads to more precise and relevant search results.

What you'll see:

* Same document stored as single chunk vs. multiple chunks
* Multiple search queries tested on both approaches
* Similarity scores and result quality comparison
* Clear demonstration of chunking benefits
* Performance summary showing improved precision

Run the chunked vector search demo:

```
cd /home/lab-user/rag-project
uv run python chunked_search.py
```

## RAG Architecture

Okay so we know the basics of RAG (Retrieval, Augmentation, Generation), let's
zoom out and look at how they look in a real rag pipeline system.

Now all of the document processing (Scraping/Exporting/Converting, Chunking,
Creating Embeddings) needs to happen before the user ever makes a query.
Because this takes time and effort.

This system is called a RAG Pipeline.

```
Fetching Docs > Chunking > Embeddings > VectorDB
```

### Lab 5 - Complete Rag Pipeline

🎯 Complete RAG Pipeline Lab

Welcome to the final lab in our RAG crash course! In this lab, you'll see how all the components from previous labs work together to create a complete RAG system.

What you'll learn:

* How document chunking integrates with vector search
* How query processing connects to retrieval
* How context augmentation feeds into response generation
* How the complete RAG pipeline works end-to-end

This lab combines everything from Labs 01-04 into one working system!

The capstone lab that brings it all together

Task 1: Set Up Virtual Environment

Navigate to the project:

```
cd /home/lab-user/rag-project
```

Activate virtual environment:

```
cd lab5/rag-project
uv venv venv
source venv/bin/activate
uv pip install -r requirements.txt
```

Task 2: Document Loading & Chunking

📄 View the script - complete_rag_demo.py to see how it works

What this section shows:
This section demonstrates how documents are loaded and chunked for better retrieval. It shows the same chunking approach from Lab 04, but now integrated into the complete RAG pipeline.

What you'll see:

* Sample policy documents loaded
* LangChain text splitter configuration
* Document chunking with specific size and overlap
* Chunk metadata and organization

Look at the script and answer: What is the chunk size used in the text splitter?

Task 3: Vector Database Setup

📄 View the script - complete_rag_demo.py to see how it works

What this section shows:
This section demonstrates how the vector database is set up to store the chunked documents. It shows ChromaDB initialization and collection creation with specific configuration.

What you'll see:

* ChromaDB client initialization
* Collection creation with metadata
* Document storage with embeddings
* Vector database configuration

Look at the script and answer: What is the name of the collection created in ChromaDB?

Task 4: Query Processing

📄 View the script - complete_rag_demo.py to see how it works

What this section shows:
This section demonstrates how user queries are processed and converted to embeddings for vector search. It shows the same embedding model approach from Lab 02.

What you'll see:

* Query preprocessing and cleaning
* Embedding model loading
* Vector conversion process
* Embedding dimensions and properties

Look at the script and answer: What embedding model is used for query processing?

Task 5: Vector Search

📄 View the script - complete_rag_demo.py to see how it works

What this section shows:
This section demonstrates how vector search finds relevant document chunks using semantic similarity. It shows the same vector search approach from Lab 03.

What you'll see:

* Vector similarity search execution
* Result ranking and scoring
* Top-k result selection
* Similarity score calculation

Look at the script and answer: How many top results are returned by default in the vector search?

Task 6: Context Augmentation

📄 View the script - complete_rag_demo.py to see how it works

What this section shows:
This section demonstrates how retrieved context is assembled into a structured prompt for the LLM. It shows how multiple sources are combined into coherent context.

What you'll see:

* Context assembly from search results
* Prompt construction with policies
* Information formatting and structure
* Context length management

Look at the entire script and answer: What similarity metric is used in the ChromaDB collection?

Task 7: Response Generation

📄 View the script - complete_rag_demo.py to see how it works

What this section shows:
This section demonstrates how the LLM generates responses using the augmented prompt. It shows the final step of the RAG pipeline.

What you'll see:

* LLM integration (simulated)
* Response formatting and structure
* Answer synthesis process
* Output generation

Look at the entire script and answer: What is the chunk overlap used in the text splitter?

Task 8: Complete RAG Pipeline Results

📄 View the script - complete_rag_demo.py to see how it works

What this demo shows:
This is the complete RAG pipeline in action! You'll see all the components working together from document loading to response generation.

What you'll see:

* Complete pipeline execution
* Step-by-step processing
* Multiple test queries
* End-to-end RAG system

Run the complete RAG pipeline demo:

```
cd /home/lab-user/rag-project
uv run python complete_rag_demo.py
```

Based on the pipeline output, how many chunks are created from the 5 policy documents?

## RAG Production Concerns - Caching, Monitoring, Error Handling

Rag systems are slow. Every query involves multiple expensive operations.

To help resolve some of this slowness we can cache at multiple levels:

* Embeddings
* Search Results
* Final Answers

The simplest is to store complete question answer pairs. This way if someone
asks the same question, we can return the exact same answer instantly.

The embeddings cache stores the computed vectors for text. This is useful
because generating embeddings is expensive.

Vector search cache stores the results of DB queries. This is useful because
similar queries will likely return similar chunks.

Lastly, we can store the generated answers.

We can use Redis to store our cached results.

### Monitoring

In order to ensure our RAG system is behaving appropriately we need to measure
and monitor our rag system.

The best general metrics are:

* Response Time: How fast we answer questions?
* Throughput: How many queries per second?
* Error Rate: What percentage of requests fail?

However, RAG systems also have their own metrics we want to track:

* Retrieval Quality: Measure how relevant returned chunks are to users question
* Embedding Performance: How long it takes to generate vector embeddings
* Chunking Efficiency: How well we're breaking up documents

We can then set alert thresholds to know when something is going wrong. So if
there's a performance issue.

### Example Production RAG System Architecture

So can split our RAG system up into these layers.

```
┌──────────────────────────────┐
│        Monitroing Stack      │  (Graphana, Prometheus, ELK)
├──────────────────────────────┤
│       Application Layer      │  (WebUI, API, etc)
├──────────────────────────────┤
│       Rag Pipeline Layer     │  (Query Service, Chunking, Embeddings, R A G)
├──────────────────────────────┤
│           Data Layer         │  (Vector DB, Redis, MySQL/Postgres)
└──────────────────────────────┘
```

This clean separation of concerns allow us to build scalable production rag
systems.

## Model Context Protocol (MCP)

### Table of Contents

* Why MCP?
* What is MCP?
* MCP Architecture
* JSON-RCP
* Using an MCP Server
* Building an MCP Server
* MCP Inspector
* Building an MCP Client

Also labs!

### Why do we need MCP?

Well out of the box an LLM can't do anything except for return text. Own their
own the models can't call an API or connect to a database, etc. If we want an
AI to take some action we need to connect it to an AI agent.

AI agents are long running with a memory of past requests and they act as a
middle man between the main LLM and tool interactions.

An agent is a lot like a traditional program except its control flow is not
pre-determined by hardcoded logic. Instead it interacts with an LLM in order to
determine what path it should take, what tool it should use, when it should
exit, etc.

A typical AI agent workflow looks like this:

* User sends request to the agent.
* The agent interacts with the LLM to extract the right details from the query.
* The LLM responds back with the extracted details.
* The agent asks the LLM what tools to interact with to get the job done.
* The LLM responds with the right tool to use.
* The agent uses those details and tools make a call to the external API with those details.
* The agent asks the LLM what to do next.
* The LLM says fetch the preferences from the database.
* The agent fetches the details from the db and asks the LLM what flight to pick.
* The LLM responds with the best flight based on preference and available flights.
* The agent books the flight and returns the flight details to the user.

### Tools & Standardization

So what is a tool really and how does an LLM use it to interact with an
airline. Well a tool is just a piece of hardcoded logic for interacting with
something in the real world. So it could be an api call or a database call or a
command runner or a file system tool, etc.

MCP follows a client server architecture. Usually MCP clients live inside of
coding agents like Claude, Cursor, Windsurf, etc.

But not every tools needs an MCP server. For example if we ask an agent about a
problem with the code in the current directory, it will look at the git history
and at the backend changes, before identifying the issue. None of this requires
an MCP server.

However, if we wanted our AI to interact with a browser and read the console
logs, we would need to put an mcp browser server in the middle.

MCP Servers can be built by anyone who wants an AI to be able to interact with
their application.

### Lab 1 - Getting Started With Roo-Code

🦘 Getting Started with Roo-Code AI Assistant

In this introductory lab, you'll get familiar with our lab environment and learn how to use Roo-Code, an AI assistant integrated directly into VSCode.

What you'll learn:

    How to navigate the KodeKloud lab environment
    Using Roo-Code AI assistant with pre-configured settings
    Basic chat functionality with an AI assistant
    Exploring AI capabilities for coding tasks



Let's start by opening the Roo-Code AI assistant interface.

    Look at the left sidebar of VSCode
    Find and click on the kangaroo icon 🦘
    This will open the Roo-Code interface panel

💡 Tip: Roo-Code is an AI assistant that can help you with coding tasks, answer questions, and assist with various development activities right within VSCode.

If Roo Code is not automatically configured with the required API keys, follow these steps:

    Click on the Import Settings option located at the bottom of the Roo Code setup panel.
    In the file picker, navigate to and select the file:

    /home/lab-user/.roo-coder/profiles/default/settings.json

    Once the file is selected, Roo Code will automatically load your saved profile and apply the settings.

### What is MCP?

MCP stands for Model Context Protocol.

* Model - The LLM
* Context - Providing access to other information
* Protocol - Standards that define how ai should talk to external services

MCP dictates that you have to use the JSON-RPC spec which is stateless.

Servers may offer these features to clients:

* Resources: Context and data, for the user of the AI model to use
* Prompts: Templated messages and workflows for users
* Tools: Functions for the AI model to execute

Clients may offer the following features to servers:

* Sampling: Server-initiated agentic behaviors and recursive LLM interactions
* Roots: Server-initiated inquires into uri or filesystem boundaries to operate in
* Elicitation: Server-initiated requests for additional information from users

### MCP Architecture

Lets think about MCP's from the perspective of someone building one from
scratch. The goal is to get our AI to talk to a third party service. We'll we'd
need to know what APIs / endpoints that service offers and we'd need to read
the API documentation to understand how to use those endpoints.

In the AI world, we define all of this inside of **Tools**. The server must
list all of its capabilities as a list of tools defined in a specific format.
Each tool should have a description of what it does and how to use it, as well
as an input and output schema.

Next if we're building an MCP from scratch we may need certain data from
various places. For example, our user may have a preference to only book
flights on providers with a refund policy. These pieces of information are
known as **Resources**.

The MCP specification has clear rules on how to define resources. Resources
should have a URI that points to the resource. They should have a name, title,
and description that defines what this resource does.

The URI could be `HTTP://`, `FILE://`, `GIT://`, etc.

Lastly, we have **Prompts**. Prompts are templated system prompt information
that a server can provide with more information about the scope and
restrictions on using for how to interact with the MCP server.

### JSON-RPC

The MCP server and client communicate with each other using JSON-RPC (2.0).

But what is JSON-RPC?

Well the JSON stands for JSON, which we already know. And the RPC, stands for
Remote Procedure Call. Together they define how to messages and call methods
remotely.

So say we have a server that exposes a method like the following:

```python
def add(a: float, b: float):
    """Add two numbers"""
    return Success(a + b)
```

The client should be able to invoke that method remotely and pass parameters to
it.

```python
rpc_request = request("add", {3,2})
```

Under the hood this would send a request and trigger a response like the
following:

Request:

```json
{
  "jsonrpc": "2.0",
  "method": "add",
  "params": {
    "a": 10,
    "b": 5
  },
  "id": "a3f2c1b4"
}
```

Response:

```json
{
  "jsonrpc": "2.0",
  "result": 15,
  "error": {}, //if error
  "id": "a3f2c1b4"
}
```

Note that JSON-RPC is a stateless and protocol independent. Meaning there's no
connection kept alive between client and server and that we can use any
protocol (HTTP, STDIO, AMQP, etc.) to transmit our messages.


### How to Use an MCP Server

Okay before we can really build an MCP server we need to learn how to use one.

Most programmers are familiar with making requests to API's by hand. But with
an MCP, that part of it is already taken care of for you. The MCP client
takes care of the connection logic and request logic. So instead all you write
is code that lists the mcp's available tools and to call those tools.

When you start a tool like calude code, it launches an mcp server in stdio mode
on the local box. You can then specify what mcp(s) to connect to in your
`mcp.json` file.


### Building an MCP Server

There's a bunch of different SDK's for building MCP servers.

https://modelcontextprotocol.io/docs/sdk

Python reference docs:

https://py.sdk.modelcontextprotocol.io/api/




















