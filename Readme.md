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

## Vector Search Lab

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














