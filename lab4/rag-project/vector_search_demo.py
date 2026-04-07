#!/usr/bin/env python3
"""
Vector Search Demo
Demonstrate semantic search using ChromaDB
"""

import chromadb
from sentence_transformers import SentenceTransformer

print("🔍 Vector Search Demo")
print("=" * 50)

# Initialize ChromaDB and model
print("1. Setting up search system...")
client = chromadb.Client()
collection = client.create_collection("techcorp_docs")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("   ✅ Search system ready")

# Add some sample documents
print("2. Adding sample documents...")
sample_docs = [
    "TechCorp allows remote work up to 3 days per week with manager approval",
    "Employees can bring their pets to work on Fridays",
    "The company provides health insurance and dental coverage",
    "Remote workers must use company-approved equipment and software"
]

collection.add(
    documents=sample_docs,
    ids=[f"sample_{i+1}" for i in range(len(sample_docs))]
)
print(f"   ✅ Added {len(sample_docs)} sample documents")

# Test queries
print("3. Testing vector search...")
test_queries = [
    "Can I work from home?",
    "What about bringing my dog to work?",
    "What benefits are available?",
    "What equipment do I need for remote work?"
]

print()
for i, query in enumerate(test_queries, 1):
    print(f"Query {i}: '{query}'")
    
    # Search using ChromaDB
    results = collection.query(
        query_texts=[query],
        n_results=2
    )
    
    # Show results
    for j, (doc, distance) in enumerate(zip(results['documents'][0], results['distances'][0])):
        similarity = 1 - distance
        print(f"   {j+1}. Similarity: {similarity:.3f} - {doc}")
    
    print()

print("💡 Vector Search Benefits:")
print("✅ Understands meaning, not just keywords")
print("✅ Finds relevant documents even with different wording")
print("✅ Fast similarity search across all documents")
print("✅ Can handle millions of documents efficiently")

print()
print("🎉 Vector Search Demo Complete!")
print("✅ Vector search demo completed!")
