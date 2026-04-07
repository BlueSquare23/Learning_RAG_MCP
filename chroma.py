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

