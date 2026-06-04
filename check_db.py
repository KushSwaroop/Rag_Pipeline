"""Quick check on LanceDB status and test vector search."""
import lancedb

db = lancedb.connect("./rag_lancedb")
t = db.open_table("documents")

row_count = t.count_rows()
print(f"Total chunks in DB: {row_count}")

# Test a vector search
from database import get_embedding
query = "climate temperature"
embedding = get_embedding(query)

if embedding:
    results = t.search(embedding).limit(3).to_list()
    print(f"\nSearch results for '{query}':")
    for r in results:
        src = r["source"]
        text_preview = r["text"][:120].replace("\n", " ")
        print(f"  [{src}]  {text_preview}...")
else:
    print("Embedding failed - is Ollama running?")
