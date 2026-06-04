import lancedb

def check_progress():
    db = lancedb.connect("./rag_lancedb")
    if "documents" not in db.table_names():
        print("Table 'documents' not found.")
        return

    t = db.open_table("documents")
    print(f"Total chunks in DB: {t.count_rows()}")
    
    # We can fetch a sample of records to see what files are in there
    # since we can't use to_pandas() without pylance
    results = t.search().limit(50000).to_list()
    sources = set([r["source"] for r in results if "source" in r])
    print("\nIngested sources so far:")
    for src in sorted(sources):
        print(f"  - {src}")

if __name__ == "__main__":
    check_progress()
