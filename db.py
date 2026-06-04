import lancedb
from pathlib import Path

DB_DIR = "./rag_lancedb"
TABLE_NAME = "documents"

# Initialize LanceDB connection
db = lancedb.connect(DB_DIR)

def get_table():
    """Returns the open LanceDB table if it exists, otherwise None."""
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return None

def create_table(data, schema=None):
    """Creates a new table in LanceDB and builds FTS index on 'text'."""
    table = db.create_table(TABLE_NAME, data=data, schema=schema, mode="overwrite")
    try:
        # Build Tantivy FTS index for native BM25 keyword search
        table.create_fts_index("text", replace=True)
    except Exception as e:
        print(f"Failed to create FTS index: {e}. Keyword search might fall back.")
    return table

def add_to_table(data):
    """Appends data to the existing table and updates the FTS index."""
    table = get_table()
    if table:
        table.add(data)
        try:
            # Recreate/update FTS index
            table.create_fts_index("text", replace=True)
        except Exception as e:
            print(f"Failed to update FTS index: {e}")
    else:
        create_table(data)
