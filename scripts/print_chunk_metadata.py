import sqlite3
import random
import json
from pathlib import Path


def pick_value(row):
    _, string_value, int_value, float_value, bool_value = row
    if string_value is not None:
        return string_value
    if int_value is not None:
        return int_value
    if float_value is not None:
        return float_value
    if bool_value is not None:
        return bool_value
    return None


def main():
    db_path = Path("chroma_db/chroma.sqlite3")
    if not db_path.exists():
        print("No chroma_db/chroma.sqlite3 found.")
        return

    conn = sqlite3.connect(str(db_path))
    cur = conn.cursor()

    cur.execute("SELECT DISTINCT id FROM embedding_metadata ORDER BY id;")
    ids = [row[0] for row in cur.fetchall()]
    if not ids:
        print("No embedding_metadata ids found.")
        return

    chunk_id = random.choice(ids)
    cur.execute(
        """
        SELECT key, string_value, int_value, float_value, bool_value
        FROM embedding_metadata
        WHERE id = ?
        ORDER BY key
        """,
        (chunk_id,),
    )
    rows = cur.fetchall()
    if not rows:
        print(f"No metadata found for id={chunk_id}")
        return

    meta = {row[0]: pick_value(row) for row in rows}

    print(f"id: {chunk_id}")
    print(f"topic_id: {meta.get('topic_id')}")
    print(f"topic_label: {meta.get('topic_label')}")
    print(f"topic_keywords: {meta.get('topic_keywords')}")

    node_content = meta.get("_node_content")
    if node_content:
        try:
            node_data = json.loads(node_content)
            text = node_data.get("text") or ""
            if text:
                print("\ntext:")
                print(text)
        except json.JSONDecodeError:
            print("\n_node_content (raw):")
            print(node_content)


if __name__ == "__main__":
    main()
