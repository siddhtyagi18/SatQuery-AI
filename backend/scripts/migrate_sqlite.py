import sqlite3
from pathlib import Path

root = Path(__file__).resolve().parent.parent.parent
for db in root.glob("**/*.db"):
    print("Checking DB:", db)
    conn = sqlite3.connect(db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [r[0] for r in cursor.fetchall()]
    print("  Tables:", tables)
    if "analyses" in tables:
        cursor.execute("PRAGMA table_info(analyses);")
        cols = [r[1] for r in cursor.fetchall()]
        print("  Existing cols:", cols)
        for new_col in [
            "compatibility",
            "limitations",
            "adaptation",
            "specialist_selected",
            "input_summary",
        ]:
            if new_col not in cols:
                print(f"  --> Adding column {new_col}...")
                cursor.execute(f"ALTER TABLE analyses ADD COLUMN {new_col} JSON;")
        conn.commit()
    conn.close()
print("Migration check complete.")
