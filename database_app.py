import sqlite3
import sys

DB_NAME = "books.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS books (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            author TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def add_book(title, author):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO books (title, author) VALUES (?, ?)", (title, author))
    conn.commit()
    conn.close()
    print(f"Added book: '{title}' by {author}")

def list_books():
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM books")
    rows = cursor.fetchall()
    conn.close()
    if not rows:
        print("No books found.")
    else:
        for row in rows:
            print(f"ID: {row[0]}, Title: {row[1]}, Author: {row[2]}")

def delete_book(book_id):
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM books WHERE id = ?", (book_id,))
    conn.commit()
    conn.close()
    print(f"Deleted book with ID: {book_id}")

def print_usage():
    print("Usage:")
    print("  python3 database_app.py add <title> <author>")
    print("  python3 database_app.py list")
    print("  python3 database_app.py delete <id>")

def main():
    init_db()
    if len(sys.argv) < 2:
        print_usage()
        return

    command = sys.argv[1].lower()

    if command == "add":
        if len(sys.argv) < 4:
            print("Error: Missing title or author.")
            print_usage()
        else:
            add_book(sys.argv[2], sys.argv[3])
    elif command == "list":
        list_books()
    elif command == "delete":
        if len(sys.argv) < 3:
            print("Error: Missing book ID.")
            print_usage()
        else:
            delete_book(sys.argv[2])
    else:
        print(f"Unknown command: {command}")
        print_usage()

if __name__ == "__main__":
    main()
