#!/usr/bin/env python3
"""
TDZ C64 Knowledge - CLI Tool
Manage the knowledge base from the command line.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server import KnowledgeBase

def main():
    parser = argparse.ArgumentParser(
        description="TDZ C64 Knowledge - CLI Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add a single PDF
  python cli.py add "C:/docs/c64_programmers_reference.pdf" --title "C64 Programmer's Reference" --tags reference basic assembly

  # Add all PDFs in a folder
  python cli.py add-folder "C:/docs/c64/" --tags reference

  # Search the knowledge base
  python cli.py search "SID register $D400"

  # List all documents
  python cli.py list

  # Show stats
  python cli.py stats
        """
    )
    
    parser.add_argument(
        "--data-dir",
        default=os.environ.get("TDZ_DATA_DIR", os.path.expanduser("~/.tdz-c64-knowledge")),
        help="Data directory (default: ~/.tdz-c64-knowledge)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Add command
    add_parser = subparsers.add_parser("add", help="Add a document")
    add_parser.add_argument("filepath", help="Path to PDF or text file")
    add_parser.add_argument("--title", "-t", help="Document title")
    add_parser.add_argument("--tags", "-g", nargs="+", default=[], help="Tags for the document")
    
    # Add folder command
    folder_parser = subparsers.add_parser("add-folder", help="Add all PDFs/text files in a folder")
    folder_parser.add_argument("folder", help="Folder path")
    folder_parser.add_argument("--tags", "-g", nargs="+", default=[], help="Tags for all documents")
    folder_parser.add_argument("--recursive", "-r", action="store_true", help="Include subfolders")
    
    # Search command
    search_parser = subparsers.add_parser("search", help="Search the knowledge base")
    search_parser.add_argument("query", help="Search query")
    search_parser.add_argument("--max", "-m", type=int, default=5, help="Max results")
    search_parser.add_argument("--tags", "-g", nargs="+", help="Filter by tags")
    
    # List command
    subparsers.add_parser("list", help="List all documents")
    
    # Remove command
    remove_parser = subparsers.add_parser("remove", help="Remove a document")
    remove_parser.add_argument("doc_id", help="Document ID")
    
    # Stats command
    subparsers.add_parser("stats", help="Show knowledge base stats")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize knowledge base
    kb = KnowledgeBase(args.data_dir)
    print(f"Data directory: {args.data_dir}\n")
    
    if args.command == "add":
        try:
            doc = kb.add_document(args.filepath, args.title, args.tags)
            print(f"Added: {doc.title}")
            print(f"  ID: {doc.doc_id}")
            print(f"  Chunks: {doc.total_chunks}")
            if doc.total_pages:
                print(f"  Pages: {doc.total_pages}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
    
    elif args.command == "add-folder":
        folder = Path(args.folder)
        if not folder.exists():
            print(f"Error: Folder not found: {folder}")
            sys.exit(1)
        
        extensions = ['.pdf', '.txt', '.md', '.asm', '.bas', '.inc', '.s']
        
        if args.recursive:
            files = [f for ext in extensions for f in folder.rglob(f"*{ext}")]
        else:
            files = [f for ext in extensions for f in folder.glob(f"*{ext}")]
        
        if not files:
            print("No PDF or text files found.")
            return
        
        print(f"Found {len(files)} files to process...\n")
        
        success = 0
        failed = 0
        for filepath in files:
            try:
                doc = kb.add_document(str(filepath), None, args.tags)
                print(f"[OK] {doc.filename} ({doc.total_chunks} chunks)")
                success += 1
            except Exception as e:
                print(f"[FAIL] {filepath.name}: {e}")
                failed += 1
        
        print(f"\nDone: {success} added, {failed} failed")
    
    elif args.command == "search":
        results = kb.search(args.query, args.max, args.tags)
        
        if not results:
            print(f"No results for: {args.query}")
            return
        
        print(f"Found {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            print(f"--- {i}. {r['title']} (score: {r['score']}) ---")
            print(f"ID: {r['doc_id']}, Chunk: {r['chunk_id']}")
            print(f"\n{r['snippet']}\n")
    
    elif args.command == "list":
        docs = kb.list_documents()
        if not docs:
            print("No documents in knowledge base.")
            return
        
        print(f"Documents ({len(docs)}):\n")
        for doc in docs:
            print(f"• {doc.title}")
            print(f"  ID: {doc.doc_id}")
            print(f"  File: {doc.filename}")
            print(f"  Chunks: {doc.total_chunks}")
            if doc.tags:
                print(f"  Tags: {', '.join(doc.tags)}")
            print()
    
    elif args.command == "remove":
        if kb.remove_document(args.doc_id):
            print(f"Removed document: {args.doc_id}")
        else:
            print(f"Document not found: {args.doc_id}")
    
    elif args.command == "stats":
        stats = kb.get_stats()
        print("Knowledge Base Statistics:")
        print(f"  Documents: {stats['total_documents']}")
        print(f"  Chunks: {stats['total_chunks']}")
        print(f"  Total Words: {stats['total_words']:,}")
        print(f"  File Types: {', '.join(stats['file_types']) or 'none'}")
        print(f"  Tags: {', '.join(stats['all_tags']) or 'none'}")


if __name__ == "__main__":
    main()
