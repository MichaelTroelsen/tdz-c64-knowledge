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

    # Bulk remove command
    remove_bulk_parser = subparsers.add_parser("remove-bulk", help="Remove multiple documents")
    remove_bulk_parser.add_argument("--doc-ids", "-d", nargs="+", help="Document IDs to remove")
    remove_bulk_parser.add_argument("--tags", "-g", nargs="+", help="Remove documents with these tags")

    # Bulk update tags command
    update_tags_parser = subparsers.add_parser("update-tags-bulk", help="Update tags for multiple documents")
    update_tags_parser.add_argument("--doc-ids", "-d", nargs="+", help="Document IDs to update")
    update_tags_parser.add_argument("--existing-tags", "-e", nargs="+", help="Find documents with these tags")
    update_tags_parser.add_argument("--add", "-a", nargs="+", help="Tags to add")
    update_tags_parser.add_argument("--remove", "-r", nargs="+", help="Tags to remove")
    update_tags_parser.add_argument("--replace", "-p", nargs="+", help="Replace all tags with these")

    # Bulk export command
    export_bulk_parser = subparsers.add_parser("export-bulk", help="Export document metadata")
    export_bulk_parser.add_argument("--doc-ids", "-d", nargs="+", help="Document IDs to export")
    export_bulk_parser.add_argument("--tags", "-g", nargs="+", help="Export documents with these tags")
    export_bulk_parser.add_argument("--format", "-f", choices=["json", "csv", "markdown"], default="json", help="Export format (default: json)")
    export_bulk_parser.add_argument("--output", "-o", help="Output file (default: stdout)")

    # Summarize command
    summarize_parser = subparsers.add_parser("summarize", help="Generate AI summary of a document")
    summarize_parser.add_argument("doc_id", help="Document ID to summarize")
    summarize_parser.add_argument("--type", "-t", choices=["brief", "detailed", "bullet"], default="brief", help="Summary type (default: brief)")
    summarize_parser.add_argument("--force", "-f", action="store_true", help="Force regeneration even if cached")

    # Summarize all command
    summarize_all_parser = subparsers.add_parser("summarize-all", help="Generate summaries for all documents")
    summarize_all_parser.add_argument("--types", "-t", nargs="+", choices=["brief", "detailed", "bullet"], default=["brief"], help="Summary types (default: brief)")
    summarize_all_parser.add_argument("--force", "-f", action="store_true", help="Force regeneration for all")
    summarize_all_parser.add_argument("--max", "-m", type=int, help="Max documents to process")

    # Extract entities command
    extract_entities_parser = subparsers.add_parser("extract-entities", help="Extract entities from a document using AI")
    extract_entities_parser.add_argument("doc_id", help="Document ID to extract entities from")
    extract_entities_parser.add_argument("--confidence", "-c", type=float, default=0.6, help="Minimum confidence threshold (0.0-1.0, default: 0.6)")
    extract_entities_parser.add_argument("--force", "-f", action="store_true", help="Force re-extraction even if entities exist")

    # Extract all entities command
    extract_all_parser = subparsers.add_parser("extract-all-entities", help="Extract entities from all documents")
    extract_all_parser.add_argument("--confidence", "-c", type=float, default=0.6, help="Minimum confidence threshold (0.0-1.0, default: 0.6)")
    extract_all_parser.add_argument("--force", "-f", action="store_true", help="Force re-extraction for all documents")
    extract_all_parser.add_argument("--max", "-m", type=int, help="Max documents to process")
    extract_all_parser.add_argument("--no-skip", action="store_true", help="Don't skip documents that already have entities")

    # Search entities command
    search_entities_parser = subparsers.add_parser("search-entity", help="Search for entities across all documents")
    search_entities_parser.add_argument("query", help="Entity to search for (e.g., 'VIC-II', '$D000')")
    search_entities_parser.add_argument("--type", "-t", choices=["hardware", "memory_address", "instruction", "person", "company", "product", "concept"], help="Filter by entity type")
    search_entities_parser.add_argument("--confidence", "-c", type=float, default=0.0, help="Minimum confidence (0.0-1.0, default: 0.0)")
    search_entities_parser.add_argument("--max", "-m", type=int, default=20, help="Max results (default: 20)")

    # Entity stats command
    entity_stats_parser = subparsers.add_parser("entity-stats", help="Show entity extraction statistics")
    entity_stats_parser.add_argument("--type", "-t", choices=["hardware", "memory_address", "instruction", "person", "company", "product", "concept"], help="Filter by entity type")

    # Extract entity relationships command
    extract_rels_parser = subparsers.add_parser("extract-relationships", help="Extract entity co-occurrence relationships from a document")
    extract_rels_parser.add_argument("doc_id", help="Document ID to extract relationships from")
    extract_rels_parser.add_argument("--confidence", "-c", type=float, default=0.6, help="Minimum confidence threshold for entities (0.0-1.0, default: 0.6)")

    # Extract all relationships command
    extract_all_rels_parser = subparsers.add_parser("extract-all-relationships", help="Extract entity relationships from all documents")
    extract_all_rels_parser.add_argument("--confidence", "-c", type=float, default=0.6, help="Minimum confidence threshold (0.0-1.0, default: 0.6)")
    extract_all_rels_parser.add_argument("--max", "-m", type=int, help="Max documents to process")

    # Show relationships command
    show_rels_parser = subparsers.add_parser("show-relationships", help="Show entities related to a specific entity")
    show_rels_parser.add_argument("entity", help="Entity to find relationships for (e.g., 'VIC-II', 'SID')")
    show_rels_parser.add_argument("--min-strength", "-s", type=float, default=0.0, help="Minimum relationship strength (0.0-1.0, default: 0.0)")
    show_rels_parser.add_argument("--max", "-m", type=int, default=20, help="Max results (default: 20)")

    # Search entity pair command
    search_pair_parser = subparsers.add_parser("search-pair", help="Find documents containing both entities")
    search_pair_parser.add_argument("entity1", help="First entity")
    search_pair_parser.add_argument("entity2", help="Second entity")
    search_pair_parser.add_argument("--max", "-m", type=int, default=10, help="Max documents (default: 10)")

    # Translate natural language query command
    translate_parser = subparsers.add_parser("translate-query", help="Translate natural language query to structured search parameters")
    translate_parser.add_argument("query", help="Natural language query (e.g., 'find sprite info on VIC-II')")
    translate_parser.add_argument("--confidence", "-c", type=float, default=0.7, help="Minimum confidence for entities (0.0-1.0, default: 0.7)")

    # Compare documents command
    compare_parser = subparsers.add_parser("compare-docs", help="Compare two documents side-by-side with similarity scoring")
    compare_parser.add_argument("doc_id_1", help="First document ID")
    compare_parser.add_argument("doc_id_2", help="Second document ID")
    compare_parser.add_argument("--type", "-t", choices=['full', 'metadata', 'content'], default='full', help="Comparison type (default: full)")

    # Export entities command
    export_entities_parser = subparsers.add_parser("export-entities", help="Export entities to CSV or JSON")
    export_entities_parser.add_argument("--format", "-f", choices=['csv', 'json'], default='csv', help="Export format (default: csv)")
    export_entities_parser.add_argument("--output", "-o", help="Output file path (prints to stdout if not specified)")
    export_entities_parser.add_argument("--types", "-t", nargs='+', choices=['hardware', 'memory_address', 'instruction', 'person', 'company', 'product', 'concept'], help="Filter by entity types")
    export_entities_parser.add_argument("--min-confidence", "-c", type=float, default=0.0, help="Minimum confidence (0.0-1.0, default: 0.0)")

    # Export relationships command
    export_rels_parser = subparsers.add_parser("export-relationships", help="Export entity relationships to CSV or JSON")
    export_rels_parser.add_argument("--format", "-f", choices=['csv', 'json'], default='csv', help="Export format (default: csv)")
    export_rels_parser.add_argument("--output", "-o", help="Output file path (prints to stdout if not specified)")
    export_rels_parser.add_argument("--types", "-t", nargs='+', choices=['hardware', 'memory_address', 'instruction', 'person', 'company', 'product', 'concept'], help="Filter by entity types")
    export_rels_parser.add_argument("--min-strength", "-s", type=float, default=0.0, help="Minimum relationship strength (0.0-1.0, default: 0.0)")

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

    elif args.command == "remove-bulk":
        if not args.doc_ids and not args.tags:
            print("Error: Must provide --doc-ids or --tags")
            sys.exit(1)

        try:
            results = kb.remove_documents_bulk(doc_ids=args.doc_ids, tags=args.tags)

            print(f"Bulk Remove Results:")
            print(f"  Removed: {len(results['removed'])} documents")
            print(f"  Failed: {len(results['failed'])} documents")

            if results['removed']:
                print("\nRemoved documents:")
                for doc_id in results['removed']:
                    print(f"  - {doc_id}")

            if results['failed']:
                print("\nFailed:")
                for failure in results['failed']:
                    print(f"  - {failure['doc_id']}: {failure['error']}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "update-tags-bulk":
        if not args.doc_ids and not args.existing_tags:
            print("Error: Must provide --doc-ids or --existing-tags")
            sys.exit(1)

        if not args.add and not args.remove and not args.replace:
            print("Error: Must provide --add, --remove, or --replace")
            sys.exit(1)

        try:
            results = kb.update_tags_bulk(
                doc_ids=args.doc_ids,
                existing_tags=args.existing_tags,
                add_tags=args.add,
                remove_tags=args.remove,
                replace_tags=args.replace
            )

            print(f"Bulk Tag Update Results:")
            print(f"  Updated: {len(results['updated'])} documents")
            print(f"  Failed: {len(results['failed'])} documents")

            if results['updated']:
                print("\nUpdated documents:")
                for update in results['updated']:
                    print(f"  - {update['doc_id']}")
                    print(f"    Old tags: {', '.join(update['old_tags']) if update['old_tags'] else 'None'}")
                    print(f"    New tags: {', '.join(update['new_tags']) if update['new_tags'] else 'None'}")

            if results['failed']:
                print("\nFailed:")
                for failure in results['failed']:
                    print(f"  - {failure['doc_id']}: {failure['error']}")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "export-bulk":
        try:
            export_data = kb.export_documents_bulk(
                doc_ids=args.doc_ids,
                tags=args.tags,
                format=args.format
            )

            if args.output:
                # Write to file
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(export_data)
                print(f"Exported to: {args.output}")
            else:
                # Print to stdout
                print(export_data)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "summarize":
        try:
            if args.doc_id not in kb.documents:
                print(f"Error: Document not found: {args.doc_id}")
                sys.exit(1)

            doc = kb.documents[args.doc_id]
            print(f"Generating {args.type} summary for: {doc.title}\n")

            summary = kb.generate_summary(
                args.doc_id,
                summary_type=args.type,
                force_regenerate=args.force
            )

            print(f"=== {args.type.upper()} SUMMARY ===\n")
            print(summary)
            print(f"\n=== END SUMMARY ===")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "summarize-all":
        try:
            print(f"Generating summaries for all documents (types: {', '.join(args.types)})\n")

            results = kb.generate_summary_all(
                summary_types=args.types,
                force_regenerate=args.force,
                max_docs=args.max
            )

            print(f"\n✓ Summarization Complete!\n")
            print(f"Statistics:")
            print(f"  Documents processed: {results['processed']}")
            print(f"  Documents failed: {results['failed']}")
            print(f"  Total summaries: {results['total_summaries']}")
            print(f"  By type:")
            for summary_type, count in results['by_type'].items():
                print(f"    - {summary_type}: {count}")

            if results['failed'] > 0:
                print(f"\nWarning: {results['failed']} documents failed. Check logs for details.")
        except Exception as e:
            print(f"Error: {e}")
            print(f"\nNote: Summarization requires LLM configuration.")
            print(f"Set LLM_PROVIDER and ANTHROPIC_API_KEY or OPENAI_API_KEY")
            sys.exit(1)

    elif args.command == "extract-entities":
        try:
            if args.doc_id not in kb.documents:
                print(f"Error: Document not found: {args.doc_id}")
                sys.exit(1)

            doc = kb.documents[args.doc_id]
            print(f"Extracting entities from: {doc.title}")
            print(f"Confidence threshold: {args.confidence}\n")

            result = kb.extract_entities(
                args.doc_id,
                confidence_threshold=args.confidence,
                force_regenerate=args.force
            )

            print(f"[OK] Extraction Complete!\n")
            print(f"Document: {result['doc_title']}")
            print(f"Total entities: {result['entity_count']}\n")

            if result['entities']:
                print(f"Entities by type:")
                for entity_type in sorted(result['types'].keys()):
                    print(f"\n{entity_type.upper().replace('_', ' ')} ({result['types'][entity_type]}):")
                    entities_of_type = [e for e in result['entities'] if e['entity_type'] == entity_type]
                    for entity in entities_of_type[:10]:  # Show first 10 per type
                        print(f"  - {entity['entity_text']} (confidence: {entity['confidence']:.2f}", end="")
                        if entity.get('occurrence_count', 1) > 1:
                            print(f", {entity['occurrence_count']}x", end="")
                        print(")")
                    if len(entities_of_type) > 10:
                        print(f"  ... and {len(entities_of_type) - 10} more")
            else:
                print("No entities found with the current confidence threshold.")
        except Exception as e:
            print(f"Error: {e}")
            print(f"\nNote: Entity extraction requires LLM configuration.")
            print(f"Set LLM_PROVIDER and ANTHROPIC_API_KEY or OPENAI_API_KEY")
            sys.exit(1)

    elif args.command == "extract-all-entities":
        try:
            print(f"Extracting entities from all documents")
            print(f"Confidence threshold: {args.confidence}\n")

            results = kb.extract_entities_bulk(
                confidence_threshold=args.confidence,
                force_regenerate=args.force,
                max_docs=args.max,
                skip_existing=not args.no_skip
            )

            print(f"\n[OK] Bulk Extraction Complete!\n")
            print(f"Statistics:")
            print(f"  Documents processed: {results['processed']}")
            print(f"  Documents skipped: {results['skipped']}")
            print(f"  Documents failed: {results['failed']}")
            print(f"  Total entities: {results['total_entities']}")

            if results['by_type']:
                print(f"\n  Entities by type:")
                for entity_type, count in sorted(results['by_type'].items(), key=lambda x: x[1], reverse=True):
                    print(f"    - {entity_type.replace('_', ' ')}: {count}")

            if results['failed'] > 0:
                print(f"\nWarning: {results['failed']} documents failed. Check logs for details.")
        except Exception as e:
            print(f"Error: {e}")
            print(f"\nNote: Entity extraction requires LLM configuration.")
            print(f"Set LLM_PROVIDER and ANTHROPIC_API_KEY or OPENAI_API_KEY")
            sys.exit(1)

    elif args.command == "search-entity":
        try:
            entity_types = [args.type] if args.type else None

            results = kb.search_entities(
                args.query,
                entity_types=entity_types,
                min_confidence=args.confidence,
                max_results=args.max
            )

            print(f"Entity search results for: {results['query']}")
            print(f"Total matches: {results['total_matches']}")
            print(f"Documents found: {len(results['documents'])}\n")

            if results['documents']:
                for i, doc in enumerate(results['documents'], 1):
                    print(f"{i}. {doc['doc_title']} ({doc['doc_id']})")
                    print(f"   Matches: {doc['match_count']}")
                    for match in doc['matches'][:3]:  # Show first 3 per doc
                        print(f"   - {match['entity_text']} ({match['entity_type']}, conf: {match['confidence']:.2f}", end="")
                        if match.get('occurrence_count', 1) > 1:
                            print(f", {match['occurrence_count']}x", end="")
                        print(")")
                    if doc['match_count'] > 3:
                        print(f"   ... and {doc['match_count'] - 3} more matches")
                    print()
            else:
                print("No entities found matching your query.")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "entity-stats":
        try:
            results = kb.get_entity_stats(entity_type=args.type)

            print(f"Entity Extraction Statistics")
            if args.type:
                print(f"Type filter: {args.type}")
            print()

            print(f"Total entities: {results['total_entities']}")
            print(f"Documents with entities: {results['total_documents_with_entities']}")

            if results['by_type']:
                print(f"\nEntities by type:")
                for entity_type, count in sorted(results['by_type'].items(), key=lambda x: x[1], reverse=True):
                    print(f"  - {entity_type.replace('_', ' ')}: {count}")

            if results['top_entities']:
                print(f"\nTop 10 entities (by document count):")
                for i, entity in enumerate(results['top_entities'][:10], 1):
                    print(f"{i}. {entity['entity_text']} ({entity['entity_type']})")
                    print(f"   - Found in {entity['document_count']} document(s)")
                    print(f"   - Total occurrences: {entity['total_occurrences']}")
                    print(f"   - Avg confidence: {entity['avg_confidence']:.2f}")

            if results['documents_with_most_entities']:
                print(f"\nDocuments with most entities:")
                for i, doc in enumerate(results['documents_with_most_entities'], 1):
                    print(f"{i}. {doc['doc_title']}: {doc['entity_count']} entities")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "extract-relationships":
        try:
            if args.doc_id not in kb.documents:
                print(f"Error: Document not found: {args.doc_id}")
                sys.exit(1)

            doc = kb.documents[args.doc_id]
            print(f"Extracting entity relationships from: {doc.title}")
            print(f"Confidence threshold: {args.confidence}\n")

            result = kb.extract_entity_relationships(doc_id=args.doc_id, min_confidence=args.confidence)

            print(f"[OK] Relationship extraction complete!")
            print(f"Relationships found: {result['relationship_count']}\n")

            if result['relationships']:
                print(f"Top relationships (by strength):\n")
                for i, rel in enumerate(result['relationships'][:15], 1):
                    print(f"{i}. {rel['entity1']} ({rel['entity1_type']}) <-> {rel['entity2']} ({rel['entity2_type']})")
                    print(f"   Strength: {rel['strength']:.2f}")
                    if rel.get('context'):
                        ctx = rel['context'][:100] + "..." if len(rel['context']) > 100 else rel['context']
                        print(f"   Context: {ctx}")
                    print()

                if len(result['relationships']) > 15:
                    print(f"... and {len(result['relationships']) - 15} more relationships")
            else:
                print("No relationships found. Ensure the document has extracted entities first.")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "extract-all-relationships":
        try:
            print(f"Extracting entity relationships from all documents with entities...")
            print(f"Confidence threshold: {args.confidence}\n")

            result = kb.extract_relationships_bulk(
                min_confidence=args.confidence,
                max_docs=args.max
            )

            print(f"\n[OK] Bulk relationship extraction complete!\n")
            print(f"Processed: {result['processed']}")
            print(f"Failed: {result['failed']}")
            print(f"Total relationships: {result['total_relationships']}")
            print(f"Avg per document: {result['total_relationships'] / max(result['processed'], 1):.1f}")

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "show-relationships":
        try:
            relationships = kb.get_entity_relationships(
                entity_text=args.entity,
                min_strength=args.min_strength,
                max_results=args.max
            )

            if not relationships:
                print(f"No relationships found for entity '{args.entity}'")
                sys.exit(0)

            print(f"Entities related to '{args.entity}' ({len(relationships)} found):\n")

            for i, rel in enumerate(relationships, 1):
                print(f"{i}. {rel['related_entity']} ({rel['related_type']})")
                print(f"   Strength: {rel['strength']:.2f} | Documents: {rel['doc_count']}")
                if rel.get('context_sample'):
                    ctx = rel['context_sample'][:100] + "..." if len(rel['context_sample']) > 100 else rel['context_sample']
                    print(f"   Context: {ctx}")
                print()

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "search-pair":
        try:
            results = kb.search_by_entity_pair(
                entity1=args.entity1,
                entity2=args.entity2,
                max_results=args.max
            )

            if not results:
                print(f"No documents found containing both '{args.entity1}' and '{args.entity2}'")
                sys.exit(0)

            print(f"Documents containing both '{args.entity1}' AND '{args.entity2}' ({len(results)} found):\n")

            for i, doc in enumerate(results, 1):
                print(f"{i}. {doc['title']}")
                print(f"   '{args.entity1}': {doc['entity1_count']} | '{args.entity2}': {doc['entity2_count']}")
                print(f"   Doc ID: {doc['doc_id']}")

                if doc.get('contexts'):
                    print(f"   Context snippets:")
                    for j, ctx in enumerate(doc['contexts'][:2], 1):
                        ctx_short = ctx[:120] + "..." if len(ctx) > 120 else ctx
                        print(f"   {j}. {ctx_short}")
                print()

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "translate-query":
        try:
            result = kb.translate_nl_query(
                query=args.query,
                confidence_threshold=args.confidence
            )

            print("=" * 70)
            print("NATURAL LANGUAGE QUERY TRANSLATION")
            print("=" * 70)
            print(f"\nOriginal Query: {result['original_query']}")
            print(f"Suggested Query: {result['suggested_query']}")
            print(f"Search Mode: {result['search_mode']}")
            print(f"Confidence: {result['confidence']:.2f}")

            if result.get('intent'):
                print(f"Intent: {result['intent']}")

            if result.get('search_terms'):
                print(f"\nSearch Terms: {', '.join(result['search_terms'])}")

            if result.get('entities_found'):
                print(f"\nEntities Detected ({len(result['entities_found'])} found):")
                for entity in result['entities_found'][:15]:  # Show top 15
                    source = "regex" if entity['source'] == 'regex' else "AI"
                    print(f"  - {entity['text']} ({entity['type']}) [confidence: {entity['confidence']:.2f}, source: {source}]")
                if len(result['entities_found']) > 15:
                    print(f"  ... and {len(result['entities_found']) - 15} more")

            if result.get('facet_filters'):
                print(f"\nFacet Filters:")
                for facet_type, values in result['facet_filters'].items():
                    print(f"  - {facet_type}: {', '.join(values)}")

            if result.get('fallback'):
                print("\n⚠️  WARNING: LLM unavailable, using fallback keyword extraction")

            print("\n" + "=" * 70)
            print("SUGGESTED NEXT STEPS")
            print("=" * 70)

            if result['search_mode'] == 'keyword':
                print(f"Run: python cli.py search \"{result['suggested_query']}\"")
            elif result['search_mode'] == 'semantic':
                print("Semantic search requires the MCP server or Python API")
                print(f"Example: kb.semantic_search(\"{result['suggested_query']}\")")
            elif result['search_mode'] == 'hybrid':
                print("Hybrid search requires the MCP server or Python API")
                print(f"Example: kb.hybrid_search(\"{result['suggested_query']}\")")

        except ValueError as e:
            print(f"Error: {e}")
            print("\nMake sure LLM_PROVIDER and appropriate API key are configured.")
            sys.exit(1)
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "compare-docs":
        try:
            result = kb.compare_documents(
                doc_id_1=args.doc_id_1,
                doc_id_2=args.doc_id_2,
                comparison_type=args.type
            )

            print("=" * 70)
            print("DOCUMENT COMPARISON")
            print("=" * 70)

            # Similarity Score
            print(f"\nSimilarity Score: {result['similarity_score']:.1%}")
            print(f"Summary: {result['summary']}")

            # Metadata Comparison
            print("\n" + "-" * 70)
            print("METADATA COMPARISON")
            print("-" * 70)
            meta = result['metadata_diff']
            print(f"Title:       {meta['title'][0]}")
            print(f"             {meta['title'][1]}")
            print(f"Filename:    {meta['filename'][0]}")
            print(f"             {meta['filename'][1]}")
            print(f"File Type:   {meta['file_type'][0]} vs {meta['file_type'][1]}")
            print(f"Pages:       {meta['total_pages'][0]} vs {meta['total_pages'][1]}")
            print(f"Chunks:      {result['chunk_count'][0]} vs {result['chunk_count'][1]}")

            # Tags Comparison
            if meta['tags']:
                print(f"\nTags:")
                if meta['tags']['common']:
                    print(f"  Common: {', '.join(meta['tags']['common'])}")
                if meta['tags']['only_in_doc1']:
                    print(f"  Only in Doc 1: {', '.join(meta['tags']['only_in_doc1'])}")
                if meta['tags']['only_in_doc2']:
                    print(f"  Only in Doc 2: {', '.join(meta['tags']['only_in_doc2'])}")
            else:
                print(f"\nTags: (none)")

            # Entity Comparison
            if result.get('entity_comparison'):
                print("\n" + "-" * 70)
                print("ENTITY COMPARISON")
                print("-" * 70)
                ent = result['entity_comparison']
                print(f"Total Entities: {ent['total_doc1']} vs {ent['total_doc2']}")
                print(f"Common Entities: {len(ent['common_entities'])}")
                print(f"Unique to Doc 1: {len(ent['unique_to_doc1'])}")
                print(f"Unique to Doc 2: {len(ent['unique_to_doc2'])}")

                if ent['common_entities']:
                    print(f"\nCommon Entities (showing first 10):")
                    for entity_text, entity_type in ent['common_entities'][:10]:
                        print(f"  - {entity_text} ({entity_type})")
                    if len(ent['common_entities']) > 10:
                        print(f"  ... and {len(ent['common_entities']) - 10} more")

            # Content Diff Preview
            if result.get('content_diff') and len(result['content_diff']) > 0:
                print("\n" + "-" * 70)
                print("CONTENT DIFF (first 20 lines)")
                print("-" * 70)
                for line in result['content_diff'][:20]:
                    # Color code diff lines
                    if line.startswith('+'):
                        print(f"\033[92m{line}\033[0m")  # Green for additions
                    elif line.startswith('-'):
                        print(f"\033[91m{line}\033[0m")  # Red for deletions
                    elif line.startswith('@'):
                        print(f"\033[94m{line}\033[0m")  # Blue for context
                    else:
                        print(line)

                if len(result['content_diff']) > 20:
                    print(f"\n... {len(result['content_diff']) - 20} more diff lines")

            print("\n" + "=" * 70)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "export-entities":
        try:
            result = kb.export_entities(
                format=args.format,
                entity_types=args.types,
                min_confidence=args.min_confidence,
                output_path=args.output
            )

            if args.output:
                # Count entities
                if args.format == 'csv':
                    count = result.count('\n') - 1
                else:
                    import json
                    count = len(json.loads(result))

                print(f"Exported {count} entities to {args.output}")
                print(f"Format: {args.format.upper()}")
                print(f"Min Confidence: {args.min_confidence:.2f}")
                if args.types:
                    print(f"Filtered Types: {', '.join(args.types)}")
            else:
                # Print to stdout
                print(result)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)

    elif args.command == "export-relationships":
        try:
            result = kb.export_relationships(
                format=args.format,
                min_strength=args.min_strength,
                entity_types=args.types,
                output_path=args.output
            )

            if args.output:
                # Count relationships
                if args.format == 'csv':
                    count = result.count('\n') - 1
                else:
                    import json
                    count = len(json.loads(result))

                print(f"Exported {count} relationships to {args.output}")
                print(f"Format: {args.format.upper()}")
                print(f"Min Strength: {args.min_strength:.2f}")
                if args.types:
                    print(f"Filtered Types: {', '.join(args.types)}")
            else:
                # Print to stdout
                print(result)

        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
