#!/usr/bin/env python3
"""
Daily URL Monitoring Script

Performs quick checks of URL-sourced documents to detect content changes.
Designed to run daily via Task Scheduler or cron.

Usage:
    python monitor_daily.py [--auto-rescrape] [--notify] [--output FILENAME]

Options:
    --auto-rescrape    Automatically re-scrape changed documents
    --notify           Send notifications for changes (requires configuration)
    --output FILE      Save results to JSON file (default: url_check_daily_TIMESTAMP.json)
"""

import os
import sys
import argparse
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase


def send_notification(results, config):
    """Send notification about check results (placeholder for future implementation)."""
    # TODO: Implement email/Slack/webhook notifications
    # For now, just print to stdout
    print("\n" + "=" * 60)
    print("NOTIFICATION SUMMARY")
    print("=" * 60)

    if results['changed']:
        print(f"\n[!] {len(results['changed'])} documents have updates available:")
        for doc in results['changed'][:5]:
            print(f"  - {doc['title']}")
            print(f"    URL: {doc['url']}")
        if len(results['changed']) > 5:
            print(f"  ... and {len(results['changed']) - 5} more")

    if results['failed']:
        print(f"\n[!] {len(results['failed'])} checks failed:")
        for doc in results['failed'][:3]:
            print(f"  - {doc['title']}: {doc['error']}")
        if len(results['failed']) > 3:
            print(f"  ... and {len(results['failed']) - 3} more")


def main():
    parser = argparse.ArgumentParser(
        description="Daily URL monitoring - quick check for content changes"
    )
    parser.add_argument(
        '--auto-rescrape',
        action='store_true',
        help='Automatically re-scrape changed documents'
    )
    parser.add_argument(
        '--notify',
        action='store_true',
        help='Send notifications for changes'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Save results to JSON file (default: url_check_daily_TIMESTAMP.json)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='monitor_config.json',
        help='Configuration file path (default: monitor_config.json)'
    )

    args = parser.parse_args()

    # Load configuration
    config = {}
    if os.path.exists(args.config):
        try:
            with open(args.config, 'r') as f:
                config = json.load(f)
                print(f"[OK] Loaded configuration from {args.config}")
        except Exception as e:
            print(f"[WARNING] Failed to load config: {e}")

    # Initialize knowledge base
    data_dir = os.path.expanduser(config.get('data_dir', '~/.tdz-c64-knowledge'))
    print(f"\n[*] Initializing knowledge base at {data_dir}")

    try:
        kb = KnowledgeBase(data_dir)
    except Exception as e:
        print(f"[ERROR] Failed to initialize knowledge base: {e}")
        return 1

    # Count URL-sourced documents
    url_docs = [d for d in kb.documents.values() if d.source_url]
    print(f"[*] Found {len(url_docs)} URL-sourced documents to check")

    if not url_docs:
        print("[*] No URL-sourced documents found. Nothing to check.")
        return 0

    # Run quick check
    print(f"\n[*] Running quick check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("[*] Mode: Quick (Last-Modified headers only)")
    print(f"[*] Auto-rescrape: {'Enabled' if args.auto_rescrape else 'Disabled'}")

    try:
        results = kb.check_url_updates(
            auto_rescrape=args.auto_rescrape,
            check_structure=False  # Quick mode - no structure discovery
        )
    except Exception as e:
        print(f"\n[ERROR] Check failed: {e}")
        return 1

    # Display results
    print("\n" + "=" * 60)
    print("CHECK RESULTS")
    print("=" * 60)
    print(f"[OK] Unchanged:     {len(results['unchanged']):3d} documents")
    print(f"[!]  Changed:       {len(results['changed']):3d} documents")
    print(f"[X]  Failed:        {len(results['failed']):3d} checks")

    if args.auto_rescrape and results['rescraped']:
        print(f"[OK] Auto-rescraped: {len(results['rescraped']):3d} documents")

    # Show changed documents
    if results['changed']:
        print("\n" + "-" * 60)
        print("CHANGED DOCUMENTS:")
        print("-" * 60)
        for doc in results['changed']:
            print(f"\n[!] {doc['title']}")
            print(f"    URL: {doc['url']}")
            print(f"    Doc ID: {doc['doc_id'][:12]}...")
            if 'last_modified' in doc:
                print(f"    Last Modified: {doc['last_modified']}")
            if 'scraped_date' in doc:
                print(f"    Scraped: {doc['scraped_date']}")
            if 'reason' in doc:
                print(f"    Reason: {doc['reason']}")

    # Show failed checks
    if results['failed']:
        print("\n" + "-" * 60)
        print("FAILED CHECKS:")
        print("-" * 60)
        for doc in results['failed']:
            print(f"\n[X] {doc['title']}")
            print(f"    URL: {doc['url']}")
            print(f"    Error: {doc['error']}")

    # Save results to file
    output_file = args.output
    if output_file is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = f"url_check_daily_{timestamp}.json"

    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[OK] Results saved to {output_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save results: {e}")

    # Send notifications if requested
    if args.notify and (results['changed'] or results['failed']):
        send_notification(results, config)

    # Return exit code
    print("\n" + "=" * 60)
    if results['changed']:
        print(f"[!] COMPLETE: {len(results['changed'])} documents need attention")
        return 2  # Exit code 2: changes detected
    elif results['failed']:
        print(f"[!] COMPLETE: {len(results['failed'])} checks failed")
        return 3  # Exit code 3: some checks failed
    else:
        print("[OK] COMPLETE: All documents up to date")
        return 0  # Exit code 0: success, no changes


if __name__ == "__main__":
    sys.exit(main())
