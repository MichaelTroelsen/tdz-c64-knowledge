#!/usr/bin/env python3
"""
Weekly URL Monitoring Script

Performs comprehensive checks with structure discovery to find new and missing pages.
Designed to run weekly via Task Scheduler or cron.

Usage:
    python monitor_weekly.py [--auto-rescrape] [--notify] [--output FILENAME]

Options:
    --auto-rescrape    Automatically re-scrape changed documents
    --notify           Send notifications for changes (requires configuration)
    --output FILE      Save results to JSON file (default: url_check_weekly_TIMESTAMP.json)
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
from notifications import NotificationManager


def send_notification(results, config):
    """Send notification about check results via configured channels."""
    # Print summary to stdout
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

    if results['new_pages']:
        print(f"\n[+] {len(results['new_pages'])} new pages discovered:")
        # Group by site
        by_site = {}
        for page in results['new_pages']:
            site = page.get('base_url', 'Unknown')
            if site not in by_site:
                by_site[site] = []
            by_site[site].append(page['url'])

        for site, urls in sorted(by_site.items()):
            print(f"\n  Site: {site}")
            print(f"  New pages: {len(urls)}")
            for url in urls[:3]:
                print(f"    - {url}")
            if len(urls) > 3:
                print(f"    ... and {len(urls) - 3} more")

    if results['missing_pages']:
        print(f"\n[-] {len(results['missing_pages'])} pages missing or inaccessible:")
        for doc in results['missing_pages'][:5]:
            print(f"  - {doc['title']}")
            print(f"    URL: {doc['url']}")
            if 'reason' in doc:
                print(f"    Reason: {doc['reason']}")
        if len(results['missing_pages']) > 5:
            print(f"  ... and {len(results['missing_pages']) - 5} more")

    if results['failed']:
        print(f"\n[X] {len(results['failed'])} checks failed:")
        for doc in results['failed'][:3]:
            print(f"  - {doc['title']}: {doc['error']}")
        if len(results['failed']) > 3:
            print(f"  ... and {len(results['failed']) - 3} more")

    # Send via configured notification channels
    if config.get('notifications', {}).get('enabled'):
        print("\n[*] Sending notifications...")
        try:
            manager = NotificationManager(config)
            status = manager.send_monitoring_alert(results, "weekly")

            for channel, success in status.items():
                if channel != "status":
                    status_str = "✓" if success else "✗"
                    print(f"  {status_str} {channel}")
        except Exception as e:
            print(f"[ERROR] Notification delivery failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Weekly URL monitoring - comprehensive check with structure discovery"
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
        help='Save results to JSON file (default: url_check_weekly_TIMESTAMP.json)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='monitor_config.json',
        help='Configuration file path (default: monitor_config.json)'
    )
    parser.add_argument(
        '--no-structure',
        action='store_true',
        help='Disable structure discovery (faster but won\'t find new/missing pages)'
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

    # Determine check mode
    check_structure = not args.no_structure

    # Run comprehensive check
    print(f"\n[*] Running comprehensive check at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Mode: {'Full (with structure discovery)' if check_structure else 'Quick (Last-Modified only)'}")
    print(f"[*] Auto-rescrape: {'Enabled' if args.auto_rescrape else 'Disabled'}")

    if check_structure:
        print("[*] This may take 10-60 seconds per site...")

    try:
        results = kb.check_url_updates(
            auto_rescrape=args.auto_rescrape,
            check_structure=check_structure
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
    print(f"[+]  New Pages:     {len(results.get('new_pages', [])):3d} discovered")
    print(f"[-]  Missing:       {len(results.get('missing_pages', [])):3d} pages")
    print(f"[X]  Failed:        {len(results['failed']):3d} checks")

    if args.auto_rescrape and results['rescraped']:
        print(f"[OK] Auto-rescraped: {len(results['rescraped']):3d} documents")

    # Show scrape session statistics
    if results.get('scrape_sessions'):
        print("\n" + "-" * 60)
        print("SCRAPE SESSION STATISTICS:")
        print("-" * 60)
        for session in results['scrape_sessions']:
            print(f"\n[*] {session['base_url']}")
            print(f"    Total docs:  {session['docs_count']:3d}")
            print(f"    Unchanged:   {session['unchanged']:3d}")
            print(f"    Changed:     {session['changed']:3d}")
            if check_structure:
                print(f"    New:         {session.get('new', 0):3d}")
                print(f"    Missing:     {session.get('missing', 0):3d}")

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

    # Show new pages
    if results.get('new_pages'):
        print("\n" + "-" * 60)
        print("NEW PAGES DISCOVERED:")
        print("-" * 60)

        # Group by site
        by_site = {}
        for page in results['new_pages']:
            site = page.get('base_url', 'Unknown')
            if site not in by_site:
                by_site[site] = []
            by_site[site].append(page['url'])

        for site, urls in sorted(by_site.items()):
            print(f"\n[+] {site}")
            print(f"    {len(urls)} new pages:")
            for url in urls[:5]:
                print(f"      - {url}")
            if len(urls) > 5:
                print(f"      ... and {len(urls) - 5} more")

    # Show missing pages
    if results.get('missing_pages'):
        print("\n" + "-" * 60)
        print("MISSING PAGES:")
        print("-" * 60)
        for doc in results['missing_pages']:
            print(f"\n[-] {doc['title']}")
            print(f"    URL: {doc['url']}")
            print(f"    Doc ID: {doc['doc_id'][:12]}...")
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
        output_file = f"url_check_weekly_{timestamp}.json"

    try:
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n[OK] Results saved to {output_file}")
    except Exception as e:
        print(f"[WARNING] Failed to save results: {e}")

    # Send notifications if requested
    if args.notify and (results['changed'] or results.get('new_pages') or results.get('missing_pages') or results['failed']):
        send_notification(results, config)

    # Return exit code
    print("\n" + "=" * 60)
    total_issues = len(results['changed']) + len(results.get('new_pages', [])) + len(results.get('missing_pages', []))

    if total_issues > 0:
        print(f"[!] COMPLETE: {total_issues} items need attention")
        return 2  # Exit code 2: changes/issues detected
    elif results['failed']:
        print(f"[!] COMPLETE: {len(results['failed'])} checks failed")
        return 3  # Exit code 3: some checks failed
    else:
        print("[OK] COMPLETE: All documents up to date, no new/missing pages")
        return 0  # Exit code 0: success, no changes


if __name__ == "__main__":
    sys.exit(main())
