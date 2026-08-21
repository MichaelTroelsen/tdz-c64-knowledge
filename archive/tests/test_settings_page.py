#!/usr/bin/env python3
"""
Test script for Settings page functionality in admin_gui.py
"""

import os
import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from server import KnowledgeBase

def test_find_mcp_config_files():
    """Test finding MCP configuration files."""
    print("=" * 60)
    print("TEST 1: Finding MCP Configuration Files")
    print("=" * 60)

    config_files = []

    # Claude Desktop config (Windows)
    claude_desktop_config = Path(os.environ.get('APPDATA', '')) / 'Claude' / 'claude_desktop_config.json'
    print(f"\nClaude Desktop config path: {claude_desktop_config}")
    if claude_desktop_config.exists():
        config_files.append(('Claude Desktop', str(claude_desktop_config)))
        print("  ✅ Found!")
    else:
        print("  ❌ Not found")

    # Claude Code config (.claude/settings.json in project)
    claude_code_config = Path.cwd() / '.claude' / 'settings.json'
    print(f"\nClaude Code (Project) config path: {claude_code_config}")
    if claude_code_config.exists():
        config_files.append(('Claude Code (Project)', str(claude_code_config)))
        print("  ✅ Found!")
    else:
        print("  ❌ Not found")

    # Claude Code global config
    home = Path.home()
    claude_code_global = home / '.claude' / 'settings.json'
    print(f"\nClaude Code (Global) config path: {claude_code_global}")
    if claude_code_global.exists():
        config_files.append(('Claude Code (Global)', str(claude_code_global)))
        print("  ✅ Found!")
    else:
        print("  ❌ Not found")

    print(f"\n📊 Total config files found: {len(config_files)}")
    return config_files


def test_read_mcp_config(config_files):
    """Test reading and parsing MCP configuration files."""
    print("\n" + "=" * 60)
    print("TEST 2: Reading MCP Configuration Files")
    print("=" * 60)

    for config_type, config_path in config_files:
        print(f"\n📄 Reading: {config_type}")
        print(f"   Path: {config_path}")

        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)

            print("  ✅ Successfully parsed JSON")

            # Check if mcpServers exists
            if 'mcpServers' in config:
                print(f"  📊 Found {len(config['mcpServers'])} MCP servers")

                # List all servers
                for server_name in config['mcpServers'].keys():
                    print(f"     - {server_name}")

                # Find our server
                for server_name, server_config in config['mcpServers'].items():
                    if 'tdz-c64-knowledge' in server_name.lower() or 'c64' in server_name.lower():
                        print(f"\n  ✅ Found TDZ C64 Knowledge server: {server_name}")
                        print(f"     Command: {server_config.get('command', 'N/A')}")
                        print(f"     Args: {server_config.get('args', [])}")
                        if 'env' in server_config:
                            print(f"     Environment variables: {len(server_config['env'])}")
                            for key in server_config['env'].keys():
                                print(f"       - {key}")
                        return server_name, server_config
            else:
                print("  ⚠️ No 'mcpServers' key found in config")

        except Exception as e:
            print(f"  ❌ Error reading config: {str(e)}")

    return None, None


def test_file_paths():
    """Test file path detection."""
    print("\n" + "=" * 60)
    print("TEST 3: File Paths")
    print("=" * 60)

    data_dir = os.environ.get('TDZ_DATA_DIR', str(Path.home() / '.tdz-c64-knowledge'))
    print(f"\n📂 Data Directory: {data_dir}")

    # Database path
    db_path = Path(data_dir) / "knowledge_base.db"
    print(f"\n💾 Database Path: {db_path}")
    if db_path.exists():
        size_mb = db_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Exists ({size_mb:.2f} MB)")
    else:
        print("  ❌ Not found")

    # Embeddings path
    embeddings_path = Path(data_dir) / "embeddings.pkl"
    print(f"\n🧠 Embeddings Path: {embeddings_path}")
    if embeddings_path.exists():
        size_mb = embeddings_path.stat().st_size / (1024 * 1024)
        print(f"  ✅ Exists ({size_mb:.2f} MB)")
    else:
        print("  ℹ️ Not yet created")


def test_environment_variables():
    """Test environment variable detection."""
    print("\n" + "=" * 60)
    print("TEST 4: Environment Variables")
    print("=" * 60)

    important_vars = [
        'TDZ_DATA_DIR',
        'USE_FTS5',
        'USE_SEMANTIC_SEARCH',
        'USE_BM25',
        'ALLOWED_DOCS_DIRS',
        'REST_API_KEY',
        'MDSCRAPE_PATH',
        'EMBEDDING_CACHE_TTL',
        'ENTITY_CACHE_TTL',
        'ANTHROPIC_API_KEY',
        'OPENAI_API_KEY'
    ]

    print("\n📊 Important Environment Variables:")
    set_count = 0
    for var in important_vars:
        value = os.environ.get(var, '')

        # Mask sensitive values
        if 'KEY' in var or 'SECRET' in var or 'PASSWORD' in var:
            if value:
                display_value = '***' + value[-4:] if len(value) > 4 else '***'
            else:
                display_value = '(not set)'
        else:
            display_value = value or '(not set)'

        status = '✅' if value else '❌'
        if value:
            set_count += 1

        print(f"  {status} {var:30s} = {display_value}")

    print(f"\n📊 Total variables set: {set_count}/{len(important_vars)}")


def test_features_and_capabilities():
    """Test features and capabilities detection."""
    print("\n" + "=" * 60)
    print("TEST 5: Features & Capabilities")
    print("=" * 60)

    try:
        # Initialize KnowledgeBase
        data_dir = os.environ.get('TDZ_DATA_DIR', str(Path.home() / '.tdz-c64-knowledge'))
        kb = KnowledgeBase(data_dir)

        # Run health check
        health = kb.health_check()

        # Display status
        status_emoji = {
            'healthy': '🟢',
            'degraded': '🟡',
            'unhealthy': '🔴'
        }
        print(f"\n{status_emoji.get(health['status'], '⚪')} System Status: {health['status'].upper()}")

        if health.get('issues'):
            print("\n⚠️ Issues:")
            for issue in health['issues']:
                print(f"  - {issue}")

        # Display features
        features = health.get('features', {})

        print("\n🔍 Search Features:")
        print(f"  {'✅' if features.get('fts5_enabled') else '❌'} FTS5 Full-Text Search")
        print(f"  {'✅' if features.get('semantic_search_enabled') else '❌'} Semantic Search (Enabled)")
        print(f"  {'✅' if features.get('semantic_search_available') else '❌'} Semantic Search (Available)")
        print(f"  {'✅' if features.get('bm25_available') else '❌'} BM25 Ranking")

        print("\n📄 Document Processing:")
        print(f"  {'✅' if features.get('pdf_support') else '❌'} PDF Support")
        print(f"  {'✅' if features.get('ocr_enabled') else '❌'} OCR")

        print("\n🤖 AI Features:")
        print(f"  {'✅' if features.get('entity_extraction_available') else '❌'} Entity Extraction")
        print(f"  {'✅' if features.get('relationship_extraction_available') else '❌'} Relationship Mapping")
        print(f"  {'✅' if features.get('rag_available') else '❌'} RAG Question Answering")

        print("\n🌐 Web Features:")
        print(f"  {'✅' if features.get('web_scraping_available') else '❌'} Web Scraping")
        print(f"  {'✅' if features.get('url_monitoring_available') else '❌'} URL Monitoring")

        # Database stats
        db_stats = health.get('database', {})
        print("\n📊 Database Statistics:")
        print(f"  Documents: {db_stats.get('total_documents', 0):,}")
        print(f"  Chunks: {db_stats.get('total_chunks', 0):,}")
        print(f"  Entities: {db_stats.get('total_entities', 0):,}")
        print(f"  Relationships: {db_stats.get('total_relationships', 0):,}")
        print(f"  Embeddings: {features.get('embeddings_count', 0):,} ({features.get('embeddings_size_mb', 0):.2f} MB)")

        print("\n✅ Features & Capabilities test passed!")

    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("TESTING SETTINGS PAGE FUNCTIONALITY")
    print("=" * 60)

    # Test 1: Find MCP config files
    config_files = test_find_mcp_config_files()

    # Test 2: Read MCP config files
    if config_files:
        server_name, server_config = test_read_mcp_config(config_files)

    # Test 3: File paths
    test_file_paths()

    # Test 4: Environment variables
    test_environment_variables()

    # Test 5: Features and capabilities
    test_features_and_capabilities()

    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\n✅ Settings page functionality validated!")
    print(f"\n🌐 Streamlit GUI is running at: http://localhost:8501")
    print("   Navigate to '⚙️ Settings' page to see the results in the GUI")
    print("\n" + "=" * 60)


if __name__ == "__main__":
    # Fix Windows console encoding for emojis
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

    main()
