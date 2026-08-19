"""Admin GUI page: ⚙️ Settings.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
from pathlib import Path
import json
import os
import pandas as pd
import streamlit as st


def render(kb):
    st.title("⚙️ Settings & Configuration")
    st.markdown("View MCP configuration, runtime settings, and environment variables.")
    def find_mcp_config_files():
        """Find potential MCP configuration files."""
        config_files = []

        # Claude Desktop config (Windows)
        claude_desktop_config = Path(os.environ.get('APPDATA', '')) / 'Claude' / 'claude_desktop_config.json'
        if claude_desktop_config.exists():
            config_files.append(('Claude Desktop', str(claude_desktop_config)))

        # Claude Code config (.claude/settings.json in project)
        claude_code_config = Path.cwd() / '.claude' / 'settings.json'
        if claude_code_config.exists():
            config_files.append(('Claude Code (Project)', str(claude_code_config)))

        # Claude Code global config
        home = Path.home()
        claude_code_global = home / '.claude' / 'settings.json'
        if claude_code_global.exists():
            config_files.append(('Claude Code (Global)', str(claude_code_global)))

        return config_files
    def read_mcp_config(config_path):
        """Read and parse MCP configuration file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except Exception as e:
            return {'error': str(e)}
    def find_server_config(config):
        """Find tdz-c64-knowledge server configuration in MCP config."""
        if 'mcpServers' in config:
            for server_name, server_config in config['mcpServers'].items():
                if 'tdz-c64-knowledge' in server_name.lower() or 'c64' in server_name.lower():
                    return server_name, server_config
        return None, None
    tab1, tab2, tab3, tab4 = st.tabs(["📂 File Paths", "🔧 MCP Configuration", "🌍 Environment Variables", "⚡ Features & Capabilities"])
    with tab1:
        st.subheader("📂 File Paths")

        # Data directory
        st.markdown("**Data Directory:**")
        st.code(st.session_state.data_dir, language="text")

        # Database path
        db_path = Path(st.session_state.data_dir) / "knowledge_base.db"
        st.markdown("**Database Path:**")
        st.code(str(db_path), language="text")
        if db_path.exists():
            size_mb = db_path.stat().st_size / (1024 * 1024)
            st.caption(f"✅ Exists ({size_mb:.2f} MB)")
        else:
            st.caption("❌ Not found")

        # Embeddings path
        embeddings_path = Path(st.session_state.data_dir) / "embeddings.pkl"
        st.markdown("**Embeddings Path:**")
        st.code(str(embeddings_path), language="text")
        if embeddings_path.exists():
            size_mb = embeddings_path.stat().st_size / (1024 * 1024)
            st.caption(f"✅ Exists ({size_mb:.2f} MB)")
        else:
            st.caption("ℹ️ Not yet created (will be generated on first semantic search)")

        # MCP config files
        st.markdown("**MCP Configuration Files:**")
        config_files = find_mcp_config_files()
        if config_files:
            for config_type, config_path in config_files:
                st.code(f"{config_type}: {config_path}", language="text")
                st.caption(f"✅ Found")
        else:
            st.warning("No MCP configuration files found")
            st.caption("Expected locations:\n"
                      "- Claude Desktop: %APPDATA%\\Claude\\claude_desktop_config.json\n"
                      "- Claude Code: .claude/settings.json")
    with tab2:
        st.subheader("🔧 MCP Server Configuration")

        config_files = find_mcp_config_files()

        if not config_files:
            st.warning("No MCP configuration files found")
            st.info("To use this server with Claude Desktop or Claude Code, you need to configure it in the MCP settings.")

            st.markdown("**Example Configuration:**")
            example_config = {
                "mcpServers": {
                    "tdz-c64-knowledge": {
                        "command": "C:\\path\\to\\.venv\\Scripts\\python.exe",
                        "args": ["C:\\path\\to\\server.py"],
                        "env": {
                            "TDZ_DATA_DIR": str(st.session_state.data_dir),
                            "USE_FTS5": "1",
                            "USE_SEMANTIC_SEARCH": "1"
                        }
                    }
                }
            }
            st.json(example_config)
        else:
            # Read and display each config file
            for config_type, config_path in config_files:
                with st.expander(f"📄 {config_type} Configuration", expanded=True):
                    st.code(config_path, language="text")

                    config = read_mcp_config(config_path)

                    if 'error' in config:
                        st.error(f"Error reading config: {config['error']}")
                    else:
                        # Find our server in the config
                        server_name, server_config = find_server_config(config)

                        if server_config:
                            st.success(f"✅ Found server configuration: **{server_name}**")

                            # Display server configuration
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown("**Command:**")
                                st.code(server_config.get('command', 'N/A'), language="text")

                                if 'args' in server_config:
                                    st.markdown("**Arguments:**")
                                    for arg in server_config['args']:
                                        st.code(arg, language="text")

                            with col2:
                                if 'env' in server_config:
                                    st.markdown("**Environment Variables:**")
                                    env_df = pd.DataFrame([
                                        {"Variable": k, "Value": v}
                                        for k, v in server_config['env'].items()
                                    ])
                                    st.dataframe(env_df, hide_index=True, use_container_width=True)

                            # Show full JSON
                            st.markdown("**Full Configuration:**")
                            st.json(server_config)
                        else:
                            st.warning("TDZ C64 Knowledge server not found in this configuration file")

                            # Show available servers
                            if 'mcpServers' in config:
                                st.markdown("**Available servers in this config:**")
                                for server_name in config['mcpServers'].keys():
                                    st.write(f"- {server_name}")
    with tab3:
        st.subheader("🌍 Runtime Environment Variables")

        # Key environment variables for this application
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
            'MAX_CHUNK_SIZE',
            'CHUNK_OVERLAP',
            'ANTHROPIC_API_KEY',
            'OPENAI_API_KEY'
        ]

        # Build dataframe of environment variables
        env_data = []
        for var in important_vars:
            value = os.environ.get(var, '')

            # Mask sensitive values
            if 'KEY' in var or 'SECRET' in var or 'PASSWORD' in var:
                if value:
                    value = '***' + value[-4:] if len(value) > 4 else '***'

            status = '✅ Set' if os.environ.get(var) else '❌ Not set'
            env_data.append({
                'Variable': var,
                'Value': value or '(not set)',
                'Status': status
            })

        df = pd.DataFrame(env_data)
        st.dataframe(df, hide_index=True, use_container_width=True)

        st.markdown("---")

        # Show all environment variables (optional expander)
        with st.expander("🔍 View All Environment Variables"):
            all_env = {k: v for k, v in os.environ.items()}
            # Mask sensitive values
            for key in all_env.keys():
                if 'KEY' in key or 'SECRET' in key or 'PASSWORD' in key or 'TOKEN' in key:
                    if all_env[key]:
                        all_env[key] = '***' + all_env[key][-4:] if len(all_env[key]) > 4 else '***'

            st.json(all_env, expanded=False)
    with tab4:
        st.subheader("⚡ Features & Capabilities")

        # Run health check to get feature status
        try:
            health = kb.health_check()

            # Display overall status
            status_color = {
                'healthy': '🟢',
                'degraded': '🟡',
                'unhealthy': '🔴'
            }
            st.markdown(f"### {status_color.get(health['status'], '⚪')} System Status: **{health['status'].upper()}**")

            if health.get('issues'):
                st.warning("**Issues:**")
                for issue in health['issues']:
                    st.write(f"- {issue}")

            st.markdown("---")

            # Feature flags in columns
            col1, col2 = st.columns(2)

            features = health.get('features', {})

            with col1:
                st.markdown("**Search Features:**")
                st.write(f"{'✅' if features.get('fts5_enabled') else '❌'} FTS5 Full-Text Search")
                st.write(f"{'✅' if features.get('semantic_search_enabled') else '❌'} Semantic Search (Enabled)")
                st.write(f"{'✅' if features.get('semantic_search_available') else '❌'} Semantic Search (Available)")
                st.write(f"{'✅' if features.get('bm25_available') else '❌'} BM25 Ranking")

                st.markdown("**Document Processing:**")
                st.write(f"{'✅' if features.get('pdf_support') else '❌'} PDF Support")
                st.write(f"{'✅' if features.get('ocr_enabled') else '❌'} OCR (Optical Character Recognition)")

            with col2:
                st.markdown("**AI Features:**")
                st.write(f"{'✅' if features.get('entity_extraction_available') else '❌'} Entity Extraction")
                st.write(f"{'✅' if features.get('relationship_extraction_available') else '❌'} Relationship Mapping")
                st.write(f"{'✅' if features.get('rag_available') else '❌'} RAG Question Answering")

                st.markdown("**Web Features:**")
                st.write(f"{'✅' if features.get('web_scraping_available') else '❌'} Web Scraping")
                st.write(f"{'✅' if features.get('url_monitoring_available') else '❌'} URL Monitoring")

            st.markdown("---")

            # Database statistics
            st.markdown("**Database Statistics:**")
            stats_col1, stats_col2, stats_col3 = st.columns(3)

            with stats_col1:
                st.metric("Total Documents", health.get('database', {}).get('total_documents', 0))
                st.metric("Total Chunks", health.get('database', {}).get('total_chunks', 0))

            with stats_col2:
                st.metric("Total Entities", health.get('database', {}).get('total_entities', 0))
                st.metric("Total Relationships", health.get('database', {}).get('total_relationships', 0))

            with stats_col3:
                embeddings_count = features.get('embeddings_count', 0)
                embeddings_size = features.get('embeddings_size_mb', 0)
                st.metric("Embeddings", f"{embeddings_count:,}")
                if embeddings_size > 0:
                    st.caption(f"({embeddings_size:.2f} MB)")

            st.markdown("---")

            # Full health check JSON
            with st.expander("🔍 View Full Health Check Results"):
                st.json(health, expanded=False)

        except Exception as e:
            st.error(f"Error retrieving features and capabilities: {str(e)}")
            st.exception(e)
