"""Admin GUI page: 🔍 Search.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
from datetime import datetime
import os
import streamlit as st


def render(kb):
    st.title("🔍 Search Knowledge Base")
    if st.session_state.get('index_status') == 'building':
        st.info("⏳ **Search index is building in the background...**\n\n"
                "You can search now, but the first search may take a moment while the index completes. "
                "The sidebar shows the current status.")
        # Auto-refresh every 2 seconds while building
        time.sleep(2)
        st.rerun()
    elif st.session_state.get('index_status') == 'ready':
        if 'index_build_time' in st.session_state:
            st.success(f"✅ Search index ready! (Built in {st.session_state.index_build_time:.1f}s)")
    search_mode = st.radio(
        "Search Mode",
        ["Keyword (FTS5)", "Semantic", "Hybrid"],
        horizontal=True
    )
    use_nl_translation = st.checkbox(
        "🤖 Use Natural Language Translation (AI-powered query parsing)",
        value=False,
        help="Enable AI to parse your natural language query and extract entities, keywords, and optimal search parameters"
    )
    with st.form(key="search_form", clear_on_submit=False):
        # Search input
        if use_nl_translation:
            query = st.text_area("Enter your natural language question:", "",
                               placeholder="e.g., 'find information about sprites on the VIC-II chip' or 'how does sound work on the C64?'",
                               height=80)
        else:
            query = st.text_input("Enter your search query:", "")

        col1, col2, col3 = st.columns(3)
        with col1:
            max_results = st.number_input("Max Results", min_value=1, max_value=50, value=10)
        with col2:
            if search_mode == "Hybrid":
                semantic_weight = st.slider("Semantic Weight", 0.0, 1.0, 0.7, 0.1)
            else:
                semantic_weight = 0.7  # Default value when not shown
        with col3:
            tag_filter = st.text_input("Filter by tags (comma-separated)", "")

        # Search button (form submit button)
        search_submitted = st.form_submit_button("🔍 Search")
    if search_submitted and query:
        tags = [t.strip() for t in tag_filter.split(',') if t.strip()] if tag_filter else None

        # Natural Language Translation
        nl_result = None
        if use_nl_translation:
            try:
                with st.spinner("🤖 Translating natural language query..."):
                    nl_result = kb.translate_nl_query(query, confidence_threshold=0.7)

                # Display translation results in an expander
                with st.expander("🔍 Query Translation Results", expanded=True):
                    st.markdown(f"**Original Query:** \"{nl_result['original_query']}\"")
                    st.markdown(f"**Search Mode Recommendation:** `{nl_result['search_mode'].upper()}`")
                    st.markdown(f"**Confidence:** {nl_result['confidence']:.0%}")
                    st.markdown(f"**Reasoning:** {nl_result['reasoning']}")

                    if nl_result.get('search_terms'):
                        st.markdown(f"**Search Terms:** {', '.join(nl_result['search_terms'])}")

                    if nl_result.get('entities_found'):
                        st.markdown(f"**Entities Detected:** {len(nl_result['entities_found'])} found")
                        entity_data = []
                        for e in nl_result['entities_found'][:10]:  # Show top 10
                            entity_data.append({
                                'Entity': e['text'],
                                'Type': e['type'],
                                'Confidence': f"{e['confidence']:.0%}"
                            })
                        st.dataframe(pd.DataFrame(entity_data), use_container_width=True)
                        if len(nl_result['entities_found']) > 10:
                            st.caption(f"... and {len(nl_result['entities_found']) - 10} more entities")

                    if nl_result.get('facet_filters'):
                        st.markdown("**Facet Filters:**")
                        for facet_type, values in nl_result['facet_filters'].items():
                            st.markdown(f"- **{facet_type}:** {', '.join(values)}")

                    # Show if LLM was unavailable
                    if nl_result.get('confidence', 1.0) < 0.6:
                        st.warning("⚠️ LLM may be unavailable - using basic keyword extraction")

                # Override search mode based on translation recommendation
                if nl_result['search_mode'] == 'semantic':
                    search_mode = "Semantic"
                elif nl_result['search_mode'] == 'hybrid':
                    search_mode = "Hybrid"
                else:
                    search_mode = "Keyword (FTS5)"

                # Use search terms from translation if available, otherwise use original query
                if nl_result.get('search_terms'):
                    query = ' '.join(nl_result['search_terms'])
                # else keep original query

            except ValueError as e:
                st.error(f"Translation error: {e}\n\nMake sure LLM_PROVIDER and API key are configured.")
                st.stop()
            except Exception as e:
                st.error(f"Error during query translation: {e}")
                st.stop()

        # Create centered containers for progress bar and status text
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            progress_bar = st.progress(0)
            status_text = st.empty()

        try:
            # Wait for index to be ready if it's still building
            if st.session_state.get('index_status') == 'building':
                status_text.text("⏳ Waiting for search index to finish building...")
                progress_bar.progress(0.1)

                # Wait for the index thread to complete (with timeout)
                if st.session_state.get('index_thread'):
                    start_wait = time.time()
                    while st.session_state.get('index_status') == 'building':
                        time.sleep(0.5)
                        # Auto-refresh the page to update status
                        if time.time() - start_wait > 0.5:  # Check every 0.5 seconds
                            st.rerun()
                        # Timeout after 60 seconds
                        if time.time() - start_wait > 60:
                            st.error("⚠️ Index building timeout. Please try again.")
                            break

            # Update status - preparing search
            status_text.text(f"🔍 Preparing {search_mode.lower()} search...")
            progress_bar.progress(0.2)

            # Perform search based on mode
            if search_mode == "Keyword (FTS5)":
                status_text.text(f"🔎 Searching for '{query}' using FTS5...")
                progress_bar.progress(0.5)
                results = kb.search(query, max_results, tags)
            elif search_mode == "Semantic":
                if not kb.use_semantic:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("Semantic search is not enabled. Set USE_SEMANTIC_SEARCH=1")
                    results = []
                else:
                    status_text.text(f"🧠 Computing semantic embeddings for '{query}'...")
                    progress_bar.progress(0.5)
                    results = kb.semantic_search(query, max_results, tags)
            else:  # Hybrid
                if not kb.use_semantic:
                    progress_bar.empty()
                    status_text.empty()
                    st.error("Hybrid search requires semantic search. Set USE_SEMANTIC_SEARCH=1")
                    results = []
                else:
                    status_text.text(f"🔬 Running hybrid search (keyword + semantic) for '{query}'...")
                    progress_bar.progress(0.5)
                    results = kb.hybrid_search(query, max_results, tags, semantic_weight)

            # Update status - processing results
            status_text.text("📊 Processing results...")
            progress_bar.progress(0.9)

            # Complete
            progress_bar.progress(1.0)
            status_text.text("✅ Search complete!")

            # Clear progress indicators after a brief moment
            import time
            time.sleep(0.3)
            progress_bar.empty()
            status_text.empty()

            # Display results
            if not results:
                st.warning("No results found.")
            else:
                st.success(f"Found {len(results)} results")

                # Export options
                col1, col2 = st.columns([3, 1])
                with col2:
                    export_format = st.selectbox("Export as", ["Markdown", "JSON", "HTML"])
                    if st.button("📤 Export Results"):
                        format_map = {"Markdown": "markdown", "JSON": "json", "HTML": "html"}
                        exported = kb.export_search_results(results, format_map[export_format], query)

                        st.download_button(
                            label=f"Download {export_format}",
                            data=exported,
                            file_name=f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format_map[export_format]}",
                            mime="text/plain"
                        )

                # Display each result
                for i, result in enumerate(results, 1):
                    # Create unique key for this result using index, doc_id, and chunk_id
                    doc_id = result.get('doc_id', 'unknown')
                    chunk_id = result.get('chunk_id', 'none')
                    unique_key = f"{i}_{doc_id}_{chunk_id}"

                    with st.expander(f"**{i}. {result.get('title', 'Untitled')}**", expanded=(i==1)):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.write(f"**File:** {result.get('filename', 'Unknown')}")
                            st.write(f"**Doc ID:** `{result.get('doc_id', 'Unknown')}`")
                            if result.get('chunk_id'):
                                st.write(f"**Chunk:** {result['chunk_id']}")
                            if result.get('page'):
                                st.write(f"**Page:** {result['page']}")

                        with col2:
                            score_key = 'score' if 'score' in result else 'similarity'
                            if score_key in result:
                                st.metric("Score", f"{result[score_key]:.4f}")

                            # File action buttons
                            st.markdown("**Actions:**")

                            # Add button to view file
                            if st.button("👁️ View File", key=f"view_file_{unique_key}", use_container_width=True):
                                if doc_id:
                                    st.session_state[f"show_viewer_{unique_key}"] = not st.session_state.get(f"show_viewer_{unique_key}", False)
                                    st.rerun()

                            # Add button to download file
                            if doc_id and doc_id in kb.documents:
                                doc = kb.documents[doc_id]
                                filepath = doc.filepath

                                # Check if file exists
                                if os.path.exists(filepath):
                                    try:
                                        with open(filepath, 'rb') as f:
                                            file_data = f.read()

                                        st.download_button(
                                            label="💾 Download",
                                            data=file_data,
                                            file_name=os.path.basename(filepath),
                                            mime="application/octet-stream",
                                            key=f"download_file_{unique_key}",
                                            use_container_width=True
                                        )
                                    except Exception as e:
                                        st.caption(f"❌ Download error: {str(e)}")
                                else:
                                    st.caption("❌ File not found")

                        # File viewer (if toggled)
                        if st.session_state.get(f"show_viewer_{unique_key}", False):
                            st.markdown("---")
                            st.subheader("📄 File Viewer")

                            if doc_id and doc_id in kb.documents:
                                doc = kb.documents[doc_id]
                                filepath = doc.filepath

                                if os.path.exists(filepath):
                                    file_ext = os.path.splitext(filepath)[1].lower()

                                    try:
                                        if file_ext == '.pdf':
                                            # Display PDF using iframe
                                            with open(filepath, 'rb') as f:
                                                pdf_data = f.read()

                                            import base64
                                            base64_pdf = base64.b64encode(pdf_data).decode('utf-8')
                                            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="800" type="application/pdf"></iframe>'
                                            st.markdown(pdf_display, unsafe_allow_html=True)

                                        elif file_ext == '.md':
                                            # Display markdown files with rendering
                                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                                content = f.read()

                                            # Toggle for raw vs rendered view
                                            view_mode = st.radio(
                                                "View Mode",
                                                ["Rendered", "Raw Markdown"],
                                                horizontal=True,
                                                key=f"md_view_mode_{unique_key}"
                                            )

                                            if view_mode == "Rendered":
                                                # Render markdown with proper formatting
                                                st.markdown("### 📄 Rendered Markdown")
                                                with st.container():
                                                    st.markdown(content, unsafe_allow_html=False)
                                            else:
                                                # Show raw markdown
                                                st.markdown("### 📝 Raw Markdown")
                                                st.code(content, language='markdown', line_numbers=True)

                                        elif file_ext == '.txt':
                                            # Display text files in a scrollable container
                                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                                content = f.read()

                                            st.markdown("### 📄 Text File Content")

                                            # Use expander for very long files
                                            line_count = content.count('\n') + 1
                                            if line_count > 50:
                                                st.info(f"📊 File contains {line_count} lines")

                                            # Display in a scrollable code block
                                            st.code(content, language='text', line_numbers=True)

                                        elif file_ext in ['.html', '.htm']:
                                            # Display HTML files
                                            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                                                html_content = f.read()

                                            st.code(html_content, language='html')

                                        elif file_ext in ['.xlsx', '.xls']:
                                            # Display Excel files
                                            try:
                                                import pandas as pd
                                                df = pd.read_excel(filepath)
                                                st.dataframe(df, use_container_width=True)
                                            except Exception as e:
                                                st.warning(f"Cannot preview Excel file: {str(e)}")

                                        else:
                                            st.info(f"Preview not available for {file_ext} files. Use the Download button to view externally.")

                                    except Exception as e:
                                        st.error(f"Error displaying file: {str(e)}")
                                else:
                                    st.error(f"❌ File not found: {filepath}")
                            else:
                                st.error("❌ Document not found in knowledge base")

                        # Snippet
                        if 'snippet' in result:
                            st.markdown("**Excerpt:**")
                            st.markdown(result['snippet'])

                        # Tags
                        if result.get('tags'):
                            st.write(f"🏷️ {', '.join(result['tags'])}")

        except Exception as e:
            st.error(f"Search error: {str(e)}")
