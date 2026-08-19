"""Admin GUI page: 🌐 Web Scraping.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import streamlit as st
from admin_common import format_timestamp, trigger_background_reindex


def render(kb):
    st.title("🌐 Web Scraping")
    st.write("Scrape documentation websites and convert them to searchable markdown documents.")
    with st.expander("➕ Add New URL to Scrape", expanded=True):
        st.subheader("Scrape Documentation Website")

        url_input = st.text_input(
            "📍 Website URL",
            placeholder="https://docs.example.com/api/",
            key="scrape_url_input",
            help="Enter the starting URL to scrape"
        )

        # Simple options (above the fold)
        col1, col2, col3 = st.columns(3)
        with col1:
            follow_links = st.checkbox(
                "Follow Links",
                value=True,
                help="Follow links to scrape sub-pages (uncheck to scrape only the single page)"
            )
        with col2:
            same_domain_only = st.checkbox(
                "Same Domain Only",
                value=True,
                help="Only follow links on the same domain (prevents scraping external sites)"
            )
        with col3:
            max_pages = st.number_input(
                "Max Pages",
                min_value=1,
                max_value=500,
                value=50,
                help="Maximum number of pages to scrape"
            )

        # Configuration options
        with st.expander("⚙️ Advanced Scraping Options", expanded=False):
            col1, col2 = st.columns(2)

            with col1:
                scrape_depth = st.number_input(
                    "Max Depth",
                    min_value=1,
                    max_value=10,
                    value=3,
                    help="Maximum link depth to follow (1=single page, 2=linked pages, 3=two levels deep)"
                )
                scrape_threads = st.number_input(
                    "Threads",
                    min_value=1,
                    max_value=20,
                    value=10,
                    help="Number of concurrent download threads for faster scraping"
                )

            with col2:
                scrape_delay = st.number_input(
                    "Delay (ms)",
                    min_value=0,
                    max_value=5000,
                    value=100,
                    help="Delay between requests to avoid overwhelming the server"
                )
                scrape_limit = st.text_input(
                    "Limit URLs (optional)",
                    placeholder="https://docs.example.com/api/",
                    help="Advanced: Only scrape URLs with this prefix (overrides Same Domain Only)"
                )

            scrape_selector = st.text_input(
                "CSS Selector (optional)",
                placeholder="article.main-content",
                help="CSS selector to extract specific content (e.g., main article body)"
            )

        col1, col2 = st.columns(2)
        with col1:
            scrape_title = st.text_input(
                "Base Title (optional)",
                "",
                key="scrape_title",
                help="Base title for scraped documents (will be combined with page titles)"
            )
        with col2:
            scrape_tags = st.text_input(
                "Tags (comma-separated)",
                "",
                key="scrape_tags",
                help="Additional tags (domain name will be auto-added)"
            )

        if st.button("🚀 Start Scraping", key="scrape_url_btn", type="primary"):
            if not url_input:
                st.error("❌ Please enter a URL")
            else:
                # Show centered loading indicator
                loading_container = st.empty()
                progress_container = st.empty()
                status_container = st.empty()
                url_container = st.empty()

                with loading_container.container():
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        st.markdown("""
                            <style>
                            .scraping-container {
                                padding: 2rem;
                                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                                border-radius: 15px;
                                box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                                text-align: center;
                                margin: 2rem 0;
                            }
                            .scraping-text {
                                color: white;
                                font-size: 1.5rem;
                                font-weight: bold;
                                margin-top: 1rem;
                            }
                            .scraping-subtext {
                                color: rgba(255,255,255,0.9);
                                font-size: 1rem;
                                margin-top: 0.5rem;
                            }
                            @keyframes pulse {
                                0%, 100% { transform: scale(1); }
                                50% { transform: scale(1.1); }
                            }
                            .scraping-icon {
                                font-size: 3rem;
                                animation: pulse 2s ease-in-out infinite;
                            }
                            .current-url {
                                font-size: 0.9rem;
                                color: rgba(255,255,255,0.8);
                                margin-top: 0.5rem;
                                font-family: monospace;
                                word-break: break-all;
                            }
                            </style>
                            <div class="scraping-container">
                                <div class="scraping-icon">🌐</div>
                                <div class="scraping-text">Scraping Website</div>
                                <div id="scraping-status" class="scraping-subtext">Initializing scraper...</div>
                                <div id="current-url" class="current-url"></div>
                            </div>
                        """, unsafe_allow_html=True)

                # Progress tracking
                progress_bar = progress_container.progress(0)

                # Progress callback for real-time updates
                def update_progress(progress_update):
                    """Update Streamlit UI with scraping progress."""
                    try:
                        # Calculate progress percentage
                        if progress_update.total > 0:
                            progress_pct = min(1.0, progress_update.current / progress_update.total)
                        else:
                            progress_pct = 0.0

                        # Update progress bar
                        progress_bar.progress(progress_pct)

                        # Update status message
                        status_msg = progress_update.message
                        if "⚠️" in status_msg:
                            status_container.warning(status_msg)
                        else:
                            status_container.info(status_msg)

                        # Update current URL
                        if progress_update.item and progress_update.item != url_input:
                            # Truncate long URLs for display
                            display_url = progress_update.item
                            if len(display_url) > 80:
                                display_url = display_url[:77] + "..."
                            url_container.markdown(f"**Current page:** `{display_url}`")

                    except Exception as e:
                        # Silently ignore UI update errors (race conditions in Streamlit)
                        pass

                try:
                    # Parse tags
                    tags = [t.strip() for t in scrape_tags.split(',') if t.strip()]

                    # Start scraping with progress callback
                    result = kb.scrape_url(
                        url=url_input,
                        title=scrape_title or None,
                        tags=tags,
                        follow_links=follow_links,
                        same_domain_only=same_domain_only,
                        max_pages=max_pages,
                        depth=scrape_depth,
                        limit=scrape_limit or None,
                        threads=scrape_threads,
                        delay=scrape_delay,
                        selector=scrape_selector or None,
                        progress_callback=update_progress
                    )

                    # Clear loading indicator and progress widgets
                    loading_container.empty()
                    progress_container.empty()
                    status_container.empty()
                    url_container.empty()

                    if result['status'] == 'success':
                        st.success(f"✅ **Scraping complete!**\n\n"
                                 f"**Files scraped:** {result['files_scraped']}\n\n"
                                 f"**Documents added:** {result['docs_added']}\n\n"
                                 f"**Output directory:** `{result['output_dir']}`")

                        # Trigger background reindexing
                        trigger_background_reindex()
                        st.rerun()

                    elif result['status'] == 'partial':
                        st.warning(f"⚠️ **Scraping partially complete**\n\n"
                                 f"**Files scraped:** {result['files_scraped']}\n\n"
                                 f"**Documents added:** {result['docs_added']}\n\n"
                                 f"**Failed:** {result['docs_failed']}\n\n"
                                 f"**Error:** {result.get('error', 'Unknown')}")

                        # Trigger background reindexing
                        if result['docs_added'] > 0:
                            trigger_background_reindex()
                            st.rerun()
                    else:
                        st.error(f"❌ **Scraping failed**\n\n{result.get('error', 'Unknown error')}")

                except Exception as e:
                    loading_container.empty()
                    progress_container.empty()
                    status_container.empty()
                    url_container.empty()
                    st.error(f"❌ **Error during scraping:**\n\n{str(e)}")
    st.markdown("---")
    st.subheader("📚 Scraped Websites")
    url_docs = [doc for doc in kb.documents.values() if doc.source_url]
    if not url_docs:
        st.info("No scraped websites yet. Add a URL using the section above!")
    else:
        # Group by domain
        from urllib.parse import urlparse
        websites = {}
        for doc in url_docs:
            domain = urlparse(doc.source_url).netloc
            if domain not in websites:
                websites[domain] = []
            websites[domain].append(doc)

        # Display statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Websites", len(websites))
        with col2:
            st.metric("Total Pages", len(url_docs))
        with col3:
            # Count successful scrapes
            successful = sum(1 for doc in url_docs if doc.scrape_status == 'success')
            st.metric("Successful", successful)

        st.markdown("---")

        # Search/filter
        search_query = st.text_input("🔎 Filter websites", "")

        # Filter websites
        filtered_websites = {}
        if search_query:
            for domain, docs in websites.items():
                matching_docs = [d for d in docs if search_query.lower() in d.title.lower() or
                               search_query.lower() in d.source_url.lower()]
                if matching_docs:
                    filtered_websites[domain] = matching_docs
        else:
            filtered_websites = websites

        st.write(f"Showing {len(filtered_websites)} of {len(websites)} websites")

        # Display each website
        for domain, docs in sorted(filtered_websites.items()):
            with st.expander(f"🌐 {domain} ({len(docs)} pages)", expanded=False):
                # Website summary
                st.write(f"**Domain:** {domain}")
                st.write(f"**Pages scraped:** {len(docs)}")

                # Latest scrape date
                latest_scrape = max([doc.scrape_date for doc in docs if doc.scrape_date], default=None)
                if latest_scrape:
                    st.write(f"**Last scraped:** {format_timestamp(latest_scrape)}")

                # Tags summary
                all_tags = set()
                for doc in docs:
                    all_tags.update(doc.tags)
                if all_tags:
                    st.write(f"**Tags:** {', '.join(sorted(all_tags))}")

                st.markdown("---")

                # Actions
                col1, col2, col3 = st.columns(3)

                with col1:
                    if st.button("🔄 Re-scrape All", key=f"rescrape_all_{domain}"):
                        with st.spinner(f"Re-scraping all pages from {domain}..."):
                            rescrape_count = 0
                            for doc in docs:
                                try:
                                    result = kb.rescrape_document(doc.doc_id)
                                    if result['status'] == 'success':
                                        rescrape_count += 1
                                except Exception as e:
                                    st.error(f"Error re-scraping {doc.title}: {str(e)}")

                            if rescrape_count > 0:
                                trigger_background_reindex()
                                st.success(f"✅ Re-scraped {rescrape_count} pages")
                                st.rerun()

                with col2:
                    if st.button("📋 View All Pages", key=f"view_pages_{domain}"):
                        st.session_state[f"show_pages_{domain}"] = not st.session_state.get(f"show_pages_{domain}", False)
                        st.rerun()

                with col3:
                    if st.button("🗑️ Delete All", key=f"delete_all_{domain}"):
                        st.session_state[f"confirm_delete_{domain}"] = True
                        st.rerun()

                # Confirm delete
                if st.session_state.get(f"confirm_delete_{domain}", False):
                    st.warning(f"⚠️ This will delete all {len(docs)} pages from {domain}")
                    confirm_col1, confirm_col2 = st.columns(2)
                    with confirm_col1:
                        if st.button("✅ Confirm Delete", key=f"confirm_del_{domain}"):
                            for doc in docs:
                                kb.remove_document(doc.doc_id)
                            trigger_background_reindex()
                            st.session_state[f"confirm_delete_{domain}"] = False
                            st.success(f"Deleted all pages from {domain}")
                            st.rerun()
                    with confirm_col2:
                        if st.button("❌ Cancel", key=f"cancel_del_{domain}"):
                            st.session_state[f"confirm_delete_{domain}"] = False
                            st.rerun()

                # Show individual pages
                if st.session_state.get(f"show_pages_{domain}", False):
                    st.markdown("---")
                    st.write("**Individual Pages:**")

                    for doc in sorted(docs, key=lambda d: d.title):
                        page_col1, page_col2, page_col3, page_col4, page_col5 = st.columns([3, 1, 1, 1, 1])

                        with page_col1:
                            status_emoji = "✅" if doc.scrape_status == "success" else "⚠️"
                            st.write(f"{status_emoji} {doc.title}")
                            st.caption(f"🔗 {doc.source_url}")

                        with page_col2:
                            st.caption(f"{doc.total_chunks} chunks")

                        with page_col3:
                            if st.button("📄", key=f"view_md_{doc.doc_id}", help="View scraped markdown"):
                                st.session_state[f"show_md_{doc.doc_id}"] = not st.session_state.get(f"show_md_{doc.doc_id}", False)
                                st.rerun()

                        with page_col4:
                            if st.button("🔄", key=f"rescrape_{doc.doc_id}", help="Re-scrape this page"):
                                with st.spinner("Re-scraping..."):
                                    try:
                                        result = kb.rescrape_document(doc.doc_id)
                                        if result['status'] == 'success':
                                            trigger_background_reindex()
                                            st.success("✅ Re-scraped successfully")
                                            st.rerun()
                                        else:
                                            st.error(f"Failed: {result.get('error')}")
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")

                        with page_col5:
                            if st.button("🗑️", key=f"delete_{doc.doc_id}", help="Delete this page"):
                                kb.remove_document(doc.doc_id)
                                trigger_background_reindex()
                                st.rerun()

                        # Show markdown content if toggled
                        if st.session_state.get(f"show_md_{doc.doc_id}", False):
                            st.markdown("---")
                            st.subheader("📄 Scraped Markdown Content")

                            try:
                                # Get the full document content
                                full_doc = kb.get_document(doc.doc_id)

                                if full_doc and 'chunks' in full_doc:
                                    # Combine all chunks
                                    markdown_content = "\n\n".join([chunk['content'] for chunk in full_doc['chunks']])

                                    # Show metadata
                                    st.info(f"**Source:** {doc.source_url}\n\n"
                                           f"**Scraped:** {format_timestamp(doc.scrape_date) if doc.scrape_date else 'N/A'}\n\n"
                                           f"**Total chunks:** {len(full_doc['chunks'])}")

                                    # Display markdown in code block for easier viewing
                                    st.code(markdown_content, language="markdown")

                                    # Download button
                                    st.download_button(
                                        label="💾 Download Markdown",
                                        data=markdown_content,
                                        file_name=f"{doc.filename}.md",
                                        mime="text/markdown",
                                        key=f"download_md_{doc.doc_id}"
                                    )
                                else:
                                    st.warning("No markdown content available")
                            except Exception as e:
                                st.error(f"Error loading markdown: {str(e)}")
