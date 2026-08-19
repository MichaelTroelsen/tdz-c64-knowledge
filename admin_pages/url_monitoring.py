"""Admin GUI page: 🌐 URL Monitoring.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import pandas as pd
import streamlit as st


def render(kb):
    st.title("🌐 URL Monitoring Dashboard")
    st.write("Monitor scraped websites for updates, new pages, and missing content.")
    url_docs = [doc for doc in kb.documents.values() if doc.source_url]
    if not url_docs:
        st.info("📋 No URL-sourced documents found. Use the **🌐 Web Scraping** page to add websites first.")
    else:
        # Top metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🌐 Monitored Sites", len(url_docs))
        with col2:
            # Count unique base URLs
            base_urls = set()
            for doc in url_docs:
                if doc.scrape_config:
                    try:
                        config = json.loads(doc.scrape_config) if isinstance(doc.scrape_config, str) else doc.scrape_config
                        if 'base_url' in config:
                            base_urls.add(config['base_url'])
                    except (json.JSONDecodeError, AttributeError, TypeError):
                        pass
            st.metric("🔗 Unique Sources", len(base_urls))
        with col3:
            # Show last check time if available
            if 'last_url_check' in st.session_state:
                last_check = st.session_state['last_url_check']
                st.metric("🕒 Last Check", last_check.strftime("%H:%M:%S"))
            else:
                st.metric("🕒 Last Check", "Never")

        st.markdown("---")

        # Check controls
        st.subheader("🔍 Run Update Check")

        col1, col2, col3 = st.columns([2, 2, 3])

        with col1:
            check_mode = st.radio(
                "Check Mode:",
                ["Quick (Fast)", "Full (Comprehensive)"],
                help="Quick: Check Last-Modified headers only (~1s/site)\nFull: Discover new/missing pages (~10-60s/site)"
            )

        with col2:
            auto_rescrape = st.checkbox(
                "Auto Re-scrape",
                value=False,
                help="Automatically re-scrape changed documents"
            )

        with col3:
            st.write("")  # Spacing
            st.write("")  # Spacing
            if st.button("▶️ Run Check", type="primary", use_container_width=True):
                check_structure = (check_mode == "Full (Comprehensive)")

                with st.spinner(f"Running {check_mode.split()[0].lower()} check on {len(url_docs)} documents..."):
                    try:
                        results = kb.check_url_updates(
                            auto_rescrape=auto_rescrape,
                            check_structure=check_structure
                        )

                        st.session_state['last_url_check'] = datetime.now()
                        st.session_state['last_check_results'] = results
                        st.success("✅ Check complete!")
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Error during check: {str(e)}")

        st.markdown("---")

        # Display results if available
        if 'last_check_results' in st.session_state:
            results = st.session_state['last_check_results']

            # Results summary
            st.subheader("📊 Check Results")

            col1, col2, col3, col4, col5 = st.columns(5)
            with col1:
                st.metric("✅ Unchanged", len(results.get('unchanged', [])))
            with col2:
                changed_count = len(results.get('changed', []))
                st.metric("🔄 Changed", changed_count, delta=changed_count if changed_count > 0 else None)
            with col3:
                new_count = len(results.get('new_pages', []))
                st.metric("🆕 New Pages", new_count, delta=new_count if new_count > 0 else None)
            with col4:
                missing_count = len(results.get('missing_pages', []))
                st.metric("❌ Missing", missing_count, delta=-missing_count if missing_count > 0 else None)
            with col5:
                failed_count = len(results.get('failed', []))
                st.metric("⚠️ Failed", failed_count)

            # Tabbed results display
            tab1, tab2, tab3, tab4, tab5 = st.tabs([
                f"🔄 Changed ({len(results.get('changed', []))})",
                f"🆕 New Pages ({len(results.get('new_pages', []))})",
                f"❌ Missing ({len(results.get('missing_pages', []))})",
                f"✅ Unchanged ({len(results.get('unchanged', []))})",
                f"📈 Sessions ({len(results.get('scrape_sessions', []))})"
            ])

            # Tab 1: Changed documents
            with tab1:
                changed = results.get('changed', [])
                if changed:
                    st.write(f"**{len(changed)} documents have updates available:**")

                    for doc_info in changed:
                        with st.expander(f"🔄 {doc_info['title']}", expanded=False):
                            st.write(f"**URL:** {doc_info['url']}")
                            st.write(f"**Document ID:** `{doc_info['doc_id'][:12]}...`")
                            if 'last_modified' in doc_info:
                                st.write(f"**Last Modified:** {doc_info['last_modified']}")
                            if 'scraped_date' in doc_info:
                                st.write(f"**Scraped Date:** {doc_info['scraped_date']}")
                            if 'reason' in doc_info:
                                st.write(f"**Reason:** {doc_info['reason']}")

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🔄 Re-scrape", key=f"rescrape_{doc_info['doc_id']}"):
                                    with st.spinner(f"Re-scraping {doc_info['title']}..."):
                                        try:
                                            new_doc_id = kb.rescrape_document(doc_info['doc_id'])
                                            st.success(f"✅ Re-scraped successfully! New ID: {new_doc_id[:12]}...")
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")
                            with col2:
                                if st.button("🌐 Open URL", key=f"open_{doc_info['doc_id']}"):
                                    st.markdown(f"[Open in browser]({doc_info['url']})")
                else:
                    st.info("✅ No changed documents")

            # Tab 2: New pages
            with tab2:
                new_pages = results.get('new_pages', [])
                if new_pages:
                    st.write(f"**{len(new_pages)} new pages discovered:**")

                    # Group by base_url
                    by_site = {}
                    for page in new_pages:
                        site = page.get('base_url', 'Unknown')
                        if site not in by_site:
                            by_site[site] = []
                        by_site[site].append(page)

                    for site, pages in by_site.items():
                        with st.expander(f"🌐 {site} ({len(pages)} new pages)", expanded=False):
                            for page in pages[:10]:  # Show first 10
                                st.write(f"- {page['url']}")
                            if len(pages) > 10:
                                st.write(f"... and {len(pages) - 10} more")

                            if st.button(f"📥 Scrape All {len(pages)} Pages", key=f"scrape_new_{hash(site)}"):
                                with st.spinner(f"Scraping {len(pages)} new pages from {site}..."):
                                    try:
                                        scraped = 0
                                        progress_bar = st.progress(0)
                                        for i, page in enumerate(pages):
                                            try:
                                                kb.scrape_url(
                                                    page['url'],
                                                    depth=1,  # Single page
                                                    follow_links=False
                                                )
                                                scraped += 1
                                            except Exception as e:
                                                st.warning(f"Failed to scrape {page['url']}: {str(e)}")
                                            progress_bar.progress((i + 1) / len(pages))
                                        st.success(f"✅ Scraped {scraped}/{len(pages)} new pages")
                                    except Exception as e:
                                        st.error(f"❌ Error: {str(e)}")
                else:
                    st.info("✅ No new pages discovered")

            # Tab 3: Missing pages
            with tab3:
                missing = results.get('missing_pages', [])
                if missing:
                    st.write(f"**{len(missing)} pages missing or inaccessible:**")

                    for doc_info in missing:
                        with st.expander(f"❌ {doc_info['title']}", expanded=False):
                            st.write(f"**URL:** {doc_info['url']}")
                            st.write(f"**Document ID:** `{doc_info['doc_id'][:12]}...`")
                            if 'reason' in doc_info:
                                st.write(f"**Reason:** {doc_info['reason']}")

                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("🗑️ Remove from DB", key=f"remove_{doc_info['doc_id']}"):
                                    if st.button("⚠️ Confirm Delete", key=f"confirm_remove_{doc_info['doc_id']}"):
                                        try:
                                            kb.remove_document(doc_info['doc_id'])
                                            st.success(f"✅ Removed {doc_info['title']}")
                                            # Clear results to force re-check
                                            if 'last_check_results' in st.session_state:
                                                del st.session_state['last_check_results']
                                            st.rerun()
                                        except Exception as e:
                                            st.error(f"❌ Error: {str(e)}")
                            with col2:
                                if st.button("🌐 Check URL", key=f"check_{doc_info['doc_id']}"):
                                    st.markdown(f"[Open in browser]({doc_info['url']})")
                else:
                    st.info("✅ No missing pages")

            # Tab 4: Unchanged documents
            with tab4:
                unchanged = results.get('unchanged', [])
                if unchanged:
                    st.write(f"**{len(unchanged)} documents are up to date:**")

                    # Display as table
                    df_data = []
                    for doc_info in unchanged:
                        df_data.append({
                            'Title': doc_info['title'][:50] + '...' if len(doc_info['title']) > 50 else doc_info['title'],
                            'Doc ID': doc_info['doc_id'][:12] + '...',
                            'URL': doc_info['url']
                        })

                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True, height=400)
                else:
                    st.info("No unchanged documents")

            # Tab 5: Scrape sessions
            with tab5:
                sessions = results.get('scrape_sessions', [])
                if sessions:
                    st.write(f"**{len(sessions)} scrape sessions monitored:**")

                    # Display as table
                    df_data = []
                    for session in sessions:
                        df_data.append({
                            'Site': session['base_url'],
                            'Total Docs': session['docs_count'],
                            'Unchanged': session['unchanged'],
                            'Changed': session['changed'],
                            'New': session.get('new', 0),
                            'Missing': session.get('missing', 0)
                        })

                    if df_data:
                        df = pd.DataFrame(df_data)
                        st.dataframe(df, use_container_width=True)

                        # Export option
                        st.markdown("---")
                        if st.button("📥 Export Results as JSON"):
                            import json
                            from datetime import datetime

                            filename = f"url_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                            json_str = json.dumps(results, indent=2, default=str)

                            st.download_button(
                                label="💾 Download JSON",
                                data=json_str,
                                file_name=filename,
                                mime="application/json"
                            )
                else:
                    st.info("No scrape sessions found")

        else:
            # No results yet
            st.info("👆 Click 'Run Check' above to check for URL updates")

        # Sites overview table
        st.markdown("---")
        st.subheader("🌐 All Monitored Sites")

        # Group documents by base URL
        by_base_url = {}
        for doc in url_docs:
            # Parse scrape_config from JSON string if it exists
            if doc.scrape_config:
                try:
                    config = json.loads(doc.scrape_config) if isinstance(doc.scrape_config, str) else doc.scrape_config
                    base_url = config.get('base_url', doc.source_url)
                except (json.JSONDecodeError, AttributeError):
                    base_url = doc.source_url
            else:
                base_url = doc.source_url

            if base_url not in by_base_url:
                by_base_url[base_url] = []
            by_base_url[base_url].append(doc)

        # Display grouped sites
        for base_url, docs in sorted(by_base_url.items()):
            with st.expander(f"🌐 {base_url} ({len(docs)} documents)", expanded=False):
                st.write(f"**Base URL:** {base_url}")
                st.write(f"**Documents:** {len(docs)}")

                # Show document list
                for doc in docs[:5]:  # Show first 5
                    st.write(f"- {doc.title}")
                if len(docs) > 5:
                    st.write(f"... and {len(docs) - 5} more")

                # Quick actions
                col1, col2, col3 = st.columns(3)
                with col1:
                    if st.button("🔄 Re-scrape All", key=f"rescrape_all_{hash(base_url)}"):
                        with st.spinner(f"Re-scraping {len(docs)} documents..."):
                            try:
                                rescraped = 0
                                progress_bar = st.progress(0)
                                for i, doc in enumerate(docs):
                                    try:
                                        kb.rescrape_document(doc.doc_id)
                                        rescraped += 1
                                    except Exception as e:
                                        st.warning(f"Failed to re-scrape {doc.title}: {str(e)}")
                                    progress_bar.progress((i + 1) / len(docs))
                                st.success(f"✅ Re-scraped {rescraped}/{len(docs)} documents")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                with col2:
                    if st.button("🗑️ Remove All", key=f"remove_all_{hash(base_url)}"):
                        if st.button("⚠️ Confirm Delete All", key=f"confirm_remove_all_{hash(base_url)}"):
                            try:
                                for doc in docs:
                                    kb.remove_document(doc.doc_id)
                                st.success(f"✅ Removed {len(docs)} documents from {base_url}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")
                with col3:
                    if st.button("📊 View Stats", key=f"stats_{hash(base_url)}"):
                        # Show detailed stats
                        total_chunks = sum(len(kb._get_chunks_db(doc.doc_id)) for doc in docs)
                        total_size = sum(len(doc.content or '') for doc in docs)
                        st.write(f"**Total Chunks:** {total_chunks}")
                        st.write(f"**Total Size:** {total_size:,} chars")
