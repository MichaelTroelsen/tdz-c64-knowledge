"""Admin GUI page: 🔍 Archive Search.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
from datetime import datetime
from datetime import timezone
from pathlib import Path
from urllib.parse import quote
import json
import os
import streamlit as st


def render(kb):
    st.title("🔍 Archive.org Search")
    st.markdown("Search and download C64-related documents from the Internet Archive.")
    try:
        import internetarchive as ia
        ia_available = True
    except ImportError:
        ia_available = False
    if not ia_available:
        st.error("📦 The `internetarchive` library is not installed.")
        st.markdown("**To install:**")
        st.code(".venv\\Scripts\\pip install internetarchive", language="bash")
        st.info("After installing, restart the Streamlit server to use this feature.")
    else:
        # Create tabs for search, suggestions, quick-added, and downloads
        tab1, tab2, tab3, tab4 = st.tabs(["🔍 Search Archive", "🤖 AI Suggestions", "⚡ Quick Added", "📥 Downloaded Files"])

        # ========== SEARCH TAB ==========
        with tab1:
            st.subheader("Search Internet Archive")

            # Search form
            col1, col2 = st.columns([2, 1])

            with col1:
                search_query = st.text_input(
                    "Search Query",
                    placeholder="e.g., commodore 64 programming, VIC-II, SID chip",
                    help="Enter keywords to search for C64-related documents"
                )

            with col2:
                max_results = st.number_input("Max Results", min_value=5, max_value=100, value=20, step=5)

            # Advanced filters
            with st.expander("🔧 Advanced Filters", expanded=False):
                filter_col1, filter_col2 = st.columns(2)

                with filter_col1:
                    # File type filter
                    file_types = st.multiselect(
                        "File Types",
                        options=["PDF", "TXT", "HTML", "DJVU", "EPUB", "MOBI"],
                        default=["PDF", "TXT"],
                        help="Select document formats to search for"
                    )

                    # Collection filter
                    collection = st.selectbox(
                        "Collection",
                        options=[
                            "All Collections",
                            "texts",
                            "software",
                            "data",
                            "movies",
                            "audio",
                            "image"
                        ],
                        help="Filter by Internet Archive collection"
                    )

                with filter_col2:
                    # Subject/tag filter
                    subject_tags = st.text_input(
                        "Subject Tags (comma-separated)",
                        placeholder="e.g., commodore-64, programming, hardware",
                        help="Filter by subject tags"
                    )

                    # Date range
                    date_filter = st.checkbox("Filter by Date Range")
                    if date_filter:
                        date_col1, date_col2 = st.columns(2)
                        with date_col1:
                            start_year = st.number_input("From Year", min_value=1900, max_value=2026, value=1980)
                        with date_col2:
                            end_year = st.number_input("To Year", min_value=1900, max_value=2026, value=2026)

            # Search button
            if st.button("🔍 Search Archive.org", type="primary"):
                if not search_query:
                    st.warning("Please enter a search query")
                else:
                    with st.spinner("Searching Internet Archive..."):
                        try:
                            # Build search query
                            query_parts = [search_query]

                            # Add collection filter
                            if collection != "All Collections":
                                query_parts.append(f"collection:{collection}")

                            # Add file type filter
                            if file_types:
                                format_query = " OR ".join([f"format:{ft}" for ft in file_types])
                                query_parts.append(f"({format_query})")

                            # Add subject filter
                            if subject_tags:
                                tags = [tag.strip() for tag in subject_tags.split(",")]
                                for tag in tags:
                                    query_parts.append(f"subject:{tag}")

                            # Add date filter
                            if date_filter:
                                query_parts.append(f"year:[{start_year} TO {end_year}]")

                            # Combine query
                            full_query = " AND ".join(query_parts)

                            # Perform search (search_items returns an iterator)
                            search = ia.search_items(full_query)

                            # Store results in session state
                            results = []
                            items_processed = 0

                            for item in search:
                                # Stop if we've reached max_results
                                if items_processed >= max_results:
                                    break

                                items_processed += 1

                                try:
                                    # Get item metadata
                                    item_obj = ia.get_item(item['identifier'])
                                    metadata = item_obj.metadata

                                    # Get file information
                                    files = []
                                    for file in item_obj.files:
                                        # Filter by selected file types
                                        file_format = file.get('format', '').upper()
                                        if not file_types or any(ft.upper() in file_format for ft in file_types):
                                            # URL-encode the filename to handle spaces and special characters
                                            filename = file.get('name', '')
                                            encoded_filename = quote(filename, safe='')
                                            files.append({
                                                'name': filename,
                                                'size': file.get('size', 0),
                                                'format': file_format,
                                                'url': f"https://archive.org/download/{item['identifier']}/{encoded_filename}"
                                            })

                                    if files:  # Only include items with matching files
                                        results.append({
                                            'identifier': item['identifier'],
                                            'title': metadata.get('title', 'Untitled'),
                                            'description': metadata.get('description', 'No description'),
                                            'creator': metadata.get('creator', 'Unknown'),
                                            'date': metadata.get('date', 'Unknown'),
                                            'subject': metadata.get('subject', []),
                                            'downloads': metadata.get('downloads', 0),
                                            'url': f"https://archive.org/details/{item['identifier']}",
                                            'files': files
                                        })

                                except Exception as e:
                                    st.warning(f"Error processing item {item.get('identifier', 'unknown')}: {str(e)}")
                                    continue

                            st.session_state.archive_results = results
                            st.session_state.archive_query = full_query

                            if results:
                                st.success(f"✅ Found {len(results)} items with matching files")
                            else:
                                st.warning("No results found. Try adjusting your search query or filters.")

                        except Exception as e:
                            st.error(f"Search error: {str(e)}")
                            st.exception(e)

            # Display results
            if 'archive_results' in st.session_state and st.session_state.archive_results:
                st.markdown("---")
                st.subheader(f"📚 Search Results ({len(st.session_state.archive_results)} items)")

                # Display query
                with st.expander("🔍 Query Details"):
                    st.code(st.session_state.archive_query, language="text")

                # Results display
                for idx, result in enumerate(st.session_state.archive_results):
                    with st.expander(f"📄 {result['title']}", expanded=False):
                        col1, col2 = st.columns([3, 1])

                        with col1:
                            st.markdown(f"**Identifier:** `{result['identifier']}`")
                            st.markdown(f"**Creator:** {result['creator']}")
                            st.markdown(f"**Date:** {result['date']}")

                            if result['description']:
                                desc = result['description']
                                if isinstance(desc, list):
                                    desc = desc[0] if desc else "No description"
                                # Truncate long descriptions
                                if len(desc) > 300:
                                    desc = desc[:300] + "..."
                                st.markdown(f"**Description:** {desc}")

                            if result['subject']:
                                subjects = result['subject'] if isinstance(result['subject'], list) else [result['subject']]
                                st.markdown(f"**Tags:** {', '.join(subjects[:5])}")

                            st.markdown(f"**Downloads:** {result['downloads']:,}")
                            st.markdown(f"**Archive URL:** [{result['url']}]({result['url']})")

                        with col2:
                            st.metric("Files", len(result['files']))

                        # Files section
                        st.markdown("**Available Files:**")
                        for file_idx, file in enumerate(result['files'][:10]):  # Limit to 10 files
                            file_col1, file_col2, file_col3 = st.columns([3, 1, 2])

                            with file_col1:
                                st.text(f"📄 {file['name']}")

                            with file_col2:
                                size_mb = int(file['size']) / (1024 * 1024) if file['size'] else 0
                                st.caption(f"{size_mb:.2f} MB")

                            with file_col3:
                                # Check if file already exists in KB by source URL
                                # Support both formats:
                                # - New format (file URL): https://archive.org/download/item-id/filename.pdf
                                # - Old format (item URL): https://archive.org/details/item-id
                                existing_doc = None
                                item_id = result['identifier']

                                for doc in kb.documents.values():
                                    if not hasattr(doc, 'source_url') or not doc.source_url:
                                        continue

                                    # Try exact file URL match (new format)
                                    if doc.source_url == file['url']:
                                        existing_doc = doc
                                        break

                                    # Try item identifier match (old format)
                                    # Check if source_url contains the item identifier
                                    if item_id in doc.source_url:
                                        # Also check if filename matches (ignore extension for flexibility)
                                        safe_filename = Path(file['name']).stem  # Get filename without extension
                                        doc_stem = Path(doc.filename).stem  # Get doc filename without extension
                                        if safe_filename.lower() in doc_stem.lower() or doc_stem.lower() in safe_filename.lower():
                                            existing_doc = doc
                                            break

                                if existing_doc:
                                    # File already in KB - show status
                                    st.success(f"✅ In KB")
                                    st.caption(f"Doc: {existing_doc.doc_id[:12]}...")
                                else:
                                    # File not in KB - show download and quick add buttons
                                    download_col1, download_col2 = st.columns(2)

                                    with download_col1:
                                        if st.button("💾 Download", key=f"download_{idx}_{file_idx}"):
                                            # Download to downloads folder
                                            downloads_dir = Path(st.session_state.data_dir) / "downloads"
                                            downloads_dir.mkdir(exist_ok=True)

                                            try:
                                                with st.spinner(f"Downloading {file['name']}..."):
                                                    import urllib.request
                                                    # Extract just the filename (no directory path) to avoid path issues
                                                    safe_filename = Path(file['name']).name
                                                    filepath = downloads_dir / safe_filename
                                                    urllib.request.urlretrieve(file['url'], filepath)
                                                    st.success(f"✅ Downloaded to {filepath}")

                                            except Exception as e:
                                                st.error(f"Download failed: {str(e)}")

                                    with download_col2:
                                        if st.button("⚡ Quick Add", key=f"quickadd_{idx}_{file_idx}"):
                                            # Download and add to knowledge base
                                            try:
                                                with st.spinner(f"Downloading and adding {file['name']}..."):
                                                    import urllib.request
                                                    import tempfile

                                                    # Create temp directory in data_dir (within allowed paths)
                                                    temp_dir = Path(st.session_state.data_dir) / "temp"
                                                    temp_dir.mkdir(exist_ok=True)

                                                    # Download to temp file in allowed directory
                                                    # Extract just the filename (no directory path) to avoid path issues
                                                    safe_filename = Path(file['name']).name
                                                    temp_filename = f"quick_add_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename}"
                                                    tmp_path = temp_dir / temp_filename
                                                    urllib.request.urlretrieve(file['url'], str(tmp_path))

                                                    # Add to knowledge base
                                                    title = f"{result['title']} - {file['name']}"
                                                    tags = result['subject'] if isinstance(result['subject'], list) else [result['subject']]
                                                    if isinstance(tags, str):
                                                        tags = [tags]

                                                    # Add document first
                                                    doc = kb.add_document(
                                                        str(tmp_path),
                                                        title=title,
                                                        tags=tags
                                                    )

                                                    # Update with source URL metadata (use file URL for matching)
                                                    with kb.db_conn:
                                                        cursor = kb.db_conn.cursor()
                                                        cursor.execute("""
                                                            UPDATE documents
                                                            SET source_url = ?,
                                                                scrape_date = ?,
                                                                scrape_status = 'success'
                                                            WHERE doc_id = ?
                                                        """, (file['url'], datetime.now(timezone.utc).isoformat(), doc.doc_id))

                                                    # Update in-memory object
                                                    doc.source_url = file['url']
                                                    kb.documents[doc.doc_id] = doc

                                                    # Record in quick-added history
                                                    st.session_state.quick_added_files.append({
                                                        'title': title,
                                                        'file_name': file['name'],
                                                        'source_url': result['url'],
                                                        'doc_id': doc.doc_id,
                                                        'status': 'success',
                                                        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                                                    })

                                                    # Clean up temp file
                                                    tmp_path.unlink()

                                                    st.success(f"✅ Added to knowledge base!\nDoc ID: {doc.doc_id[:12]}...")
                                                    st.rerun()

                                            except Exception as e:
                                                # Clean up temp file if it exists
                                                if 'tmp_path' in locals() and tmp_path.exists():
                                                    tmp_path.unlink()

                                                # Record failed attempt
                                                st.session_state.quick_added_files.append({
                                                    'title': f"{result['title']} - {file['name']}",
                                                    'file_name': file['name'],
                                                    'source_url': result['url'],
                                                    'doc_id': None,
                                                    'status': 'failed',
                                                    'error': str(e),
                                                    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                                                })
                                                st.error(f"Quick add failed: {str(e)}")
                                            st.exception(e)

                        if len(result['files']) > 10:
                            st.caption(f"... and {len(result['files']) - 10} more files")

        # ========== AI SUGGESTIONS TAB ==========
        with tab2:
            st.subheader("🤖 AI-Powered Suggestions")
            st.markdown("Get AI recommendations for the most relevant files to download based on your search.")

            # Check if we have search results
            if 'archive_results' not in st.session_state or not st.session_state.archive_results:
                st.info("👆 Perform a search first to get AI-powered suggestions on which files to download.")
            else:
                # Display search context
                st.markdown(f"**Search Query:** `{st.session_state.get('archive_query', 'N/A')}`")
                st.markdown(f"**Results:** {len(st.session_state.archive_results)} items found")

                # Show exclusion info if present
                if 'ai_exclusions' in st.session_state and st.session_state.ai_exclusions:
                    st.info(f"ℹ️ Regenerating with {len(st.session_state.ai_exclusions)} file(s) excluded (already in KB)")

                # Button to generate AI suggestions
                if st.button("🤖 Generate AI Suggestions", type="primary"):
                    # Check if Anthropic API key is available
                    api_key = os.environ.get('ANTHROPIC_API_KEY')
                    if not api_key:
                        st.error("❌ ANTHROPIC_API_KEY not found in environment variables.")
                        st.info("Set the ANTHROPIC_API_KEY environment variable to use AI suggestions.")
                    else:
                        spinner_text = "🤖 AI is analyzing search results..."
                        if 'ai_exclusions' in st.session_state and st.session_state.ai_exclusions:
                            spinner_text = f"🤖 AI is analyzing search results (excluding {len(st.session_state.ai_exclusions)} files already in KB)..."
                        with st.spinner(spinner_text):
                            try:
                                import anthropic

                                # Prepare data for AI analysis
                                results_summary = []
                                for idx, result in enumerate(st.session_state.archive_results[:10]):  # Limit to first 10
                                    files_info = []
                                    for file in result['files'][:5]:  # Limit files per item
                                        files_info.append({
                                            'name': file['name'],
                                            'format': file['format'],
                                            'size_mb': round(int(file['size']) / (1024 * 1024), 2) if file['size'] else 0
                                        })

                                    results_summary.append({
                                        'index': idx,
                                        'title': result['title'],
                                        'creator': result['creator'],
                                        'date': result['date'],
                                        'description': result['description'][:300] if result['description'] else "No description",
                                        'subject': result['subject'][:5] if isinstance(result['subject'], list) else result['subject'],
                                        'downloads': result['downloads'],
                                        'files': files_info
                                    })

                                # Check if there are exclusions from previous recommendations
                                exclusion_text = ""
                                if 'ai_exclusions' in st.session_state and st.session_state.ai_exclusions:
                                    exclusion_text = f"""
IMPORTANT: DO NOT recommend the following files (they are already in the knowledge base):
{json.dumps(st.session_state.ai_exclusions, indent=2)}

You must exclude these files and recommend different files instead.
"""
                                    # Clear exclusions after using them
                                    del st.session_state.ai_exclusions

                                # Create AI prompt
                                prompt = f"""You are an expert in Commodore 64 documentation and retro computing. Analyze these search results from archive.org and recommend the TOP 5 most valuable files to download for building a C64 knowledge base.

Search Query: {st.session_state.get('archive_query', 'N/A')}

Search Results:
{json.dumps(results_summary, indent=2)}
{exclusion_text}
For each recommendation, provide:
1. Item title and specific file name
2. Why it's valuable (relevance, completeness, historical significance)
3. What knowledge it adds to a C64 documentation collection
4. Priority level (High/Medium/Low)

Focus on:
- Technical accuracy and depth
- Historical documentation (manuals, specifications)
- Unique or rare content
- Completeness and quality
- File format suitability (prefer PDF, TXT for text documents)

IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanations, just the JSON object.

Required JSON structure:
{{
  "recommendations": [
    {{
      "item_index": 0,
      "item_title": "...",
      "file_name": "...",
      "priority": "High|Medium|Low",
      "rationale": "Why this file is valuable...",
      "knowledge_value": "What specific knowledge it provides...",
      "score": 95
    }}
  ],
  "summary": "Overall analysis of the search results..."
}}

Rules:
- Provide exactly 10 recommendations
- Order by score (highest first)
- Use valid JSON syntax (proper quotes, commas, no trailing commas)
- Escape special characters in strings
- Return ONLY the JSON object, nothing else"""

                                # Call Claude API
                                # Use configurable model (fallback to most widely available model)
                                model = os.environ.get("AI_SUGGESTIONS_MODEL", "claude-haiku-4-5-20251001")
                                client = anthropic.Anthropic(api_key=api_key)
                                message = client.messages.create(
                                    model=model,
                                    max_tokens=4096,
                                    messages=[
                                        {"role": "user", "content": prompt}
                                    ]
                                )

                                # Parse AI response
                                response_text = message.content[0].text

                                # Extract JSON from response (handle markdown code blocks)
                                if "```json" in response_text:
                                    json_start = response_text.find("```json") + 7
                                    json_end = response_text.find("```", json_start)
                                    response_text = response_text[json_start:json_end].strip()
                                elif "```" in response_text:
                                    json_start = response_text.find("```") + 3
                                    json_end = response_text.find("```", json_start)
                                    response_text = response_text[json_start:json_end].strip()

                                # Try to parse JSON with better error handling
                                try:
                                    suggestions = json.loads(response_text)
                                except json.JSONDecodeError as je:
                                    # Show the problematic JSON for debugging
                                    st.error(f"❌ Failed to parse AI response as JSON")
                                    st.error(f"Error at line {je.lineno}, column {je.colno}: {je.msg}")
                                    with st.expander("Show AI Response (for debugging)"):
                                        st.code(response_text, language="json")
                                    raise

                                # Store in session state
                                st.session_state.ai_suggestions = suggestions

                                st.success(f"✅ AI generated {len(suggestions['recommendations'])} recommendations!")

                            except ImportError:
                                st.error("❌ The `anthropic` library is not installed.")
                                st.code("pip install anthropic", language="bash")
                            except Exception as e:
                                st.error(f"❌ Error generating AI suggestions: {str(e)}")
                                st.exception(e)

                # Display AI suggestions if available
                if 'ai_suggestions' in st.session_state and st.session_state.ai_suggestions:
                    suggestions = st.session_state.ai_suggestions

                    # Display summary
                    if 'summary' in suggestions:
                        st.markdown("---")
                        st.markdown("### 📊 AI Analysis Summary")
                        st.info(suggestions['summary'])

                    # Display recommendations
                    st.markdown("---")
                    st.markdown("### 🎯 Top Recommendations")

                    for rec in suggestions['recommendations']:
                        priority_emoji = {
                            'High': '🔴',
                            'Medium': '🟡',
                            'Low': '🟢'
                        }

                        with st.expander(
                            f"{priority_emoji.get(rec.get('priority', 'Medium'), '⚪')} "
                            f"**{rec.get('item_title', 'Unknown')}** - {rec.get('file_name', 'Unknown')} "
                            f"(Score: {rec.get('score', 0)}/100)",
                            expanded=True
                        ):
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown(f"**Priority:** {rec.get('priority', 'N/A')}")
                                st.markdown(f"**File:** `{rec.get('file_name', 'N/A')}`")

                                st.markdown("**Why It's Valuable:**")
                                st.write(rec.get('rationale', 'No rationale provided'))

                                st.markdown("**Knowledge Value:**")
                                st.write(rec.get('knowledge_value', 'No knowledge value provided'))

                            with col2:
                                st.metric("Score", f"{rec.get('score', 0)}/100")

                                # Quick action buttons
                                item_idx = rec.get('item_index', -1)
                                if item_idx >= 0 and item_idx < len(st.session_state.archive_results):
                                    result = st.session_state.archive_results[item_idx]

                                    # Find the matching file
                                    matching_file = None
                                    for file in result['files']:
                                        if file['name'] == rec.get('file_name'):
                                            matching_file = file
                                            break

                                    if matching_file:
                                        # Check if file already exists in KB by source URL
                                        # Support both formats:
                                        # - New format (file URL): https://archive.org/download/item-id/filename.pdf
                                        # - Old format (item URL): https://archive.org/details/item-id
                                        existing_doc = None
                                        item_id = result['identifier']

                                        for doc in kb.documents.values():
                                            if not hasattr(doc, 'source_url') or not doc.source_url:
                                                continue

                                            # Try exact file URL match (new format)
                                            if doc.source_url == matching_file['url']:
                                                existing_doc = doc
                                                break

                                            # Try item identifier match (old format)
                                            # Check if source_url contains the item identifier
                                            if item_id in doc.source_url:
                                                # Also check if filename matches (ignore extension for flexibility)
                                                safe_filename = Path(matching_file['name']).stem  # Get filename without extension
                                                doc_stem = Path(doc.filename).stem  # Get doc filename without extension
                                                if safe_filename.lower() in doc_stem.lower() or doc_stem.lower() in safe_filename.lower():
                                                    existing_doc = doc
                                                    break

                                        if existing_doc:
                                            # File already in KB - show status
                                            st.success(f"✅ In KB")
                                            st.caption(f"Doc: {existing_doc.doc_id[:12]}...")
                                        else:
                                            # File not in KB - show Quick Add and Download buttons
                                            if st.button("⚡ Quick Add", key=f"ai_add_{item_idx}_{rec.get('file_name')}"):
                                                try:
                                                    with st.spinner(f"Downloading and adding {matching_file['name']}..."):
                                                        import urllib.request
                                                        import tempfile

                                                        # Create temp directory in data_dir (within allowed paths)
                                                        temp_dir = Path(st.session_state.data_dir) / "temp"
                                                        temp_dir.mkdir(exist_ok=True)

                                                        # Download to temp file in allowed directory
                                                        # Extract just the filename (no directory path) to avoid path issues
                                                        safe_filename = Path(matching_file['name']).name
                                                        temp_filename = f"quick_add_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_filename}"
                                                        tmp_path = temp_dir / temp_filename
                                                        urllib.request.urlretrieve(matching_file['url'], str(tmp_path))

                                                        # Add to knowledge base
                                                        title = f"{result['title']} - {matching_file['name']}"
                                                        tags = result['subject'] if isinstance(result['subject'], list) else [result['subject']]
                                                        if isinstance(tags, str):
                                                            tags = [tags]

                                                        # Add document first
                                                        doc = kb.add_document(
                                                            str(tmp_path),
                                                            title=title,
                                                            tags=tags
                                                        )

                                                        # Update with source URL metadata (use file URL for matching)
                                                        with kb.db_conn:
                                                            cursor = kb.db_conn.cursor()
                                                            cursor.execute("""
                                                                UPDATE documents
                                                                SET source_url = ?,
                                                                    scrape_date = ?,
                                                                    scrape_status = 'success'
                                                                WHERE doc_id = ?
                                                            """, (matching_file['url'], datetime.now(timezone.utc).isoformat(), doc.doc_id))

                                                        # Update in-memory object
                                                        doc.source_url = matching_file['url']
                                                        kb.documents[doc.doc_id] = doc

                                                        # Record in quick-added history
                                                        st.session_state.quick_added_files.append({
                                                            'title': title,
                                                            'file_name': matching_file['name'],
                                                            'source_url': result['url'],
                                                            'doc_id': doc.doc_id,
                                                            'status': 'success',
                                                            'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                                                        })

                                                        # Clean up temp file
                                                        tmp_path.unlink()

                                                        st.success(f"✅ Added to knowledge base!\nDoc ID: {doc.doc_id[:12]}...")
                                                        st.rerun()

                                                except Exception as e:
                                                    # Clean up temp file if it exists
                                                    if 'tmp_path' in locals() and tmp_path.exists():
                                                        tmp_path.unlink()

                                                    # Record failed attempt
                                                    st.session_state.quick_added_files.append({
                                                        'title': f"{result['title']} - {matching_file['name']}",
                                                        'file_name': matching_file['name'],
                                                        'source_url': result['url'],
                                                        'doc_id': None,
                                                        'status': 'failed',
                                                        'error': str(e),
                                                        'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
                                                    })
                                                    st.error(f"Quick add failed: {str(e)}")

                                        if st.button("💾 Download", key=f"ai_dl_{item_idx}_{rec.get('file_name')}"):
                                            try:
                                                with st.spinner(f"Downloading {matching_file['name']}..."):
                                                    import urllib.request
                                                    downloads_dir = Path(st.session_state.data_dir) / "downloads"
                                                    downloads_dir.mkdir(exist_ok=True)
                                                    # Extract just the filename (no directory path) to avoid path issues
                                                    safe_filename = Path(matching_file['name']).name
                                                    filepath = downloads_dir / safe_filename
                                                    urllib.request.urlretrieve(matching_file['url'], filepath)
                                                    st.success(f"✅ Downloaded to {filepath}")

                                            except Exception as e:
                                                st.error(f"Download failed: {str(e)}")

                    # Check if all recommendations are already in KB
                    st.markdown("---")
                    in_kb_count = 0
                    total_recs = len(suggestions['recommendations'])

                    for rec in suggestions['recommendations']:
                        item_idx = rec.get('item_index', -1)
                        if item_idx >= 0 and item_idx < len(st.session_state.archive_results):
                            result = st.session_state.archive_results[item_idx]

                            # Find matching file
                            matching_file = None
                            for file in result['files']:
                                if file['name'] == rec.get('file_name'):
                                    matching_file = file
                                    break

                            if matching_file:
                                # Check if in KB (same logic as above)
                                existing_doc = None
                                item_id = result['identifier']

                                for doc in kb.documents.values():
                                    if not hasattr(doc, 'source_url') or not doc.source_url:
                                        continue

                                    if doc.source_url == matching_file['url']:
                                        existing_doc = doc
                                        break

                                    if item_id in doc.source_url:
                                        safe_filename = Path(matching_file['name']).stem
                                        doc_stem = Path(doc.filename).stem
                                        if safe_filename.lower() in doc_stem.lower() or doc_stem.lower() in safe_filename.lower():
                                            existing_doc = doc
                                            break

                                if existing_doc:
                                    in_kb_count += 1

                    # Show status and regenerate option if all are in KB
                    if in_kb_count == total_recs and total_recs > 0:
                        st.warning(f"⚠️ All {total_recs} recommendations are already in your knowledge base!")
                        if st.button("🔄 Generate New Recommendations (excluding files already in KB)", type="primary"):
                            # Build exclusion list from files already in KB
                            excluded_files = []
                            for rec in suggestions['recommendations']:
                                item_idx = rec.get('item_index', -1)
                                if item_idx >= 0 and item_idx < len(st.session_state.archive_results):
                                    result = st.session_state.archive_results[item_idx]

                                    # Find matching file
                                    matching_file = None
                                    for file in result['files']:
                                        if file['name'] == rec.get('file_name'):
                                            matching_file = file
                                            break

                                    if matching_file:
                                        # Check if in KB
                                        existing_doc = None
                                        item_id = result['identifier']

                                        for doc in kb.documents.values():
                                            if not hasattr(doc, 'source_url') or not doc.source_url:
                                                continue

                                            if doc.source_url == matching_file['url']:
                                                existing_doc = doc
                                                break

                                            if item_id in doc.source_url:
                                                safe_filename = Path(matching_file['name']).stem
                                                doc_stem = Path(doc.filename).stem
                                                if safe_filename.lower() in doc_stem.lower() or doc_stem.lower() in safe_filename.lower():
                                                    existing_doc = doc
                                                    break

                                        if existing_doc:
                                            excluded_files.append({
                                                'item_title': result['title'],
                                                'file_name': matching_file['name']
                                            })

                            # Store exclusions in session state for regeneration
                            st.session_state.ai_exclusions = excluded_files
                            st.session_state.ai_suggestions = None  # Clear current suggestions
                            st.rerun()
                    elif in_kb_count > 0:
                        st.info(f"📊 Status: {in_kb_count}/{total_recs} recommendations already in KB, {total_recs - in_kb_count} new files available")
                        if st.button("🔄 Regenerate (excluding files already in KB)", type="secondary"):
                            # Build exclusion list from files already in KB
                            excluded_files = []
                            for rec in suggestions['recommendations']:
                                item_idx = rec.get('item_index', -1)
                                if item_idx >= 0 and item_idx < len(st.session_state.archive_results):
                                    result = st.session_state.archive_results[item_idx]

                                    # Find matching file
                                    matching_file = None
                                    for file in result['files']:
                                        if file['name'] == rec.get('file_name'):
                                            matching_file = file
                                            break

                                    if matching_file:
                                        # Check if in KB
                                        existing_doc = None
                                        item_id = result['identifier']

                                        for doc in kb.documents.values():
                                            if not hasattr(doc, 'source_url') or not doc.source_url:
                                                continue

                                            if doc.source_url == matching_file['url']:
                                                existing_doc = doc
                                                break

                                            if item_id in doc.source_url:
                                                safe_filename = Path(matching_file['name']).stem
                                                doc_stem = Path(doc.filename).stem
                                                if safe_filename.lower() in doc_stem.lower() or doc_stem.lower() in safe_filename.lower():
                                                    existing_doc = doc
                                                    break

                                        if existing_doc:
                                            excluded_files.append({
                                                'item_title': result['title'],
                                                'file_name': matching_file['name']
                                            })

                            # Store exclusions in session state for regeneration
                            st.session_state.ai_exclusions = excluded_files
                            st.session_state.ai_suggestions = None  # Clear current suggestions
                            st.rerun()
                    else:
                        st.info(f"📊 Status: All {total_recs} recommendations are new files!")

        # ========== QUICK ADDED TAB ==========
        with tab3:
            st.subheader("⚡ Quick Added Files")
            st.markdown("Files added directly to the knowledge base using Quick Add.")

            if st.session_state.quick_added_files:
                st.write(f"**Total quick-added:** {len(st.session_state.quick_added_files)}")

                # Add clear all button
                if st.button("🗑️ Clear History"):
                    st.session_state.quick_added_files = []
                    st.success("✅ History cleared")
                    st.rerun()

                st.markdown("---")

                # Display quick-added files in reverse order (newest first)
                for idx, entry in enumerate(reversed(st.session_state.quick_added_files)):
                    # Get actual index in the list (since we're iterating reversed)
                    actual_idx = len(st.session_state.quick_added_files) - 1 - idx

                    with st.container():
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

                        with col1:
                            st.text(f"📄 {entry['title']}")
                            st.caption(f"Source: {entry['source_url'][:60]}..." if len(entry['source_url']) > 60 else f"Source: {entry['source_url']}")

                        with col2:
                            # Status indicator
                            if entry['status'] == 'success':
                                st.success(f"✅ {entry['status'].upper()}")
                                st.caption(f"Doc ID: {entry['doc_id'][:12]}...")
                            else:
                                st.error(f"❌ {entry['status'].upper()}")
                                if 'error' in entry:
                                    st.caption(f"Error: {entry['error'][:50]}...")

                        with col3:
                            st.caption(f"Added: {entry['timestamp']}")
                            if entry['file_name']:
                                st.caption(f"File: {entry['file_name']}")

                        with col4:
                            # Add remove button for individual entries
                            if st.button("🗑️", key=f"remove_quick_{actual_idx}", help="Remove this entry"):
                                st.session_state.quick_added_files.pop(actual_idx)
                                st.success("✅ Entry removed")
                                st.rerun()

                        st.markdown("---")
            else:
                st.info("👆 No files have been quick-added yet. Use the Quick Add button in the Search Archive or AI Suggestions tabs.")

        # ========== DOWNLOADS TAB ==========
        with tab4:
            st.subheader("📥 Downloaded Files")

            downloads_dir = Path(st.session_state.data_dir) / "downloads"

            if downloads_dir.exists():
                files = list(downloads_dir.glob("*"))

                if files:
                    st.write(f"**Location:** `{downloads_dir}`")
                    st.write(f"**Total files:** {len(files)}")

                    # File list
                    for file in files:
                        col1, col2, col3, col4 = st.columns([3, 1, 1.5, 1])

                        with col1:
                            st.text(f"📄 {file.name}")

                        with col2:
                            size_mb = file.stat().st_size / (1024 * 1024)
                            st.caption(f"{size_mb:.2f} MB")

                        with col3:
                            # Check if file exists in KB (check both session state and actual database)
                            file_key = file.name
                            existing_doc = None

                            # First check session state
                            if file_key in st.session_state.downloaded_files_added:
                                doc_id = st.session_state.downloaded_files_added[file_key]
                                # Verify it still exists in KB
                                if doc_id in kb.documents:
                                    existing_doc = kb.documents[doc_id]

                            # If not in session state, search KB by filename
                            if not existing_doc:
                                for doc in kb.documents.values():
                                    # Check if document's file_path ends with this filename
                                    if hasattr(doc, 'file_path') and doc.file_path and file.name in str(doc.file_path):
                                        existing_doc = doc
                                        break

                            if existing_doc:
                                # File exists in KB - show status
                                st.success(f"✅ In KB")
                                st.caption(f"Doc: {existing_doc.doc_id[:12]}...")
                            else:
                                # File not in KB - show Add button
                                if st.button("➕ Add to KB", key=f"add_{file.name}"):
                                    try:
                                        with st.spinner(f"Adding {file.name}..."):
                                            doc = kb.add_document(
                                                str(file),
                                                title=file.stem
                                            )
                                            # Track that this file has been added
                                            st.session_state.downloaded_files_added[file_key] = doc.doc_id
                                            st.success(f"✅ Added!\nDoc ID: {doc.doc_id[:12]}...")
                                            st.rerun()
                                    except Exception as e:
                                        st.error(f"Error: {str(e)}")

                        with col4:
                            if st.button("🗑️ Delete", key=f"delete_{file.name}"):
                                try:
                                    file.unlink()
                                    # Remove from tracking if it was added
                                    if file_key in st.session_state.downloaded_files_added:
                                        del st.session_state.downloaded_files_added[file_key]
                                    st.success(f"✅ Deleted {file.name}")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                    # Bulk actions
                    st.markdown("---")
                    st.subheader("Bulk Actions")
                    bulk_col1, bulk_col2 = st.columns(2)

                    with bulk_col1:
                        if st.button("➕ Add All to Knowledge Base"):
                            success_count = 0
                            error_count = 0

                            progress_bar = st.progress(0)
                            status_text = st.empty()

                            for i, file in enumerate(files):
                                file_key = file.name
                                # Skip if already added
                                if file_key in st.session_state.downloaded_files_added:
                                    continue

                                try:
                                    status_text.text(f"Adding {file.name}...")
                                    doc = kb.add_document(str(file), title=file.stem)
                                    # Track that this file has been added
                                    st.session_state.downloaded_files_added[file_key] = doc.doc_id
                                    success_count += 1
                                except Exception as e:
                                    st.warning(f"Failed to add {file.name}: {str(e)}")
                                    error_count += 1

                                progress_bar.progress((i + 1) / len(files))

                            status_text.text("")
                            progress_bar.empty()

                            st.success(f"✅ Added {success_count} files")
                            if error_count > 0:
                                st.warning(f"⚠️ {error_count} files failed")

                            st.rerun()

                    with bulk_col2:
                        if st.button("🗑️ Clear All Downloads"):
                            if st.checkbox("Confirm deletion of all downloaded files"):
                                try:
                                    for file in files:
                                        file.unlink()
                                        # Remove from tracking
                                        file_key = str(file)
                                        if file_key in st.session_state.downloaded_files_added:
                                            del st.session_state.downloaded_files_added[file_key]
                                    st.success(f"✅ Deleted {len(files)} files")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error: {str(e)}")

                else:
                    st.info("No downloaded files yet. Use the Search tab to find and download documents.")
            else:
                st.info("Downloads directory will be created when you download your first file.")
