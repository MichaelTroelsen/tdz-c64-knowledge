"""Admin GUI page: 📉 System Analytics.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import pandas as pd
import streamlit as st


def render(kb):
    st.title("📉 Search Analytics")
    col1, col2 = st.columns([1, 3])
    with col1:
        days = st.selectbox("Time Range", [7, 14, 30, 60, 90], index=2)
    if st.button("📊 Generate Report"):
        try:
            analytics = kb.get_search_analytics(days=days, limit=100)

            if 'error' in analytics:
                st.error(f"Error: {analytics['error']}")
            else:
                # Overview metrics
                st.subheader("📊 Overview")
                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric("Total Searches", f"{analytics.get('total_searches', 0):,}")
                with col2:
                    st.metric("Unique Queries", f"{analytics.get('unique_queries', 0):,}")
                with col3:
                    st.metric("Avg Results", f"{analytics.get('avg_results', 0):.1f}")
                with col4:
                    st.metric("Avg Time", f"{analytics.get('avg_execution_time_ms', 0):.1f}ms")

                st.markdown("---")

                # Two columns for charts
                col1, col2 = st.columns(2)

                with col1:
                    # Top queries
                    st.subheader("🔝 Top Queries")
                    if analytics.get('top_queries'):
                        df = pd.DataFrame(analytics['top_queries'][:10])
                        st.dataframe(df, hide_index=True, use_container_width=True)
                    else:
                        st.info("No query data available")

                with col2:
                    # Search modes
                    st.subheader("🔍 Search Mode Usage")
                    if analytics.get('search_modes'):
                        df = pd.DataFrame(analytics['search_modes'])
                        st.bar_chart(df.set_index('mode')['count'])
                    else:
                        st.info("No search mode data available")

                st.markdown("---")

                # Failed searches
                st.subheader("❌ Failed Searches (0 results)")
                if analytics.get('failed_searches'):
                    df = pd.DataFrame(analytics['failed_searches'][:10])
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.info("No failed searches")

                # Popular tags
                if analytics.get('popular_tags'):
                    st.markdown("---")
                    st.subheader("🏷️ Popular Tags")
                    df = pd.DataFrame(analytics['popular_tags'][:10])
                    st.bar_chart(df.set_index('tag')['count'])

        except Exception as e:
            st.error(f"Error generating analytics: {str(e)}")
