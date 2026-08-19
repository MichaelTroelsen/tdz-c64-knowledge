"""Admin GUI page: 🛰️ MCP Monitor.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import pandas as pd
import streamlit as st


def render(kb):
    st.title("🛰️ MCP Monitor")
    st.write("Usage stats for every MCP tool call, logged to `mcp_call_log` so recurring problems and popular tools are visible instead of buried in `server.log`.")
    window_label = st.selectbox("Time Window", ["Last hour", "Last 6 hours", "Last 24 hours", "Last 7 days"], index=2)
    window_hours = {"Last hour": 1, "Last 6 hours": 6, "Last 24 hours": 24, "Last 7 days": 24 * 7}[window_label]
    stats = kb.get_mcp_call_stats(hours=window_hours)
    if stats['total_calls'] == 0:
        st.info(f"No MCP tool calls logged in the {window_label.lower()}. Calls are recorded automatically as Claude Code (or any other MCP client) uses this server's tools.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("📞 Total Calls", f"{stats['total_calls']:,}")
        with col2:
            st.metric("❌ Errors", f"{stats['error_count']:,}", delta=f"{stats['error_rate']:.1%} error rate", delta_color="inverse")
        with col3:
            st.metric("⏱️ Avg Latency", f"{stats['avg_duration_ms']:.0f} ms")
        with col4:
            st.metric("🐌 Max Latency", f"{stats['max_duration_ms']:.0f} ms")

        st.markdown("---")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**📊 Calls Over Time**")
            bucket_minutes = 15 if window_hours <= 6 else (60 if window_hours <= 24 else 60 * 6)
            timeline = kb.get_mcp_calls_over_time(hours=window_hours, bucket_minutes=bucket_minutes)
            if timeline:
                df_timeline = pd.DataFrame(timeline).set_index('bucket')
                st.line_chart(df_timeline[['calls', 'errors']])
            else:
                st.caption("No data to chart yet.")

        with col2:
            st.markdown("**🏆 Top Tools by Call Count**")
            if stats['top_tools']:
                df_tools = pd.DataFrame(stats['top_tools']).set_index('tool_name')
                st.bar_chart(df_tools['calls'])
            else:
                st.caption("No data to chart yet.")

        st.markdown("**🔧 Per-Tool Breakdown**")
        df_breakdown = pd.DataFrame(stats['top_tools'])
        if not df_breakdown.empty:
            df_breakdown['avg_duration_ms'] = df_breakdown['avg_duration_ms'].round(1)
            st.dataframe(
                df_breakdown.rename(columns={
                    'tool_name': 'Tool', 'calls': 'Calls',
                    'avg_duration_ms': 'Avg ms', 'errors': 'Errors'
                }),
                use_container_width=True, hide_index=True
            )

        st.markdown("---")
        st.markdown("**📋 Recent Calls**")
        col1, col2, col3 = st.columns([2, 1, 1])
        with col1:
            all_tool_names = sorted({t['tool_name'] for t in stats['top_tools']})
            filter_tool = st.selectbox("Filter by tool", ["All tools"] + all_tool_names)
        with col2:
            only_errors = st.checkbox("Errors only", value=False)
        with col3:
            call_limit = st.number_input("Max rows", min_value=10, max_value=1000, value=100, step=10)

        recent = kb.get_recent_mcp_calls(
            limit=call_limit,
            tool_name=None if filter_tool == "All tools" else filter_tool,
            only_errors=only_errors
        )
        if recent:
            df_recent = pd.DataFrame(recent)
            df_recent['duration_ms'] = df_recent['duration_ms'].round(1)
            st.dataframe(
                df_recent.rename(columns={
                    'call_id': 'ID', 'tool_name': 'Tool', 'called_at': 'When',
                    'duration_ms': 'ms', 'success': 'OK',
                    'error_message': 'Error', 'args_summary': 'Args'
                }),
                use_container_width=True, hide_index=True
            )
        else:
            st.caption("No calls match the current filters.")
