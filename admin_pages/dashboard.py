"""Admin GUI page: 📊 Dashboard.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
import streamlit as st
from admin_common import format_bytes


def render(kb):
    st.title("📊 Knowledge Base Dashboard")
    stats = kb.get_stats()
    health = kb.health_check()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Documents", stats['total_documents'])
    with col2:
        st.metric("Chunks", f"{stats['total_chunks']:,}")
    with col3:
        st.metric("Total Words", f"{stats['total_words']:,}")
    with col4:
        status_color = "🟢" if health['status'] == 'healthy' else "🔴"
        st.metric("Status", f"{status_color} {health['status'].upper()}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📁 Database Info")
        if health['database']:
            st.write(f"**Size:** {format_bytes(health['database'].get('size_mb', 0) * 1024 * 1024)}")
            st.write(f"**Integrity:** {health['database'].get('integrity', 'Unknown')}")
            st.write(f"**Free Disk Space:** {health['database'].get('disk_free_gb', 0):.2f} GB")
    with col2:
        st.subheader("⚙️ Features")
        if health['features']:
            for feature, enabled in health['features'].items():
                icon = "✅" if enabled else "❌"
                st.write(f"{icon} {feature.replace('_', ' ').title()}")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📄 File Types")
        if stats['file_types']:
            for ftype in stats['file_types']:
                st.write(f"• {ftype}")
        else:
            st.info("No documents yet")
    with col2:
        st.subheader("🏷️ Tags")
        if stats['all_tags']:
            for tag in stats['all_tags']:
                st.write(f"• {tag}")
        else:
            st.info("No tags yet")
    if health.get('issues'):
        st.markdown("---")
        st.error("⚠️ **Issues Detected:**")
        for issue in health['issues']:
            st.write(f"• {issue}")
