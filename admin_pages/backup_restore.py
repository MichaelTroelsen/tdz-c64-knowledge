"""Admin GUI page: 💾 Backup & Restore.

Moved out of admin_gui.py's 16-branch if/elif chain by R18. `kb` is a
parameter rather than a module global so this module never imports
admin_gui, which dispatches to it.
"""
from pathlib import Path
import streamlit as st


def render(kb):
    st.title("💾 Backup & Restore")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📦 Create Backup")

        backup_dir = st.text_input("Backup Directory", value=str(Path.home() / "c64kb_backups"))
        compress = st.checkbox("Compress to ZIP", value=True)

        if st.button("🔄 Create Backup"):
            try:
                with st.spinner("Creating backup..."):
                    backup_path = kb.create_backup(backup_dir, compress)
                    st.success(f"✅ Backup created successfully!\n\n**Location:** `{backup_path}`")
            except Exception as e:
                st.error(f"Backup failed: {str(e)}")
    with col2:
        st.subheader("♻️ Restore Backup")

        st.warning("⚠️ **Warning:** Restoring will replace the current database. A safety backup will be created automatically.")

        restore_path = st.text_input("Backup Path (file or directory)")
        verify = st.checkbox("Verify backup before restoring", value=True)

        if st.button("⚠️ Restore Backup", type="primary"):
            if not restore_path:
                st.error("Please provide a backup path")
            else:
                try:
                    with st.spinner("Restoring backup..."):
                        result = kb.restore_from_backup(restore_path, verify)
                        st.success(f"✅ Restore completed successfully!\n\n"
                                 f"**Documents restored:** {result['restored_documents']}\n"
                                 f"**Time:** {result['elapsed_seconds']:.2f}s")
                        st.rerun()
                except Exception as e:
                    st.error(f"Restore failed: {str(e)}")
