
import os
import json
import streamlit as st
from langchain_nvidia_ai_endpoints import ChatNVIDIA, NVIDIAEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

try:
    import nest_asyncio
    nest_asyncio.apply()
except (ValueError, ImportError):
    pass

# --- Configuration ---
st.set_page_config(layout="wide", page_title="Data Governance RAG Demo")

os.environ["NVIDIA_API_KEY"] = "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"

BASE_DIR = os.path.dirname(__file__)
DOCUMENTS_PATH = os.path.join(BASE_DIR, "documents.json")
GROUPS_PATH = os.path.join(BASE_DIR, "groups.json")
TABLE_ACCESS_PATH = os.path.join(BASE_DIR, "table_access.json")
USERS_PATH = os.path.join(BASE_DIR, "users.json")
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://govdemo:govdemo123@postgres:5432/governance")
ALL_ROLES = ["intern", "hr", "finance", "admin"]
CLASSIFICATIONS = ["Confidential", "Internal", "Public"]

# =====================================================================
# HELPERS — FILES
# =====================================================================

def load_documents():
    if os.path.exists(DOCUMENTS_PATH):
        with open(DOCUMENTS_PATH, "r") as f:
            return json.load(f)
    return []

def save_documents(docs):
    with open(DOCUMENTS_PATH, "w") as f:
        json.dump(docs, f, indent=2, ensure_ascii=False)

def load_groups():
    if os.path.exists(GROUPS_PATH):
        with open(GROUPS_PATH, "r") as f:
            return json.load(f)
    return {"General Staff": ["intern"]}

def save_groups(groups):
    with open(GROUPS_PATH, "w") as f:
        json.dump(groups, f, indent=2, ensure_ascii=False)

def load_table_access():
    if os.path.exists(TABLE_ACCESS_PATH):
        with open(TABLE_ACCESS_PATH, "r") as f:
            return json.load(f)
    return {}

def save_table_access(ta):
    with open(TABLE_ACCESS_PATH, "w") as f:
        json.dump(ta, f, indent=2, ensure_ascii=False)

def load_users():
    if os.path.exists(USERS_PATH):
        with open(USERS_PATH, "r") as f:
            return json.load(f)
    return {"Alice": "HR Team", "Bob": "Finance Team", "Charlie": "General Staff"}

def save_users(users):
    with open(USERS_PATH, "w") as f:
        json.dump(users, f, indent=2, ensure_ascii=False)

# =====================================================================
# HELPERS — VECTOR STORE
# =====================================================================

def build_vector_store(docs_list):
    """Build a fresh in-memory ChromaDB — new client each time."""
    import chromadb
    client = chromadb.Client()
    embeddings = NVIDIAEmbeddings(model="nvidia/nv-embedqa-e5-v5")
    docs = [
        Document(
            page_content=d["text"],
            metadata={
                "source": d["source"],
                "allowed_roles": d["allowed_roles"],
                "description": d.get("description", ""),
                "classification": d.get("classification", "Public"),
                "owner": d.get("owner", "Unknown"),
            }
        )
        for d in docs_list
    ]
    if not docs:
        return Chroma(client=client, collection_name="governance_demo_store",
                      embedding_function=embeddings)
    return Chroma.from_documents(documents=docs, client=client,
                                 embedding=embeddings,
                                 collection_name="governance_demo_store")

# =====================================================================
# HELPERS — POSTGRESQL
# =====================================================================

def query_postgres(sql, params=None):
    """Execute a read-only SQL query and return rows as list of dicts."""
    import psycopg2
    import psycopg2.extras
    try:
        conn = psycopg2.connect(DATABASE_URL)
        conn.set_session(readonly=True, autocommit=True)
        cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        cur.execute(sql, params or ())
        rows = [dict(r) for r in cur.fetchall()]
        cur.close()
        conn.close()
        return rows
    except Exception as e:
        return {"error": str(e)}

def get_table_schema(table_name):
    """Get column names and types for a table."""
    rows = query_postgres(
        "SELECT column_name, data_type FROM information_schema.columns "
        "WHERE table_name = %s ORDER BY ordinal_position",
        (table_name,)
    )
    if isinstance(rows, dict) and "error" in rows:
        return None
    return rows

def test_postgres_connection():
    """Quick check if postgres is reachable."""
    result = query_postgres("SELECT 1 AS ok")
    return not (isinstance(result, dict) and "error" in result)

# =====================================================================
# HELPERS — UI
# =====================================================================

def classification_badge(classification):
    colors = {
        "Confidential": ("#dc3545", "🔴"),
        "Internal": ("#ffc107", "🟡"),
        "Public": ("#28a745", "🟢"),
    }
    color, dot = colors.get(classification, ("#6c757d", "⚪"))
    return f'<span style="background:{color};color:#fff;padding:2px 10px;border-radius:12px;font-size:0.8em;font-weight:600;">{dot} {classification}</span>'

def has_any_role(user_roles, allowed_roles):
    return bool(set(user_roles) & set(allowed_roles))

def render_card(col, title, icon, badge_html, meta_lines, role_tags_html, access, grant_reason_html=""):
    """Render a styled access card (works for both files and tables)."""
    access_class = "access-granted" if access else "access-denied"
    status_text = "✅ ACCESS GRANTED" if access else "⛔ ACCESS DENIED"
    if access:
        banner_style = "margin-top:12px;padding:6px 12px;border-radius:8px;font-weight:700;font-size:0.95em;text-align:center;background:rgba(40,167,69,0.15);color:#28a745;border:1px solid rgba(40,167,69,0.3);"
    else:
        banner_style = "margin-top:12px;padding:6px 12px;border-radius:8px;font-weight:700;font-size:0.95em;text-align:center;background:rgba(220,53,69,0.15);color:#dc3545;border:1px solid rgba(220,53,69,0.3);"
    parts = [
        f'<div class="file-card {access_class}">',
        f'<h4>{icon} {title}</h4>',
        f'<div class="meta-row">{badge_html}</div>',
    ]
    for line in meta_lines:
        parts.append(f'<div class="meta-row">{line}</div>')
    parts.append(f'<div class="meta-row">🔑 Allowed: {role_tags_html}</div>')
    if grant_reason_html:
        parts.append(grant_reason_html)
    parts.append(f'<div style="{banner_style}">{status_text}</div>')
    parts.append('</div>')
    col.markdown("\n".join(parts), unsafe_allow_html=True)

def build_role_tags(allowed_roles, current_roles):
    tags = ""
    matching = set(current_roles) & set(allowed_roles)
    for role in ALL_ROLES:
        if role in allowed_roles:
            if role in matching:
                tags += f'<span class="role-tag matched">{role} ✓</span>'
            else:
                tags += f'<span class="role-tag other">{role}</span>'
    return tags, matching

# =====================================================================
# CUSTOM CSS
# =====================================================================
st.markdown("""
<style>
    .file-card {
        border: 1px solid #2d2d2d; border-radius: 12px; padding: 18px;
        margin-bottom: 12px;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .file-card:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0,0,0,0.4); }
    .file-card.access-granted { border-left: 5px solid #28a745; }
    .file-card.access-denied  { border-left: 5px solid #dc3545; }
    .file-card h4 { margin: 0 0 8px 0; color: #e0e0e0; font-size: 1.1em; }
    .file-card .meta-row {
        display: flex; align-items: center; gap: 8px;
        margin: 4px 0; font-size: 0.9em; color: #b0b0b0;
    }
    .role-tag { display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 0.78em; margin: 2px; font-weight: 500; }
    .role-tag.matched  { background: #28a745; color: #fff; }
    .role-tag.inherited { background: #0d6efd; color: #fff; }
    .role-tag.other    { background: #3a3a5c; color: #ccc; }
    .privilege-table { width: 100%; border-collapse: collapse; font-size: 0.9em; }
    .privilege-table th, .privilege-table td { padding: 8px 12px; border: 1px solid #333; text-align: center; }
    .privilege-table th { background: #1a1a2e; color: #aaa; font-weight: 600; }
    .sidebar-summary {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 16px; margin: 8px 0; border: 1px solid #2d2d2d;
    }
    .sidebar-summary h3 { margin: 0 0 6px 0; font-size: 1em; color: #e0e0e0; }
    .sidebar-summary .big-num { font-size: 2em; font-weight: 700; color: #0d6efd; }
    .inheritance-chain {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px; padding: 14px; margin: 8px 0;
        border: 1px solid #2d2d2d; font-size: 0.9em;
    }
    .inheritance-chain .chain-arrow { color: #ffc107; font-weight: 700; margin: 0 6px; }
    .inheritance-chain .chain-item  { color: #e0e0e0; }
    .inheritance-chain .chain-roles { color: #0d6efd; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =====================================================================
# SESSION STATE
# =====================================================================
if "vector_store" not in st.session_state:
    docs = load_documents()
    st.session_state.vector_store = build_vector_store(docs)
    st.session_state.index_count = len(docs)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "user_groups" not in st.session_state:
    st.session_state.user_groups = load_users()

# =====================================================================
# LOAD DATA
# =====================================================================
documents = load_documents()
groups = load_groups()
table_access = load_table_access()

# =====================================================================
# SIDEBAR — USER & GROUP
# =====================================================================
st.sidebar.header("👤 User & Group")

user_name = st.sidebar.selectbox("Select User:", list(st.session_state.user_groups.keys()))

available_groups = list(groups.keys())
current_group = st.session_state.user_groups.get(user_name, available_groups[0] if available_groups else "")
group_index = available_groups.index(current_group) if current_group in available_groups else 0

selected_group = st.sidebar.selectbox(
    f"Assign **{user_name}** to group:",
    available_groups, index=group_index, key=f"group_selector_{user_name}"
)

if selected_group != st.session_state.user_groups.get(user_name):
    st.session_state.user_groups[user_name] = selected_group
    save_users(st.session_state.user_groups)
    st.rerun()

current_roles = groups.get(selected_group, [])

roles_display = ", ".join(f"<b>{r}</b>" for r in current_roles) if current_roles else "<i>none</i>"
st.sidebar.markdown(f'<div class="inheritance-chain"><span class="chain-item">👤 {user_name}</span><span class="chain-arrow">→</span><span class="chain-item">👥 {selected_group}</span><span class="chain-arrow">→</span><span class="chain-roles">🔑 {roles_display}</span></div>', unsafe_allow_html=True)

# Access Summary (docs + tables)
acc_docs = sum(1 for d in documents if has_any_role(current_roles, d.get("allowed_roles", [])))
acc_tables = sum(1 for t in table_access.values() if has_any_role(current_roles, t.get("allowed_roles", [])))
total_docs = len(documents)
total_tables = len(table_access)
st.sidebar.markdown(f"""
<div class="sidebar-summary">
<h3>📊 Access Summary</h3>
<div class="big-num">{acc_docs + acc_tables} / {total_docs + total_tables}</div>
<div style="color:#b0b0b0;font-size:0.85em;">data sources accessible ({acc_docs} files, {acc_tables} tables)</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("""
**Legend** · 🔴 Confidential · 🟡 Internal · 🟢 Public
""")
st.sidebar.divider()

# =====================================================================
# SIDEBAR — USER MANAGER
# =====================================================================
st.sidebar.header("🧑‍💼 User Manager")

user_action = st.sidebar.radio(
    "User action:", ["Add User", "Rename User", "Delete User"],
    horizontal=True, label_visibility="collapsed", key="user_mgr_action"
)

if user_action == "Add User":
    with st.sidebar.form("add_user_form", clear_on_submit=True):
        st.markdown("**➕ Add New User**")
        new_user_name = st.text_input("User Name", placeholder="e.g. Dana")
        new_user_group = st.selectbox("Assign to Group", list(groups.keys()))
        if st.form_submit_button("Add User", use_container_width=True):
            if new_user_name and new_user_name not in st.session_state.user_groups:
                st.session_state.user_groups[new_user_name] = new_user_group
                save_users(st.session_state.user_groups)
                st.toast(f"✅ Added user **{new_user_name}**", icon="🧑‍💼")
                st.rerun()
            elif new_user_name in st.session_state.user_groups:
                st.error("User already exists.")

elif user_action == "Rename User":
    existing_users = list(st.session_state.user_groups.keys())
    if existing_users:
        with st.sidebar.form("rename_user_form"):
            rename_target = st.selectbox("Select user:", existing_users)
            new_name = st.text_input("New Name", value=rename_target)
            if st.form_submit_button("Rename", use_container_width=True):
                if new_name and new_name != rename_target and new_name not in st.session_state.user_groups:
                    old_group = st.session_state.user_groups.pop(rename_target)
                    st.session_state.user_groups[new_name] = old_group
                    save_users(st.session_state.user_groups)
                    st.toast(f"✅ Renamed **{rename_target}** → **{new_name}**", icon="✏️")
                    st.rerun()

elif user_action == "Delete User":
    existing_users = list(st.session_state.user_groups.keys())
    if len(existing_users) > 1:
        del_user = st.sidebar.selectbox("Select user:", existing_users, key="del_user_select")
        if st.sidebar.button(f"🗑️ Delete {del_user}", use_container_width=True, type="primary"):
            del st.session_state.user_groups[del_user]
            save_users(st.session_state.user_groups)
            st.toast(f"🗑️ Deleted **{del_user}**", icon="🗑️")
            st.rerun()
    elif len(existing_users) == 1:
        st.sidebar.warning("Cannot delete the last user.")

st.sidebar.divider()

# =====================================================================
# SIDEBAR — GROUP MANAGER
# =====================================================================
st.sidebar.header("👥 Group Manager")

group_action = st.sidebar.radio(
    "Group action:", ["View", "Create", "Edit", "Delete"],
    horizontal=True, label_visibility="collapsed"
)

if group_action == "View":
    for g_name, g_roles in groups.items():
        st.sidebar.markdown(f"**{g_name}** → {', '.join(f'`{r}`' for r in g_roles)}")

elif group_action == "Create":
    with st.sidebar.form("create_group_form", clear_on_submit=True):
        st.markdown("**➕ New Group**")
        new_g_name = st.text_input("Group Name")
        new_g_roles = st.multiselect("Roles", ALL_ROLES)
        if st.form_submit_button("Create", use_container_width=True):
            if new_g_name and new_g_name not in groups:
                groups[new_g_name] = new_g_roles
                save_groups(groups)
                st.toast(f"✅ Created **{new_g_name}**", icon="👥")
                st.rerun()

elif group_action == "Edit":
    if groups:
        edit_g = st.sidebar.selectbox("Group:", list(groups.keys()))
        with st.sidebar.form("edit_group_form"):
            ed_roles = st.multiselect("Roles", ALL_ROLES, default=groups.get(edit_g, []))
            if st.form_submit_button("Save", use_container_width=True):
                groups[edit_g] = ed_roles
                save_groups(groups)
                st.toast(f"✅ Updated **{edit_g}**", icon="✏️")
                st.rerun()

elif group_action == "Delete":
    if groups:
        del_g = st.sidebar.selectbox("Group:", list(groups.keys()), key="del_group")
        if st.sidebar.button(f"🗑️ Delete {del_g}", use_container_width=True, type="primary"):
            del groups[del_g]
            save_groups(groups)
            fallback = list(groups.keys())[0] if groups else ""
            for u, g in st.session_state.user_groups.items():
                if g == del_g:
                    st.session_state.user_groups[u] = fallback
            st.rerun()

st.sidebar.divider()

# =====================================================================
# SIDEBAR — DOCUMENT MANAGER
# =====================================================================
st.sidebar.header("📝 Document Manager")

doc_action = st.sidebar.radio(
    "Doc:", ["Add", "Edit", "Delete"],
    horizontal=True, label_visibility="collapsed"
)

if doc_action == "Add":
    with st.sidebar.form("add_doc_form", clear_on_submit=True):
        st.markdown("**➕ Add Document**")
        new_source = st.text_input("File Name", placeholder="e.g. Policy.pdf")
        new_text = st.text_area("Content", height=80)
        new_desc = st.text_input("Description")
        new_classification = st.selectbox("Classification", CLASSIFICATIONS)
        new_owner = st.text_input("Owner")
        new_roles = st.multiselect("Allowed Roles", ALL_ROLES, default=["admin"])
        if st.form_submit_button("Add", use_container_width=True):
            if new_source and new_text:
                documents.append({"text": new_text, "source": new_source, "allowed_roles": new_roles,
                                  "description": new_desc, "classification": new_classification, "owner": new_owner})
                save_documents(documents)
                st.toast(f"✅ Added **{new_source}**", icon="📄")
                st.rerun()

elif doc_action == "Edit":
    if documents:
        source_names = [d["source"] for d in documents]
        edit_idx = st.sidebar.selectbox("File:", range(len(source_names)), format_func=lambda i: source_names[i])
        doc = documents[edit_idx]
        with st.sidebar.form("edit_doc_form"):
            ed_source = st.text_input("File Name", value=doc["source"])
            ed_text = st.text_area("Content", value=doc["text"], height=80)
            ed_desc = st.text_input("Description", value=doc.get("description", ""))
            ed_class = st.selectbox("Classification", CLASSIFICATIONS,
                                     index=CLASSIFICATIONS.index(doc.get("classification", "Public")))
            ed_owner = st.text_input("Owner", value=doc.get("owner", ""))
            ed_roles = st.multiselect("Allowed Roles", ALL_ROLES, default=doc.get("allowed_roles", []))
            if st.form_submit_button("Save", use_container_width=True):
                documents[edit_idx] = {"text": ed_text, "source": ed_source, "allowed_roles": ed_roles,
                                       "description": ed_desc, "classification": ed_class, "owner": ed_owner}
                save_documents(documents)
                st.toast(f"✅ Updated **{ed_source}**", icon="✏️")
                st.rerun()

elif doc_action == "Delete":
    if documents:
        source_names = [d["source"] for d in documents]
        del_idx = st.sidebar.selectbox("File:", range(len(source_names)), format_func=lambda i: source_names[i])
        if st.sidebar.button(f"🗑️ Delete {source_names[del_idx]}", use_container_width=True, type="primary"):
            removed = documents.pop(del_idx)
            save_documents(documents)
            st.toast(f"🗑️ Deleted **{removed['source']}**", icon="🗑️")
            st.rerun()

st.sidebar.divider()

# =====================================================================
# SIDEBAR — TABLE ACCESS MANAGER
# =====================================================================
st.sidebar.header("🗄️ Table Access Manager")

if table_access:
    for tbl_name, tbl_cfg in table_access.items():
        tbl_roles = tbl_cfg.get("allowed_roles", [])
        tbl_has = has_any_role(current_roles, tbl_roles)
        icon = "✅" if tbl_has else "⛔"
        st.sidebar.markdown(f"**{tbl_name}** {icon} → {', '.join(f'`{r}`' for r in tbl_roles)}")

    edit_tbl = st.sidebar.selectbox("Edit table access:", list(table_access.keys()), key="edit_tbl")
    with st.sidebar.form("edit_table_access_form"):
        tbl_cfg = table_access[edit_tbl]
        ta_roles = st.multiselect("Allowed Roles", ALL_ROLES, default=tbl_cfg.get("allowed_roles", []))
        ta_class = st.selectbox("Classification", CLASSIFICATIONS,
                                 index=CLASSIFICATIONS.index(tbl_cfg.get("classification", "Confidential")))
        ta_desc = st.text_input("Description", value=tbl_cfg.get("description", ""))
        if st.form_submit_button("Save Table Access", use_container_width=True):
            table_access[edit_tbl]["allowed_roles"] = ta_roles
            table_access[edit_tbl]["classification"] = ta_class
            table_access[edit_tbl]["description"] = ta_desc
            save_table_access(table_access)
            st.toast(f"✅ Updated **{edit_tbl}** access", icon="🗄️")
            st.rerun()
else:
    st.sidebar.info("No table access rules defined.")

st.sidebar.divider()

# --- Re-Index ---
if st.sidebar.button("🔄 Re-index Documents", use_container_width=True, type="primary"):
    with st.sidebar:
        with st.spinner("Re-embedding documents..."):
            fresh_docs = load_documents()
            st.session_state.vector_store = build_vector_store(fresh_docs)
            st.session_state.index_count = len(fresh_docs)
    st.toast(f"🔄 Re-indexed **{len(fresh_docs)}** documents!", icon="✅")
    st.rerun()

st.sidebar.caption(f"📦 Indexed: **{st.session_state.get('index_count', 0)}** docs")

# Check Postgres status
pg_ok = test_postgres_connection()
st.sidebar.caption(f"🗄️ PostgreSQL: {'🟢 Connected' if pg_ok else '🔴 Disconnected'}")

# =====================================================================
# MAIN AREA
# =====================================================================
st.title("🔐 Data Governance RAG Demo")
st.markdown("### Hybrid Document + Database Retrieval with Group-Based Access Control")

roles_str = ", ".join(current_roles) if current_roles else "none"
st.info(f"👤 **{user_name}** in group **{selected_group}** → roles: **{roles_str}**")

# =====================================================================
# FILE ACCESS DASHBOARD
# =====================================================================
st.subheader("📂 Document Access")

cols = st.columns(min(len(documents), 3) if documents else 1)
if not documents:
    st.info("No documents loaded.")

for i, doc in enumerate(documents):
    col = cols[i % len(cols)]
    allowed = doc.get("allowed_roles", [])
    access = has_any_role(current_roles, allowed)
    role_tags, matching = build_role_tags(allowed, current_roles)
    grant = f'<div class="meta-row" style="color:#28a745;">🔓 Via: <b>{", ".join(matching)}</b></div>' if matching else ""
    render_card(col, doc["source"], "📄",
                classification_badge(doc.get("classification", "Public")),
                [f"📝 {doc.get('description', 'No description')}",
                 f"👤 Owner: <b>{doc.get('owner', 'Unknown')}</b>"],
                role_tags, access, grant)

# =====================================================================
# TABLE ACCESS DASHBOARD
# =====================================================================
if table_access:
    st.subheader("🗄️ Database Table Access")
    t_cols = st.columns(min(len(table_access), 3))
    for i, (tbl_name, tbl_cfg) in enumerate(table_access.items()):
        col = t_cols[i % len(t_cols)]
        allowed = tbl_cfg.get("allowed_roles", [])
        access = has_any_role(current_roles, allowed)
        role_tags, matching = build_role_tags(allowed, current_roles)
        grant = f'<div class="meta-row" style="color:#28a745;">🔓 Via: <b>{", ".join(matching)}</b></div>' if matching else ""

        # Get row count if accessible
        meta_lines = [f"📝 {tbl_cfg.get('description', '')}",
                      f"👤 Owner: <b>{tbl_cfg.get('owner', 'Database')}</b>"]
        if access and pg_ok:
            count = query_postgres(f"SELECT COUNT(*) as cnt FROM {tbl_name}")
            if isinstance(count, list) and count:
                meta_lines.append(f"📊 <b>{count[0]['cnt']}</b> rows")

        render_card(col, tbl_name, "🗄️",
                    classification_badge(tbl_cfg.get("classification", "Confidential")),
                    meta_lines, role_tags, access, grant)

# =====================================================================
# PRIVILEGE MATRIX
# =====================================================================
all_sources = [(d["source"], d.get("allowed_roles", []), "📄") for d in documents]
all_sources += [(t, cfg.get("allowed_roles", []), "🗄️") for t, cfg in table_access.items()]

if all_sources and groups:
    with st.expander("🔐 Group × Data Source Privilege Matrix", expanded=False):
        g_names = list(groups.keys())
        header = "<tr><th>Source</th><th>Type</th>" + "".join(f"<th>{g}</th>" for g in g_names) + "</tr>"
        rows = ""
        for src_name, src_roles, src_icon in all_sources:
            cells = f"<td style='text-align:left;font-weight:600;'>{src_name}</td><td>{src_icon}</td>"
            for g in g_names:
                g_roles = groups[g]
                has = bool(set(g_roles) & set(src_roles))
                icon = "✅" if has else "❌"
                hl = "background:rgba(13,110,253,0.12);" if g == selected_group else ""
                cells += f"<td style='{hl}'>{icon}</td>"
            rows += f"<tr>{cells}</tr>"
        st.markdown(f'<table class="privilege-table"><thead>{header}</thead><tbody>{rows}</tbody></table>', unsafe_allow_html=True)

st.divider()

# =====================================================================
# CHAT INTERFACE — HYBRID AGENT
# =====================================================================
st.subheader("💬 Corporate Assistant Chat")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Ask a question about company data...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- 1. Document Retrieval (ChromaDB) ---
    doc_results = []
    seen = set()
    doc_filters_used = []
    for role in current_roles:
        f = {"allowed_roles": {"$contains": role}}
        doc_filters_used.append(f)
        try:
            retriever = st.session_state.vector_store.as_retriever(search_kwargs={"filter": f, "k": 3})
            for d in retriever.invoke(prompt):
                if d.page_content not in seen:
                    doc_results.append(d)
                    seen.add(d.page_content)
        except Exception:
            pass

    # --- 2. Database Retrieval (PostgreSQL) ---
    db_context = ""
    sql_query_used = ""
    db_results_raw = []
    accessible_tables = [t for t, cfg in table_access.items()
                         if has_any_role(current_roles, cfg.get("allowed_roles", []))]
    denied_tables = [t for t, cfg in table_access.items()
                     if not has_any_role(current_roles, cfg.get("allowed_roles", []))]

    if accessible_tables and pg_ok:
        # Build schema context for the LLM to generate SQL
        schema_info = ""
        for tbl in accessible_tables:
            cols = get_table_schema(tbl)
            if cols:
                col_defs = ", ".join(f"{c['column_name']} ({c['data_type']})" for c in cols)
                schema_info += f"Table: {tbl} — Columns: {col_defs}\n"

        if schema_info:
            # Ask LLM to generate SQL
            from langchain_openai import ChatOpenAI
            sql_llm = ChatOpenAI(base_url="http://llama:11434/v1", api_key="ollama",
                                  model="llama3.2:3b", temperature=0.0)

            sql_prompt = (
                f"You are a SQL expert. Given the following database schema:\n{schema_info}\n"
                f"Generate a PostgreSQL SELECT query to answer this question: {prompt}\n"
                f"Rules:\n"
                f"- Return ONLY the SQL query, nothing else. No explanation.\n"
                f"- Only use SELECT statements, never INSERT/UPDATE/DELETE.\n"
                f"- Only query tables listed above.\n"
                f"- For text searches, ALWAYS use ILIKE with % wildcards for partial matching.\n"
                f"  Example: WHERE employee_name ILIKE '%Mohamed%'\n"
                f"- Prefer SELECT * to return all columns unless the question asks for specific fields.\n"
                f"- If the question is not related to the data in these tables, respond with: NONE\n"
            )

            try:
                sql_response = sql_llm.invoke(sql_prompt).content.strip()
                # Clean up — remove markdown fences if present
                if sql_response.startswith("```"):
                    sql_response = sql_response.strip("`").strip()
                    if sql_response.lower().startswith("sql"):
                        sql_response = sql_response[3:].strip()

                if sql_response.upper() != "NONE" and sql_response.upper().startswith("SELECT"):
                    sql_query_used = sql_response
                    db_results_raw = query_postgres(sql_response)
                    if isinstance(db_results_raw, dict) and "error" in db_results_raw:
                        db_context = f"[Database query error: {db_results_raw['error']}]"
                        db_results_raw = []
                    elif db_results_raw:
                        # Format as readable text
                        rows_text = "\n".join(str(r) for r in db_results_raw[:20])
                        db_context = f"Database query results from table(s) {', '.join(accessible_tables)}:\n{rows_text}"
            except Exception as e:
                db_context = f"[Database query failed: {e}]"

    # --- 3. Governance Logs ---
    with st.expander("🔍 Governance Logs", expanded=True):
        log_col1, log_col2 = st.columns(2)

        with log_col1:
            st.markdown("**📄 Document Retrieval**")
            st.code(f"Filters: {doc_filters_used}", language="json")
            if doc_results:
                st.write(f"✅ Retrieved {len(doc_results)} chunks:")
                for d in doc_results:
                    st.json(d.metadata)
                    st.text(d.page_content[:200])
                    st.divider()
            else:
                st.error("⛔ No documents retrieved.")

        with log_col2:
            st.markdown("**🗄️ Database Retrieval**")
            if accessible_tables:
                st.write(f"✅ Accessible tables: {', '.join(accessible_tables)}")
                if sql_query_used:
                    st.code(sql_query_used, language="sql")
                if db_results_raw and isinstance(db_results_raw, list):
                    st.dataframe(db_results_raw, use_container_width=True)
                elif not sql_query_used:
                    st.info("No SQL generated (question may not relate to tables).")
            else:
                st.error(f"⛔ No table access. Denied: {', '.join(denied_tables)}")
            if denied_tables:
                st.warning(f"Tables denied: {', '.join(denied_tables)}")

    # --- 4. Generate Hybrid Answer ---
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(base_url="http://llama:11434/v1", api_key="ollama",
                      model="llama3.2:3b", temperature=0.1)

    system_prompt = (
        "You are a secure corporate assistant with access to both documents and database tables. "
        "Answer ONLY based on the provided context. If the information is not in the context, "
        "say it is not available to the user. Do not use outside knowledge. "
        "When answering, mention which source (document or database) the information comes from."
    )

    doc_context = "\n\n".join([d.page_content for d in doc_results]) if doc_results else ""
    full_context = ""
    if doc_context:
        full_context += f"=== DOCUMENT CONTEXT ===\n{doc_context}\n\n"
    if db_context:
        full_context += f"=== DATABASE CONTEXT ===\n{db_context}\n\n"

    if not full_context:
        ai_response = "I do not have access to that information based on your current permissions."
    else:
        full_prompt = f"{system_prompt}\n\n{full_context}\nQuestion: {prompt}"
        result = llm.invoke(full_prompt)
        ai_response = result.content

    with st.chat_message("assistant"):
        st.markdown(ai_response)

    st.session_state.messages.append({"role": "assistant", "content": ai_response})
