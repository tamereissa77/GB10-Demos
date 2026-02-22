"""
app.py
======
Streamlit front-end for the Arabic Financial Document Extraction pipeline.

Two extraction modes:
  A) Table Extraction: Unstructured segments → select tables → Chandra OCR
  B) Full Page OCR:    Send entire page to Chandra → get all labeled elements

Post-extraction pipeline stores results to:
  - PostgreSQL   (structured table data)
  - ArangoDB     (knowledge graph triples via Ollama)
  - Qdrant       (vector embeddings via sentence-transformers)
  - MinIO        (cropped images + text references for citation)
"""

import io
import logging
import re
from typing import List, Tuple, Dict, Any

import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont

# Local modules
from processor import file_bytes_to_images, process_document
from model_handler import (
    load_model,
    extract_table_from_image,
    extract_page_layout,
    parse_layout_blocks,
    _get_native_prompt,
    _get_layout_prompt,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Arabic Financial Document Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .stApp { background-color: #0e1117; }
    .rtl-text {
        direction: rtl; text-align: right;
        font-family: 'Noto Sans Arabic', 'Segoe UI', Tahoma, sans-serif;
    }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 8px 12px; border: 1px solid #30363d; }
    th { background-color: #161b22; }
    .backend-ok { color: #4ECDC4; }
    .backend-err { color: #FF6B6B; }
    .pipeline-badge {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.8em; margin: 2px;
    }
    .badge-pg { background: #336791; color: white; }
    .badge-kg { background: #6C3483; color: white; }
    .badge-vec { background: #1ABC9C; color: white; }
    .badge-obj { background: #E67E22; color: white; }
    .edit-hint {
        background: #1a1a2e; border: 1px solid #FFE66D; border-radius: 6px;
        padding: 8px 12px; margin: 6px 0; font-size: 0.85em; color: #FFE66D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Label colors for layout visualization
# ---------------------------------------------------------------------------
LABEL_COLORS = {
    "Table": "#FF6B6B",
    "Text": "#4ECDC4",
    "Section-Header": "#FFE66D",
    "Caption": "#A8E6CF",
    "Footnote": "#DDA0DD",
    "Image": "#87CEEB",
    "Figure": "#87CEEB",
    "Page-Header": "#FFA07A",
    "Page-Footer": "#FFA07A",
    "List-Group": "#98D8C8",
    "Equation-Block": "#F7DC6F",
    "Form": "#BB8FCE",
    "Complex-Block": "#F0B27A",
    "Code-Block": "#82E0AA",
    "Table-Of-Contents": "#AED6F1",
}

DEFAULT_COLOR = "#CCCCCC"


def get_label_color(label: str) -> str:
    return LABEL_COLORS.get(label, DEFAULT_COLOR)


# ---------------------------------------------------------------------------
# Cached Model Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading Chandra VLM (Qwen3-VL) ... first run downloads weights")
def get_model():
    model, processor, device = load_model()
    return model, processor, device


# ---------------------------------------------------------------------------
# Cached Pipeline Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Connecting to storage backends...")
def get_pipeline():
    """Initialize the post-extraction pipeline (lazy connections)."""
    from pipeline import ExtractionPipeline
    return ExtractionPipeline()


# ---------------------------------------------------------------------------
# Helper Utilities
# ---------------------------------------------------------------------------

def _flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            " | ".join(str(c) for c in col if str(c) not in ("", "nan"))
            if isinstance(col, tuple) else str(col)
            for col in df.columns
        ]
    return df


def html_tables_to_dataframes(html_text: str) -> List[pd.DataFrame]:
    results = []
    try:
        dfs = pd.read_html(io.StringIO(html_text))
        for df in dfs:
            df = _flatten_multiindex_columns(df)
            if not df.empty:
                results.append(df)
    except Exception:
        pass
    return results


def markdown_table_to_dataframe(md_text: str) -> pd.DataFrame | None:
    lines = [
        line.strip()
        for line in md_text.strip().splitlines()
        if line.strip() and not re.match(r"^\|?\s*[-:]+", line)
    ]
    if len(lines) < 2:
        return None

    def split_row(row: str) -> List[str]:
        return [c.strip() for c in row.strip().strip("|").split("|")]

    try:
        headers = split_row(lines[0])
        data = [split_row(r) for r in lines[1:]]
        return pd.DataFrame(data, columns=headers)
    except Exception:
        return None


def extract_tables_from_text(text: str) -> List[pd.DataFrame]:
    dfs = html_tables_to_dataframes(text)
    if dfs:
        return dfs
    df = markdown_table_to_dataframe(text)
    if df is not None and not df.empty:
        return [df]
    return []


def dataframe_to_excel_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Extracted Table")
    return buf.getvalue()


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


# ---------------------------------------------------------------------------
# Editable Table Widget (headers + add/remove columns)
# ---------------------------------------------------------------------------

def editable_table_widget(
    df: pd.DataFrame,
    widget_key: str,
    *,
    show_raw_html: bool = False,
    raw_html: str = "",
) -> pd.DataFrame:
    """
    Render a fully-editable table where **both headers and cells** can be
    modified, and the user can add / remove columns.

    The trick: we store the header as the first row of the DataFrame so that
    ``st.data_editor`` treats it as regular editable data.  On output we
    reconstruct the real DataFrame with the (possibly edited) header.

    Returns the final DataFrame with the edited header applied.
    """

    # --- session-state keys ---------------------------------------------------
    hdr_key   = f"_hdr_{widget_key}"       # edited header list
    addcol_key = f"_addcol_{widget_key}"    # flag: user clicked "Add Column"
    delcol_key = f"_delcol_{widget_key}"    # column index to delete

    # Initialise header in session state on first render
    if hdr_key not in st.session_state:
        st.session_state[hdr_key] = list(df.columns.astype(str))

    current_headers: list = st.session_state[hdr_key]

    # --- Handle "Add Column" --------------------------------------------------
    if st.session_state.get(addcol_key, False):
        new_col_name = f"Column {len(current_headers) + 1}"
        current_headers.append(new_col_name)
        st.session_state[hdr_key] = current_headers
        st.session_state[addcol_key] = False

    # --- Handle "Delete Column" -----------------------------------------------
    del_idx = st.session_state.get(delcol_key, -1)
    if isinstance(del_idx, int) and 0 <= del_idx < len(current_headers):
        current_headers.pop(del_idx)
        st.session_state[hdr_key] = current_headers
        st.session_state[delcol_key] = -1

    # --- Build a working DataFrame with generic column names ------------------
    # Ensure the data has the same number of columns as the header
    data_rows = df.values.tolist()
    n_cols = len(current_headers)

    # Pad or trim each row to match the header width
    adjusted_rows = []
    for row in data_rows:
        row = list(row)
        if len(row) < n_cols:
            row.extend([""] * (n_cols - len(row)))
        elif len(row) > n_cols:
            row = row[:n_cols]
        adjusted_rows.append(row)

    # Use generic column names for the data editor (Col_0, Col_1, ...)
    generic_cols = [f"Col_{i}" for i in range(n_cols)]
    work_df = pd.DataFrame(adjusted_rows, columns=generic_cols) if adjusted_rows else pd.DataFrame(columns=generic_cols)

    # Convert all values to strings so TextColumn config is always compatible
    work_df = work_df.astype(str)
    work_df = work_df.replace({"nan": "", "None": ""})

    # --- Editable header row --------------------------------------------------
    st.markdown(
        '<div class="edit-hint">'
        "✏️ <b>Editable headers</b> — change column names below. "
        "Use <b>+ Add Column</b> / <b>🗑 Remove</b> to adjust the table structure."
        "</div>",
        unsafe_allow_html=True,
    )

    hdr_cols = st.columns(n_cols + 2)  # extra cols for buttons
    new_headers = []
    for i in range(n_cols):
        with hdr_cols[i]:
            val = st.text_input(
                f"Header {i+1}",
                value=current_headers[i],
                key=f"{widget_key}_hdr_{i}",
                label_visibility="collapsed",
            )
            new_headers.append(val)
    # Persist any header edits
    st.session_state[hdr_key] = new_headers

    with hdr_cols[n_cols]:
        def _add_col(k=addcol_key):
            st.session_state[k] = True
        st.button("➕ Col", key=f"{widget_key}_add_col_btn",
                  on_click=_add_col,
                  help="Add a new column")

    with hdr_cols[n_cols + 1]:
        if n_cols > 1:
            del_choice = st.selectbox(
                "Del col",
                options=list(range(n_cols)),
                format_func=lambda x: new_headers[x] if x < len(new_headers) else f"Col {x+1}",
                key=f"{widget_key}_del_sel",
                label_visibility="collapsed",
            )
            def _del_col(k=delcol_key, v=del_choice):
                st.session_state[k] = v
            st.button("🗑", key=f"{widget_key}_del_col_btn",
                      on_click=_del_col,
                      help="Remove selected column")

    # --- Editable data cells --------------------------------------------------
    st.markdown(
        '<div class="edit-hint">'
        "✏️ Click any cell to edit. Add / delete <b>rows</b> with the toolbar."
        "</div>",
        unsafe_allow_html=True,
    )

    edited_work = st.data_editor(
        work_df,
        width="stretch",
        num_rows="dynamic",
        key=f"{widget_key}_data",
        column_config={
            generic_cols[i]: st.column_config.TextColumn(new_headers[i])
            for i in range(n_cols)
        },
    )

    # --- Reconstruct final DataFrame with real headers ------------------------
    final_df = edited_work.copy()
    final_df.columns = new_headers

    return final_df


def draw_layout_boxes(
    page_image: Image.Image,
    blocks: List[Dict[str, Any]],
) -> Image.Image:
    """Draw labeled bounding boxes on the page image."""
    overlay = page_image.copy().convert("RGBA")
    draw = ImageDraw.Draw(overlay)

    for block in blocks:
        label = block["label"]
        bbox = block["bbox"]
        color = get_label_color(label)

        draw.rectangle(bbox, outline=color, width=3)

        tag = f" {label} "
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), tag, font=font)
        tw = text_bbox[2] - text_bbox[0]
        th = text_bbox[3] - text_bbox[1]

        tag_x = bbox[0]
        tag_y = max(bbox[1] - th - 4, 0)
        draw.rectangle([tag_x, tag_y, tag_x + tw + 4, tag_y + th + 4], fill=color)
        draw.text((tag_x + 2, tag_y + 2), tag, fill="black", font=font)

    return overlay.convert("RGB")


def render_pipeline_badges(result: dict):
    """Render colored badges showing which backends received data."""
    backends = result.get("backends", {})
    badges = []
    for name, info in backends.items():
        status = info.get("status", "error") if isinstance(info, dict) else "error"
        if name == "postgres" and status == "ok":
            count = info.get("count", 0)
            badges.append(f'<span class="pipeline-badge badge-pg">🐘 PG: {count} table(s)</span>')
        elif name == "knowledge_graph" and status == "ok":
            count = info.get("triples_extracted", 0)
            badges.append(f'<span class="pipeline-badge badge-kg">🔗 KG: {count} triples</span>')
        elif name == "vector_db" and status == "ok":
            count = info.get("vectors_stored", 0)
            badges.append(f'<span class="pipeline-badge badge-vec">🔍 Vec: {count} vectors</span>')
        elif name == "object_store" and status == "ok":
            badges.append(f'<span class="pipeline-badge badge-obj">📦 Stored</span>')
        elif status == "error":
            err = info.get("error", "unknown") if isinstance(info, dict) else ""
            badges.append(f'<span class="pipeline-badge" style="background:#555;color:#FF6B6B;">❌ {name}</span>')

    if badges:
        st.markdown(" ".join(badges), unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/financial-analytics.png",
        width=64,
    )
    st.title("Settings")
    st.markdown("---")

    mode = st.radio(
        "Extraction Mode",
        ["Table Extraction", "Full Page OCR"],
        index=0,
        help=(
            "**Table Extraction**: Unstructured detects tables, then Chandra OCR extracts each.\n\n"
            "**Full Page OCR**: Chandra processes the entire page and identifies all elements "
            "(Tables, Text, Headers, Captions, Footnotes, Images, Forms, etc.) with bounding boxes."
        ),
    )

    st.markdown("---")
    st.markdown("### Chandra VLM")
    st.info(
        "**Chandra VLM** (`datalab-to/chandra`, Qwen3-VL) is loaded once "
        "and cached in GPU memory."
    )

    if mode == "Table Extraction":
        use_native = st.checkbox(
            "Use Chandra's native OCR prompt (recommended)",
            value=True,
        )
        if use_native:
            custom_prompt = "USE_NATIVE"
            st.caption("Using Chandra's built-in OCR prompt.")
        else:
            custom_prompt = st.text_area(
                "Custom prompt",
                value=_get_native_prompt(),
                height=180,
            )
    else:
        custom_prompt = "USE_LAYOUT"
        st.caption("Using Chandra's layout-OCR prompt with bounding boxes.")

    max_tokens = st.slider(
        "Max new tokens",
        min_value=1024,
        max_value=12384,
        value=8192 if mode == "Table Extraction" else 12384,
        step=512,
    )

    # Pipeline toggle
    st.markdown("---")
    st.markdown("### 💾 Storage Pipeline")
    enable_pipeline = st.checkbox(
        "Enable post-extraction storage",
        value=True,
        help=(
            "When enabled, extracted data is automatically stored to:\n"
            "- **PostgreSQL** — structured table data\n"
            "- **ArangoDB** — knowledge graph (triples via Ollama)\n"
            "- **Qdrant** — vector embeddings for semantic search\n"
            "- **MinIO** — cropped images with text references"
        ),
    )

    if enable_pipeline:
        st.caption("✅ Tables → PostgreSQL")
        st.caption("✅ Text → Knowledge Graph (Ollama → ArangoDB)")
        st.caption("✅ All content → Vector DB (Qdrant)")
        st.caption("✅ Images → Object Store (MinIO)")

        if st.button("🔍 Check Backend Status", use_container_width=True):
            pipeline = get_pipeline()
            stats = pipeline.get_all_stats()
            for name, info in stats.items():
                status = info.get("status", "unknown")
                if status in ("connected", "ok"):
                    st.markdown(f'<span class="backend-ok">✅ {name}</span>', unsafe_allow_html=True)
                else:
                    err = info.get("error", "")
                    st.markdown(f'<span class="backend-err">❌ {name}: {err[:60]}</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Streamlit + Unstructured + Chandra VLM (Qwen3-VL)")
    st.caption("PostgreSQL · ArangoDB · Qdrant · MinIO · Ollama")

# ---------------------------------------------------------------------------
# Main Area
# ---------------------------------------------------------------------------
st.title("Arabic Financial Document Extractor")
st.markdown(
    "Upload an Arabic financial document (PDF or image). "
    f"**Mode: {mode}**"
)

uploaded_file = st.file_uploader(
    "Upload a document",
    type=["pdf", "png", "jpg", "jpeg", "tiff", "tif", "bmp", "webp"],
)

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    filename = uploaded_file.name
    _file_id = f"{filename}_{len(file_bytes)}"

    if st.session_state.get("_file_id") != _file_id:
        st.session_state["_file_id"] = _file_id
        st.session_state.pop("_page_images", None)
        st.session_state.pop("_table_results", None)
        st.session_state.pop("_extracted_tables", None)
        st.session_state.pop("_ocr_page_data", None)
        for key in list(st.session_state.keys()):
            if key.startswith("sel_"):
                del st.session_state[key]

    # Step A: Page images (cached)
    st.markdown("## Uploaded Document")
    if "_page_images" not in st.session_state:
        try:
            st.session_state["_page_images"] = file_bytes_to_images(file_bytes, filename)
        except Exception as exc:
            st.error(f"Could not read the uploaded file: {exc}")
            st.stop()

    page_images = st.session_state["_page_images"]
    cols = st.columns(min(len(page_images), 3))
    for idx, img in enumerate(page_images):
        with cols[idx % len(cols)]:
            st.image(img, caption=f"Page {idx + 1}", width="stretch")

    # ======================================================================
    # MODE: Full Page OCR
    # ======================================================================
    if mode == "Full Page OCR":
        st.markdown("## Full Page OCR with Layout Detection")
        st.markdown(
            "Select pages to process. Chandra will identify **all elements** "
            "(Tables, Text, Headers, Captions, Footnotes, Images, Forms, etc.) "
            "with bounding boxes."
        )

        page_options = [f"Page {i+1}" for i in range(len(page_images))]
        selected_pages = st.multiselect(
            "Select pages to OCR",
            options=page_options,
            default=page_options[:1],
        )
        selected_page_indices = [int(p.split()[-1]) - 1 for p in selected_pages]

        if not selected_page_indices:
            st.warning("Select at least one page.")
            st.stop()

        ocr_btn = st.button(
            f"Run Full Page OCR on {len(selected_page_indices)} page(s)",
            type="primary",
            use_container_width=True,
        )

        # --- Phase 1: Run OCR and save results to session state ---
        if ocr_btn:
            try:
                model, processor, device = get_model()
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
                st.stop()

            ocr_results = {}  # page_idx -> {blocks, raw_html, annotated}

            for page_idx in selected_page_indices:
                page_img = page_images[page_idx]

                with st.spinner(f"Running Chandra layout-OCR on page {page_idx + 1}..."):
                    raw_html = extract_page_layout(
                        image=page_img,
                        model=model,
                        processor=processor,
                        device=device,
                        max_new_tokens=max_tokens,
                    )

                if raw_html.startswith("[ERROR]"):
                    st.error(raw_html)
                    continue

                blocks = parse_layout_blocks(raw_html, page_img)
                if not blocks:
                    st.warning(f"Page {page_idx+1}: No layout blocks detected.")
                    continue

                annotated = draw_layout_boxes(page_img, blocks)

                ocr_results[page_idx] = {
                    "blocks": blocks,
                    "raw_html": raw_html,
                    "annotated": annotated,
                }

            st.session_state["_ocr_page_data"] = ocr_results
            st.session_state["_extraction_filename"] = filename

        # --- Phase 2: Display results with editable tables ---
        if "_ocr_page_data" in st.session_state:
            ocr_data = st.session_state["_ocr_page_data"]

            for page_idx, page_data in sorted(ocr_data.items()):
                page_img = page_images[page_idx]
                blocks = page_data["blocks"]
                raw_html = page_data["raw_html"]
                annotated = page_data["annotated"]

                st.markdown(f"### Page {page_idx + 1}")

                col_ann, col_details = st.columns([1, 1])
                with col_ann:
                    st.image(annotated, caption="Annotated Layout", width="stretch")
                with col_details:
                    label_counts = {}
                    for b in blocks:
                        label_counts[b["label"]] = label_counts.get(b["label"], 0) + 1
                    st.markdown("**Detected Elements:**")
                    summary_df = pd.DataFrame(
                        [{"Element Type": k, "Count": v} for k, v in sorted(label_counts.items())]
                    )
                    st.dataframe(summary_df, hide_index=True)

                # Show each block with editable tables
                st.markdown("#### Extracted Elements")

                for b_idx, block in enumerate(blocks):
                    label = block["label"]
                    bbox = block["bbox"]

                    with st.expander(
                        f"{label} #{b_idx + 1}  —  bbox: {bbox}",
                        expanded=(label == "Table"),
                    ):
                        if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                            cropped = page_img.crop(bbox)
                            st.image(cropped, caption=f"{label} region", width="stretch")

                        st.markdown("**Content (HTML):**")
                        st.code(block["content_html"], language="html")

                        if block["content_text"]:
                            st.markdown("**Content (Text):**")
                            st.markdown(
                                f'<div class="rtl-text">{block["content_text"]}</div>',
                                unsafe_allow_html=True,
                            )

                        # Editable table for Table blocks
                        if label == "Table":
                            dfs = html_tables_to_dataframes(block["content_html"])
                            for t_idx, df in enumerate(dfs):
                                edited_df = editable_table_widget(
                                    df,
                                    widget_key=f"edit_p{page_idx}_b{b_idx}_t{t_idx}",
                                )
                                dl1, dl2 = st.columns(2)
                                with dl1:
                                    st.download_button(
                                        "📥 CSV", dataframe_to_csv_bytes(edited_df),
                                        f"page{page_idx+1}_block{b_idx+1}.csv",
                                        "text/csv",
                                        key=f"lcsv_{page_idx}_{b_idx}_{t_idx}",
                                    )
                                with dl2:
                                    st.download_button(
                                        "📥 Excel", dataframe_to_excel_bytes(edited_df),
                                        f"page{page_idx+1}_block{b_idx+1}.xlsx",
                                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                        key=f"lxls_{page_idx}_{b_idx}_{t_idx}",
                                    )

                with st.expander("Raw VLM Output"):
                    st.code(raw_html, language="html")

                st.markdown("---")

            # --- Phase 3: Confirm & Store button ---
            if enable_pipeline:
                table_count = sum(
                    1
                    for pd_data in ocr_data.values()
                    for b in pd_data["blocks"]
                    if b["label"] == "Table"
                )
                total_elements = sum(len(pd_data["blocks"]) for pd_data in ocr_data.values())

                st.markdown("## ✅ Confirm & Store to Backends")
                st.markdown(
                    f"**{total_elements}** element(s) across **{len(ocr_data)}** page(s) "
                    f"({table_count} tables). "
                    "Review and edit tables above, then click to store."
                )

                if st.button(
                    f"✅ Confirm & Store All Pages",
                    type="primary",
                    use_container_width=True,
                    key="confirm_store_ocr",
                ):
                    pipeline = get_pipeline()

                    for page_idx, page_data in sorted(ocr_data.items()):
                        page_img = page_images[page_idx]
                        blocks = page_data["blocks"]

                        st.markdown(f"#### 💾 Storing Page {page_idx + 1}...")

                        # Build dataframes_map from edited data_editors
                        dataframes_map = {}
                        for b_idx, block in enumerate(blocks):
                            if block["label"] == "Table":
                                collected_dfs = []
                                # Try to read from the editable_table_widget state
                                # The widget stores data under {key}_data and headers under _hdr_{key}
                                # Check all possible sub-table indices (t0, t1, t2, ...)
                                for t_idx in range(20):  # up to 20 sub-tables per block
                                    editor_key = f"edit_p{page_idx}_b{b_idx}_t{t_idx}_data"
                                    hdr_key = f"_hdr_edit_p{page_idx}_b{b_idx}_t{t_idx}"
                                    if editor_key in st.session_state:
                                        edited_df = st.session_state[editor_key]
                                        if isinstance(edited_df, pd.DataFrame) and not edited_df.empty:
                                            edited_df = edited_df.copy()
                                            # Apply the edited headers
                                            if hdr_key in st.session_state:
                                                headers = st.session_state[hdr_key]
                                                if len(headers) == len(edited_df.columns):
                                                    edited_df.columns = headers
                                            collected_dfs.append(edited_df)
                                    else:
                                        break  # No more sub-tables

                                if collected_dfs:
                                    dataframes_map[b_idx] = collected_dfs
                                    logger.info(
                                        "Page %d block %d: got %d table(s) from widget state",
                                        page_idx + 1, b_idx, len(collected_dfs),
                                    )
                                else:
                                    # Fallback: parse from HTML
                                    logger.warning(
                                        "Page %d block %d: no widget state found, "
                                        "falling back to HTML parse. "
                                        "Checked keys like edit_p%d_b%d_t0_data",
                                        page_idx + 1, b_idx, page_idx, b_idx,
                                    )
                                    dfs = html_tables_to_dataframes(block["content_html"])
                                    if dfs:
                                        dataframes_map[b_idx] = dfs
                                        logger.info(
                                            "Page %d block %d: fallback parsed %d table(s) from HTML",
                                            page_idx + 1, b_idx, len(dfs),
                                        )
                                    else:
                                        # Last resort: create a single-cell DataFrame from text
                                        content_text = block.get("content_text", "").strip()
                                        if content_text:
                                            fallback_df = pd.DataFrame({"Content": [content_text]})
                                            dataframes_map[b_idx] = [fallback_df]
                                            logger.info(
                                                "Page %d block %d: last-resort text fallback",
                                                page_idx + 1, b_idx,
                                            )

                        logger.info(
                            "Page %d: dataframes_map has %d table blocks: %s",
                            page_idx + 1,
                            len(dataframes_map),
                            {k: [df.shape for df in v] for k, v in dataframes_map.items()},
                        )

                        with st.spinner(f"Page {page_idx + 1}: KG + embeddings + storage..."):
                            page_pipeline_result = pipeline.process_page(
                                blocks=blocks,
                                page_image=page_img,
                                document_name=filename,
                                page_number=page_idx + 1,
                                dataframes_map=dataframes_map,
                            )

                        pipeline_results = page_pipeline_result.get("elements", [])
                        page_kg = page_pipeline_result.get("page_kg", {})
                        page_vec = page_pipeline_result.get("page_vectors", {})

                        pg_count = sum(
                            1 for r in pipeline_results
                            if r.get("backends", {}).get("postgres", {}).get("status") == "ok"
                        )
                        kg_triples = page_kg.get("triples_extracted", 0)
                        vec_count = page_vec.get("vectors_stored", 0)
                        obj_count = sum(
                            1 for r in pipeline_results
                            if r.get("backends", {}).get("object_store", {}).get("status") == "ok"
                        )

                        mcols = st.columns(4)
                        with mcols[0]:
                            st.metric("🐘 Tables → PG", pg_count)
                        with mcols[1]:
                            st.metric("🔗 KG Triples", kg_triples)
                        with mcols[2]:
                            st.metric("🔍 Vectors", vec_count)
                        with mcols[3]:
                            st.metric("📦 Images", obj_count)

                        if page_kg.get("status") == "ok" and page_kg.get("triples"):
                            with st.expander(f"🔗 Page {page_idx+1} KG — {kg_triples} triples"):
                                for triple in page_kg["triples"]:
                                    st.markdown(
                                        f"  `{triple.get('subject', '')}` → "
                                        f"**{triple.get('predicate', '')}** → "
                                        f"`{triple.get('object', '')}`"
                                    )

                        for elem in pipeline_results:
                            render_pipeline_badges(elem)

                        st.markdown("---")

                    st.success("✅ All pages stored successfully!")
                    del st.session_state["_ocr_page_data"]

    # ======================================================================
    # MODE: Table Extraction
    # ======================================================================
    else:
        # Step B: Segment (cached)
        st.markdown("## Table Detection & Segmentation")
        if "_table_results" not in st.session_state:
            with st.spinner("Running Unstructured layout analysis..."):
                try:
                    st.session_state["_table_results"] = process_document(
                        file_bytes, filename
                    )
                except Exception as exc:
                    st.error(f"Segmentation failed: {exc}")
                    st.stop()

        table_results = st.session_state["_table_results"]

        if not table_results:
            st.warning("No tables were detected in the document.")
            st.stop()

        st.success(f"Detected **{len(table_results)}** table region(s).")

        # Step C: Select
        st.markdown("## Select Tables to Extract")
        st.markdown("**Uncheck** any image to skip, then click **Extract**.")

        NUM_COLS = 4
        selected = {}
        for row_start in range(0, len(table_results), NUM_COLS):
            row_items = table_results[row_start : row_start + NUM_COLS]
            grid = st.columns(NUM_COLS)
            for ci, (crop, _el) in enumerate(row_items):
                ai = row_start + ci
                with grid[ci]:
                    st.image(crop, caption=f"Table {ai+1}", width="stretch")
                    selected[ai] = st.checkbox(f"Extract {ai+1}", True, key=f"sel_{ai}")

        sc1, sc2, _ = st.columns([1, 1, 4])
        with sc1:
            if st.button("Select All"):
                for k in selected:
                    st.session_state[f"sel_{k}"] = True
                st.rerun()
        with sc2:
            if st.button("Deselect All"):
                for k in selected:
                    st.session_state[f"sel_{k}"] = False
                st.rerun()

        chosen = [i for i, c in selected.items() if c]
        if not chosen:
            st.warning("No tables selected.")
            st.stop()

        st.info(f"**{len(chosen)}** of {len(table_results)} selected.")

        if st.button(
            f"Extract {len(chosen)} Selected Table(s)",
            type="primary",
            use_container_width=True,
        ):
            st.markdown("## Chandra VLM Extraction")
            try:
                model, processor, device = get_model()
            except Exception as exc:
                st.error(f"Failed to load model: {exc}")
                st.stop()

            pbar = st.progress(0, "Starting VLM extraction...")
            extracted_tables = []

            for step, ai in enumerate(chosen):
                crop, _el = table_results[ai]
                pbar.progress(step / len(chosen), f"Table {ai+1} ({step+1}/{len(chosen)})")

                page_num = getattr(
                    getattr(_el, "metadata", None), "page_number", 1
                ) or 1

                st.markdown(f"### Table {ai + 1}")
                c1, c2 = st.columns(2)

                with c1:
                    st.image(crop, caption=f"Cropped Table {ai+1}", width="stretch")

                with c2:
                    with st.spinner(f"Extracting table {ai+1}..."):
                        raw = extract_table_from_image(
                            crop, model, processor, device,
                            prompt=custom_prompt,
                            max_new_tokens=max_tokens,
                        )

                    st.markdown("**Raw VLM Output:**")
                    st.code(raw, language="html")

                    dfs = extract_tables_from_text(raw)
                    if dfs:
                        st.markdown(
                            f"**Parsed {len(dfs)} table(s) — ✏️ Edit below before storing:**"
                            if len(dfs) > 1
                            else "**Parsed Table — ✏️ Edit below before storing:**"
                        )
                        edited_dfs = []
                        for ti, df in enumerate(dfs):
                            if len(dfs) > 1:
                                st.markdown(f"**Sub-table {ti+1}**")
                            edited_df = editable_table_widget(
                                df,
                                widget_key=f"tedit_{ai}_{ti}",
                            )
                            edited_dfs.append(edited_df)
                            sfx = f"_{ti+1}" if len(dfs) > 1 else ""
                            d1, d2 = st.columns(2)
                            with d1:
                                st.download_button(
                                    f"📥 CSV{' #'+str(ti+1) if len(dfs)>1 else ''}",
                                    dataframe_to_csv_bytes(edited_df),
                                    f"table_{ai+1}{sfx}.csv", "text/csv",
                                    key=f"csv_{ai}_{ti}",
                                )
                            with d2:
                                st.download_button(
                                    f"📥 Excel{' #'+str(ti+1) if len(dfs)>1 else ''}",
                                    dataframe_to_excel_bytes(edited_df),
                                    f"table_{ai+1}{sfx}.xlsx",
                                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    key=f"xlsx_{ai}_{ti}",
                                )
                        dfs = edited_dfs
                    else:
                        dfs = []
                        st.info("Could not auto-parse. Copy the raw output above.")

                    extracted_tables.append({
                        "image": crop,
                        "raw_text": raw,
                        "raw_html": raw,
                        "dataframes": dfs,
                        "page_number": page_num,
                        "element_index": ai,
                        "bbox": [0, 0, crop.width, crop.height],
                    })

                st.markdown("---")

            pbar.progress(1.0, "VLM extraction complete!")

            st.session_state["_extracted_tables"] = extracted_tables
            st.session_state["_extraction_filename"] = filename

        # --- Confirm & Store for Table Extraction mode ---
        if "_extracted_tables" in st.session_state and enable_pipeline:
            extracted_tables = st.session_state["_extracted_tables"]
            store_filename = st.session_state.get("_extraction_filename", filename)

            tables_with_data = sum(
                1 for t in extracted_tables if t.get("dataframes")
            )

            st.markdown("---")
            st.markdown("## ✅ Confirm & Store to Backends")
            st.markdown(
                f"**{tables_with_data}** table(s) ready to store. "
                "Review and edit the tables above, then click the button below "
                "to send the **edited** data to all storage backends."
            )

            if st.button(
                f"✅ Confirm & Store {tables_with_data} Table(s)",
                type="primary",
                use_container_width=True,
                key="confirm_store_btn",
            ):
                pipeline = get_pipeline()

                st.markdown("### 💾 Page-by-Page Storage Pipeline")

                with st.spinner("Running page-by-page KG + embedding + storage..."):
                    page_results = pipeline.process_tables_by_page(
                        tables=extracted_tables,
                        document_name=store_filename,
                    )

                for page_num, page_result in sorted(page_results.items()):
                    page_kg = page_result.get("page_kg", {})
                    page_vec = page_result.get("page_vectors", {})
                    elements = page_result.get("elements", [])

                    st.markdown(f"#### Page {page_num}")

                    pg_count = sum(
                        1 for r in elements
                        if r.get("backends", {}).get("postgres", {}).get("status") == "ok"
                    )
                    kg_triples = page_kg.get("triples_extracted", 0)
                    vec_count = page_vec.get("vectors_stored", 0)
                    obj_count = sum(
                        1 for r in elements
                        if r.get("backends", {}).get("object_store", {}).get("status") == "ok"
                    )

                    mcols = st.columns(4)
                    with mcols[0]:
                        st.metric("🐘 Tables → PG", pg_count)
                    with mcols[1]:
                        st.metric("🔗 KG Triples", kg_triples)
                    with mcols[2]:
                        st.metric("🔍 Vectors", vec_count)
                    with mcols[3]:
                        st.metric("📦 Images", obj_count)

                    if page_kg.get("status") == "ok" and page_kg.get("triples"):
                        with st.expander(f"🔗 Page {page_num} KG — {kg_triples} triples"):
                            for triple in page_kg["triples"]:
                                st.markdown(
                                    f"  `{triple.get('subject', '')}` → "
                                    f"**{triple.get('predicate', '')}** → "
                                    f"`{triple.get('object', '')}`"
                                )

                    for elem in elements:
                        render_pipeline_badges(elem)

                    st.markdown("---")

                st.success("✅ All tables stored successfully!")
                del st.session_state["_extracted_tables"]

else:
    st.markdown(
        '<div style="text-align:center;padding:60px 20px;">'
        '<h2 style="color:#888;">Upload a document to get started</h2>'
        '<p style="color:#666;">Supported: PDF, PNG, JPG, TIFF, BMP, WebP</p>'
        '<p style="color:#555;font-size:0.9em;">'
        "Storage: PostgreSQL · ArangoDB · Qdrant · MinIO"
        "</p>"
        "</div>",
        unsafe_allow_html=True,
    )
