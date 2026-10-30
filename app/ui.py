"""Research Desk — premium local RAG workspace."""
from __future__ import annotations

import html
from datetime import datetime

import streamlit as st

from app.agent import run_research
from app.ingest import build_vector_store


st.set_page_config(
    page_title="Research Desk | Evidence Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root{--bg:#f5f6fa;--surface:#fff;--ink:#182235;--muted:#7b8598;--line:#e5e9f1;--accent:#5965e8;--nav:#111827}
    html,body,[class*="css"]{font-family:'DM Sans',sans-serif}.stApp{background:var(--bg);color:var(--ink)}
    .block-container{max-width:1560px;padding:1.5rem 2.2rem 3rem}
    [data-testid="stHeader"]{background:transparent}[data-testid="stToolbar"]{right:1rem}footer{visibility:hidden}
    [data-testid="stSidebar"]{background:var(--nav);border-right:1px solid #253047}
    [data-testid="stSidebar"] *{color:#edf1f8}[data-testid="stSidebar"] hr{border-color:#2a3549}
    [data-testid="stSidebar"] [data-testid="stRadio"] label{padding:8px 10px;border-radius:9px}
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover{background:#202a3b}
    .brand{display:flex;align-items:center;gap:11px;padding:8px 4px 22px}
    .brandmark{width:40px;height:40px;display:flex;align-items:center;justify-content:center;border-radius:12px;background:linear-gradient(135deg,#8792ff,#4d58d7);font:800 20px Manrope;color:white}
    .brandname{font:800 16px Manrope;letter-spacing:-.03em}.brandsub{font-size:10px;color:#9da9bd;letter-spacing:.14em;text-transform:uppercase;margin-top:3px}
    .navlabel{font-size:10px;font-weight:800;letter-spacing:.15em;color:#8190a7;margin:10px 0}
    .sidebarbox{background:#182234;border:1px solid #2a374b;border-radius:12px;padding:13px;font-size:12px;line-height:1.8}
    .breadcrumb{font-size:12px;color:var(--muted);margin-bottom:20px}
    .eyebrow{font-size:10px;font-weight:800;letter-spacing:.16em;color:var(--accent);margin-bottom:10px}
    h1.page-title{font:800 clamp(29px,3vw,42px)/1.13 Manrope;letter-spacing:-.045em;color:var(--ink);margin:0}
    .copy{font-size:14px;color:var(--muted);line-height:1.75;max-width:760px;margin:10px 0 24px}
    .surface{background:white;border:1px solid var(--line);border-radius:16px;padding:20px 21px;box-shadow:0 4px 18px #14203a06}
    .section-title{font:800 17px Manrope;letter-spacing:-.02em;color:var(--ink);margin-bottom:5px}
    .section-copy{font-size:12px;color:var(--muted);line-height:1.65;margin-bottom:15px}
    .metric{height:100%;background:#fff;border:1px solid var(--line);border-radius:15px;padding:17px 19px}
    .metriclabel{font-size:11px;font-weight:700;color:var(--muted)}.metricvalue{font:800 29px Manrope;letter-spacing:-.04em;color:var(--ink);margin-top:6px}.metricnote{font-size:11px;color:#9aa3b2;margin-top:3px}
    .file-row{display:flex;align-items:center;gap:12px;padding:12px 0;border-bottom:1px solid #eef0f5}
    .file-icon{width:38px;height:42px;display:flex;align-items:center;justify-content:center;background:#fff1f0;border:1px solid #ffe0dd;color:#d45b51;border-radius:9px;font-size:11px;font-weight:800;flex-shrink:0}
    .file-name{font-size:13px;font-weight:700;color:var(--ink);overflow-wrap:anywhere}.file-meta{font-size:11px;color:var(--muted);margin-top:4px}
    .tag{display:inline-block;background:#effaf4;color:#16835e;border:1px solid #d8efe4;border-radius:7px;padding:4px 8px;font-size:10px;font-weight:800}
    .answerpanel{background:white;border:1px solid var(--line);border-radius:15px;padding:22px 24px;line-height:1.8;overflow-wrap:anywhere}
    .stButton>button,.stDownloadButton>button{border-radius:10px;min-height:42px;font-weight:700;border:1px solid #dfe4ed}
    .stButton>button[kind="primary"]{background:var(--accent);border-color:var(--accent);color:white;box-shadow:0 5px 13px #5965e81c}
    .stButton>button[kind="primary"]:hover{background:#4754d4;border-color:#4754d4;transform:translateY(-1px)}
    .stTextArea textarea,.stTextInput input{border:1px solid #dce2ed!important;border-radius:11px!important;background:#fff!important;color:var(--ink)!important;line-height:1.65!important}
    .stTextArea textarea:focus,.stTextInput input:focus{border-color:#8791ef!important;box-shadow:0 0 0 3px #5965e817!important}
    .stTextArea textarea::placeholder,.stTextInput input::placeholder{color:#98a1b0!important;opacity:1!important}
    [data-testid="stFileUploader"]{background:#fafbff;border:1px dashed #c9d3e7;border-radius:12px;padding:10px}
    [data-testid="stExpander"]{background:white;border:1px solid var(--line);border-radius:11px}[data-testid="stAlert"]{border-radius:11px}hr{border-color:var(--line)}
    .smallmuted{font-size:11px;color:var(--muted);line-height:1.65}
    @media(max-width:900px){.block-container{padding:1.2rem 1rem 2rem}.surface{padding:16px}h1.page-title{font-size:30px}}
    </style>
    """,
    unsafe_allow_html=True,
)

DEFAULTS = {
    "vector_store": None,
    "indexed_sources": [],
    "chunk_count": 0,
    "last_result": None,
    "last_question": "",
    "research_history": [],
    "document_details": {},
    "page": "Research",
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def open_history_result(item: dict) -> None:
    """Run before the next script render so navigation state can be changed safely."""
    st.session_state.last_question = item["question"]
    st.session_state.last_result = {
        "answer": item.get("answer", ""),
        "citation_check": item.get("citation_check", ""),
        "trace": item.get("trace", []),
        "evidence": item.get("evidence", []),
    }
    st.session_state.page = "Research"
    # This callback executes before Streamlit instantiates the radio widget again.
    st.session_state.main_navigation = "Research"


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brandmark">R</div><div><div class="brandname">Research Desk</div><div class="brandsub">Evidence intelligence</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="navlabel">WORKSPACE</div>', unsafe_allow_html=True)
    nav = ["Research", "Documents", "Activity", "Settings"]
    if st.session_state.page not in nav:
        st.session_state.page = "Research"
    if "main_navigation" not in st.session_state:
        st.session_state.main_navigation = st.session_state.page
    page = st.radio(
        "Navigation",
        nav,
        index=nav.index(st.session_state.page),
        label_visibility="collapsed",
        key="main_navigation",
    )
    st.session_state.page = page
    st.markdown("---")
    st.markdown(
        '<div class="navlabel">LOCAL AI ENGINE</div><div class="sidebarbox">● &nbsp; Local engine<br>Ollama · llama3.2:3b<br>Sentence Transformers embeddings<br>ChromaDB retrieval</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.caption("Local processing. Citation checks do not prove factual entailment.")
    st.markdown("---")
    st.markdown(
        '<div class="smallmuted">RESEARCH DESK · LOCAL PROTOTYPE</div>',
        unsafe_allow_html=True,
    )

st.markdown(
    f'<div class="breadcrumb">Workspace&nbsp; / &nbsp;{html.escape(page)}</div>',
    unsafe_allow_html=True,
)
sources = st.session_state.indexed_sources
chunks = st.session_state.chunk_count
ready = st.session_state.vector_store is not None

if page == "Research":
    st.markdown(
        '<div class="eyebrow">EVIDENCE-FIRST AI</div><h1 class="page-title">Research, with receipts.</h1><div class="copy">Ask questions across your PDFs. Get concise answers, inspect retrieved evidence, and keep every source close to the claim.</div>',
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    for col, label, value, note in [
        (m1, "Documents indexed", len(sources), "Files in this session"),
        (m2, "Evidence chunks", chunks, "Searchable passages"),
        (m3, "Research engine", "Ready" if ready else "Waiting", "Local processing"),
    ]:
        with col:
            size = 25 if isinstance(value, str) else 29
            st.markdown(
                f'<div class="metric"><div class="metriclabel">{label}</div><div class="metricvalue" style="font-size:{size}px">{value}</div><div class="metricnote">{note}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    left, right = st.columns([1, 1.15], gap="large")
    with left:
        st.markdown(
            '<div class="surface"><div class="section-title">Build your evidence base</div><div class="section-copy">Add text-based PDFs to create a local searchable index.</div></div>',
            unsafe_allow_html=True,
        )
        uploads = st.file_uploader(
            "Choose PDF documents",
            type=["pdf"],
            accept_multiple_files=True,
            key="research_pdf_uploader",
            help="Scanned PDFs require OCR.",
        )
        if uploads:
            st.caption(f"{len(uploads)} PDF(s) selected")
            for file in uploads:
                st.markdown(
                    f'<div class="file-row"><div class="file-icon">PDF</div><div><div class="file-name">{html.escape(file.name)}</div><div class="file-meta">{file.size / 1024:.0f} KB · Ready to index</div></div></div>',
                    unsafe_allow_html=True,
                )
        if st.button(
            "Index documents",
            type="primary",
            use_container_width=True,
            disabled=not uploads,
            key="index_documents_button",
        ):
            try:
                with st.spinner("Extracting text, chunking, and building local embeddings…"):
                    store, count, indexed = build_vector_store(uploads, api_key="")
                st.session_state.vector_store = store
                st.session_state.chunk_count = count
                st.session_state.indexed_sources = indexed
                st.session_state.document_details = {
                    file.name: {
                        "size": file.size,
                        "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
                    }
                    for file in uploads
                }
                st.session_state.last_result = None
                st.success(f"Indexed {len(indexed)} document(s) into {count} evidence chunks.")
                st.rerun()
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")

        if sources:
            st.markdown(
                '<div class="section-title" style="margin-top:20px">Current evidence library</div>',
                unsafe_allow_html=True,
            )
            for source in sources:
                details = st.session_state.document_details.get(source, {})
                st.markdown(
                    f'<div class="file-row"><div class="file-icon">PDF</div><div style="flex:1;min-width:0"><div class="file-name">{html.escape(source)}</div><div class="file-meta">Indexed · {html.escape(details.get("indexed_at", "current session"))}</div></div><span class="tag">READY</span></div>',
                    unsafe_allow_html=True,
                )

    with right:
        st.markdown(
            '<div class="surface"><div class="section-title">Ask your research copilot</div><div class="section-copy">Ask for a summary, comparison, specific fact, or what the evidence does not establish.</div></div>',
            unsafe_allow_html=True,
        )
        question = st.text_area(
            "Research question",
            placeholder="e.g. Summarize key requirements, deadlines, and consequences. Cite the supporting evidence.",
            height=150,
            key="research_question",
            label_visibility="collapsed",
            disabled=not ready,
        )
        q1, q2 = st.columns(2)
        with q1:
            analyze = st.button(
                "Run evidence analysis →",
                type="primary",
                use_container_width=True,
                disabled=(not ready or not question.strip()),
                key="run_evidence_analysis_button",
            )
        with q2:
            if st.button("Clear question", use_container_width=True):
                st.session_state.research_question = ""
                st.rerun()
        if not ready:
            st.info("Index a PDF to activate the research copilot.")
        if analyze:
            try:
                with st.spinner("Retrieving evidence and generating a local answer…"):
                    result = run_research(
                        question.strip(), st.session_state.vector_store, api_key=""
                    )
                st.session_state.last_result = result
                st.session_state.last_question = question.strip()
                st.session_state.research_history.insert(
                    0,
                    {
                        "question": question.strip(),
                        "answer": result.get("answer", ""),
                        "time": datetime.now().strftime("%H:%M · %d %b %Y"),
                        "citation_check": result.get("citation_check", ""),
                        "trace": result.get("trace", []),
                        "evidence": result.get("evidence", []),
                    },
                )
                st.session_state.research_history = st.session_state.research_history[:20]
                st.rerun()
            except Exception as exc:
                st.error(f"Analysis failed: {exc}")

    result = st.session_state.last_result
    if result:
        st.markdown(
            '<div class="eyebrow" style="margin-top:30px">RESEARCH OUTPUT</div><h1 class="page-title" style="font-size:27px">Evidence report</h1>',
            unsafe_allow_html=True,
        )
        st.caption(f"Question: {st.session_state.last_question}")
        answer_col, verification_col = st.columns([1.5, 1], gap="large")
        with answer_col:
            st.markdown('<div class="section-title">Answer</div>', unsafe_allow_html=True)
            st.markdown('<div class="answerpanel">', unsafe_allow_html=True)
            st.markdown(result.get("answer", "No answer returned."))
            st.markdown("</div>", unsafe_allow_html=True)
            st.download_button(
                "Download answer (.md)",
                data="# Evidence report\n\n**Question:** "
                + st.session_state.last_question
                + "\n\n"
                + result.get("answer", "")
                + "\n",
                file_name="evidence_report.md",
                mime="text/markdown",
            )
        with verification_col:
            st.markdown('<div class="section-title">Verification & activity</div>', unsafe_allow_html=True)
            st.info(result.get("citation_check", "No citation check returned."))
            for index, item in enumerate(result.get("trace", []), 1):
                st.markdown(f"**{index:02d}** · {html.escape(str(item))}", unsafe_allow_html=True)

        st.markdown(
            '<div class="section-title" style="margin-top:18px">Retrieved source passages</div>',
            unsafe_allow_html=True,
        )
        evidence = result.get("evidence", [])
        if not evidence:
            st.caption("No evidence objects were returned for this saved result.")
        for index, doc in enumerate(evidence, 1):
            with st.expander(
                f"[S{index}] {doc.metadata.get('source', 'Unknown source')} · page {doc.metadata.get('page', '?')}",
                expanded=(index == 1),
            ):
                st.write(doc.page_content)
                st.caption("Inspect this passage before relying on the associated claim.")

elif page == "Documents":
    st.markdown(
        '<div class="eyebrow">KNOWLEDGE BASE</div><h1 class="page-title">Your documents.</h1><div class="copy">Review the PDFs indexed in the current session and their processing status.</div>',
        unsafe_allow_html=True,
    )
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div class="metric"><div class="metriclabel">Indexed files</div><div class="metricvalue">{len(sources)}</div><div class="metricnote">Current session</div></div>',
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f'<div class="metric"><div class="metriclabel">Searchable chunks</div><div class="metricvalue">{chunks}</div><div class="metricnote">Extracted passages</div></div>',
            unsafe_allow_html=True,
        )
    st.write("")
    search = st.text_input("Search documents", placeholder="Search by file name…", key="document_search")
    matching = [source for source in sources if search.lower() in source.lower()]
    if not matching:
        st.info("No matching documents." if sources else "No documents indexed yet. Go to Research to upload a PDF.")
    else:
        st.markdown('<div class="surface">', unsafe_allow_html=True)
        for source in matching:
            details = st.session_state.document_details.get(source, {})
            st.markdown(
                f'<div class="file-row"><div class="file-icon">PDF</div><div style="flex:1;min-width:0"><div class="file-name">{html.escape(source)}</div><div class="file-meta">Indexed {html.escape(details.get("indexed_at", "in this session"))} · Local embeddings</div></div><span class="tag">INDEXED</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
    st.caption("This library reflects the current session. Persistent storage and delete/re-index controls are future improvements.")

elif page == "Activity":
    st.markdown(
        '<div class="eyebrow">RESEARCH LOG</div><h1 class="page-title">Recent activity.</h1><div class="copy">Return to questions you have run in this session.</div>',
        unsafe_allow_html=True,
    )
    if not st.session_state.research_history:
        st.info("Your research history appears here after your first question.")
    else:
        for index, item in enumerate(st.session_state.research_history):
            with st.container(border=True):
                st.caption(item["time"])
                st.markdown(f"**{item['question']}**")
                st.markdown(item.get("answer", ""))
                st.caption(item.get("citation_check", ""))
                st.button(
                    "Open this result",
                    key=f"history_open_{index}",
                    on_click=open_history_result,
                    args=(item,),
                )

else:
    st.markdown(
        '<div class="eyebrow">PREFERENCES</div><h1 class="page-title">Workspace settings.</h1><div class="copy">Runtime information and current local processing configuration.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="surface"><div class="section-title">Local AI configuration</div><div class="section-copy">Current models and components configured in this project.</div></div>',
        unsafe_allow_html=True,
    )
    for label, value in [
        ("Answer generation", "Ollama · llama3.2:3b"),
        ("Embeddings", "sentence-transformers/all-MiniLM-L6-v2"),
        ("Vector store", "ChromaDB · in-memory index"),
        ("Workflow", "LangGraph"),
        ("PDF extraction", "PyMuPDF"),
    ]:
        st.markdown(
            f'<div class="file-row"><div style="flex:1" class="file-name">{label}</div><div class="file-meta">{value}</div></div>',
            unsafe_allow_html=True,
        )
    st.warning(
        "Citation checks validate source-label formatting and references; they do not prove that every cited passage supports its claim."
    )
    if st.button("Clear current session index"):
        st.session_state.vector_store = None
        st.session_state.indexed_sources = []
        st.session_state.chunk_count = 0
        st.session_state.document_details = {}
        st.session_state.last_result = None
        st.success("Current in-memory index cleared.")
        st.rerun()

st.markdown("---")
st.markdown(
    '<div class="smallmuted">Research Desk · Evidence-first local RAG prototype · Text-based PDFs supported; OCR for scanned documents is not included.</div>',
    unsafe_allow_html=True,
)
