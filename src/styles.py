APP_CSS = """
<style>
:root {
    --surface: #f6f8fb;
    --sidebar: #f8fafc;
    --panel: #ffffff;
    --ink: #0f172a;
    --muted: #64748b;
    --line: #dbe3ef;
    --accent: #0d9488;
    --accent-strong: #0f766e;
    --accent-soft: #ccfbf1;
    --blue: #2563eb;
    --blue-soft: #dbeafe;
}
.stApp {
    background:
        radial-gradient(circle at 20% 0%, rgba(13, 148, 136, 0.10), transparent 28rem),
        linear-gradient(180deg, #f6f8fb 0%, #eef4f8 100%);
}
.main .block-container {
    max-width: 1040px;
    padding-top: 2rem;
}
[data-testid="stSidebar"] {
    background: var(--sidebar);
    border-right: 1px solid var(--line);
}
[data-testid="stSidebar"],
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3,
[data-testid="stSidebar"] small {
    color: var(--ink);
}
[data-testid="stSidebar"] .stCaptionContainer,
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] small {
    color: var(--muted);
}
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: var(--ink);
}
[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
    color: var(--muted);
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section {
    background: #ffffff !important;
    border: 1px dashed #94a3b8 !important;
    border-radius: 12px !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section * {
    color: var(--ink) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section div,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section article,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section aside,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section button {
    background-color: #ffffff !important;
    color: var(--ink) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section svg,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section svg * {
    color: var(--blue) !important;
    fill: none !important;
    stroke: currentColor !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section p,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section span,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section small {
    color: var(--ink) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section small,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section span:last-child {
    color: var(--muted) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] button {
    background: #ffffff;
    border: 1px solid var(--line);
    color: var(--accent-strong);
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] button svg,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] button svg * {
    color: var(--accent-strong) !important;
    fill: none !important;
    stroke: currentColor !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div:not(:first-child),
[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div:not(:first-child) > div,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div:not(:first-child) > div > div {
    background: #ffffff !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div:not(:first-child) {
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04) !important;
    margin-top: 0.45rem !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div:not(:first-child) *,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [class*="file"] *,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [class*="File"] * {
    color: var(--ink) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div:not(:first-child) small,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] > div:not(:first-child) span:last-child {
    color: var(--muted) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] ul,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] li,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"],
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileData"],
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileName"],
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"] {
    background: #ffffff !important;
    color: var(--ink) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] {
    border: 1px solid var(--line) !important;
    border-radius: 10px !important;
    box-shadow: 0 6px 18px rgba(15, 23, 42, 0.04) !important;
    margin-top: 0.45rem !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] *,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] li *,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] svg {
    color: var(--blue) !important;
    fill: none !important;
    stroke: currentColor !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderFileSize"],
[data-testid="stSidebar"] div[data-testid="stFileUploader"] small {
    color: var(--muted) !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"],
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] button {
    background: #ffffff !important;
    border-color: var(--line) !important;
    color: #dc2626 !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] [data-testid="stFileUploaderDeleteBtn"] svg * {
    color: #dc2626 !important;
    fill: none !important;
    stroke: currentColor !important;
}
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg,
[data-testid="stSidebar"] [data-testid="stTooltipIcon"] svg *,
[data-testid="stSidebar"] [aria-label*="help" i] svg,
[data-testid="stSidebar"] [title*="help" i] svg {
    color: var(--blue) !important;
    fill: none !important;
    stroke: currentColor !important;
}
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section button:not([data-testid="stFileUploaderDeleteBtn"]) svg,
[data-testid="stSidebar"] div[data-testid="stFileUploader"] section button:not([data-testid="stFileUploaderDeleteBtn"]) svg * {
    color: var(--accent-strong) !important;
    fill: none !important;
    stroke: currentColor !important;
}
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background: #ffffff;
    border-color: var(--line);
    color: var(--ink);
}
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: var(--ink);
}
[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    color: var(--muted);
    fill: var(--muted);
}
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: #ffffff;
    border: 1px solid var(--line);
    color: var(--ink);
}
[data-testid="stAlert"] {
    color: var(--ink) !important;
}
[data-testid="stAlert"] *,
[data-testid="stAlert"] [data-testid="stMarkdownContainer"],
[data-testid="stAlert"] [data-testid="stMarkdownContainer"] * {
    color: var(--ink) !important;
}
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="menu"],
div[data-baseweb="menu"] ul,
div[data-baseweb="menu"] li {
    background: #ffffff;
    color: var(--ink);
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="menu"] li:hover {
    background: #f0fdfa;
    color: var(--accent-strong);
}
div[data-baseweb="popover"] svg,
div[data-baseweb="menu"] svg {
    color: var(--muted);
    fill: var(--muted);
}
.hero-panel {
    background: rgba(255, 255, 255, 0.86);
    border: 1px solid var(--line);
    border-radius: 16px;
    box-shadow: 0 18px 55px rgba(15, 23, 42, 0.08);
    margin-bottom: 1.25rem;
    padding: 1.35rem 1.45rem 1.25rem;
    position: relative;
}
.hero-panel:before {
    background: linear-gradient(90deg, var(--accent-strong), var(--blue));
    border-radius: 16px 16px 0 0;
    content: "";
    height: 4px;
    left: -1px;
    position: absolute;
    right: -1px;
    top: -1px;
}
.app-kicker {
    color: var(--accent-strong);
    font-size: 0.84rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    margin-bottom: 0.2rem;
    text-transform: uppercase;
}
.app-title {
    color: var(--ink);
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: 0;
    line-height: 1.08;
    margin-bottom: 0.35rem;
}
.app-subtitle {
    color: #475569;
    font-size: 1rem;
    line-height: 1.6;
    margin-bottom: 0;
}
.doc-row {
    background: #ffffff;
    border: 1px solid var(--line);
    border-radius: 10px;
    box-shadow: 0 8px 24px rgba(15, 23, 42, 0.04);
    margin-bottom: 0.45rem;
    padding: 0.65rem 0.7rem;
}
.doc-name {
    color: var(--ink);
    font-size: 0.88rem;
    font-weight: 700;
    line-height: 1.25;
    overflow-wrap: anywhere;
}
.doc-meta {
    color: var(--muted);
    font-size: 0.76rem;
    margin-top: 0.22rem;
}
.doc-badge {
    background: var(--accent-soft);
    border-radius: 999px;
    color: #115e59;
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 800;
    margin-top: 0.42rem;
    padding: 0.12rem 0.45rem;
    text-transform: uppercase;
}
.status-pill {
    border-radius: 999px;
    display: inline-block;
    font-size: 0.78rem;
    font-weight: 700;
    margin-bottom: 0.6rem;
    padding: 0.28rem 0.65rem;
}
.status-ready {
    background: #dcfce7;
    color: #166534;
}
.status-missing {
    background: #fee2e2;
    color: #991b1b;
}
.status-stale {
    background: #fef3c7;
    color: #92400e;
}
div[data-testid="stChatMessage"] {
    background: rgba(255, 255, 255, 0.74);
    border: 1px solid rgba(219, 227, 239, 0.85);
    border-radius: 10px;
    color: var(--ink);
    padding: 0.55rem;
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] p,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] span,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] li,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] strong,
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] em {
    color: var(--ink);
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] a {
    color: var(--blue);
}
div[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] code {
    background: #e2e8f0;
    border-radius: 6px;
    color: #1e293b;
    padding: 0.1rem 0.28rem;
}
div[data-testid="stChatMessage"] [data-testid="stAlert"] {
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #78350f;
}
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"],
div[class*="stChatFloatingInputContainer"] {
    background: #f6f8fb !important;
}
[data-testid="stChatInput"] {
    background: #f6f8fb !important;
    padding-bottom: 0.75rem !important;
}
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form,
[data-testid="stChatInput"] label {
    color: var(--ink) !important;
}
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] form {
    background: #ffffff !important;
    border: 1px solid var(--line) !important;
    border-radius: 14px !important;
    box-shadow: 0 12px 36px rgba(15, 23, 42, 0.08) !important;
    min-height: 3.25rem !important;
}
[data-testid="stChatInput"] div[data-baseweb="textarea"],
[data-testid="stChatInput"] div[data-baseweb="textarea"] > div,
[data-testid="stChatInput"] div[data-baseweb="textarea"] > div > div,
[data-testid="stChatInput"] [data-baseweb="base-input"],
[data-testid="stChatInput"] [data-baseweb="base-input"] > div,
[data-testid="stChatInput"] textarea {
    background: #ffffff !important;
    border: 0 !important;
    box-shadow: none !important;
    color: var(--ink) !important;
    line-height: 1.45 !important;
}
[data-testid="stChatInput"] textarea {
    caret-color: var(--accent-strong) !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    color: var(--muted) !important;
    opacity: 1 !important;
}
[data-testid="stChatInput"] button {
    background: transparent !important;
    border: 0 !important;
    border-radius: 999px !important;
    box-shadow: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    min-height: 2.25rem !important;
    min-width: 2.25rem !important;
    position: relative !important;
}
[data-testid="stChatInput"] button * {
    display: none !important;
}
[data-testid="stChatInput"] button::before {
    background: var(--accent-strong);
    content: "";
    height: 0.85rem;
    left: 50%;
    position: absolute;
    top: 50%;
    transform: translate(-50%, -30%);
    width: 0.13rem;
}
[data-testid="stChatInput"] button::after {
    border-left: 0.13rem solid var(--accent-strong);
    border-top: 0.13rem solid var(--accent-strong);
    content: "";
    height: 0.55rem;
    left: 50%;
    position: absolute;
    top: 50%;
    transform: translate(-50%, -72%) rotate(45deg);
    width: 0.55rem;
}
[data-testid="stChatInput"] button:hover {
    background: #f0fdfa !important;
}
div.stButton > button {
    background: #ffffff !important;
    border-radius: 10px !important;
    border-color: var(--line) !important;
    color: var(--ink) !important;
    font-weight: 700 !important;
}
div.stButton > button:hover {
    background: #f0fdfa !important;
    border-color: var(--accent-strong) !important;
    color: var(--accent-strong) !important;
}
div.stButton > button[kind="primary"] {
    background: var(--accent-strong) !important;
    border-color: var(--accent-strong) !important;
    color: #ffffff !important;
}
div.stButton > button[kind="primary"]:hover {
    background: #115e59 !important;
    border-color: #115e59 !important;
    color: #ffffff !important;
}
[data-testid="stSidebar"] div.stButton > button[kind="primary"],
[data-testid="stSidebar"] div.stButton > button[kind="primary"] * {
    color: #ffffff !important;
}
[data-testid="stSidebar"] div.stButton > button,
[data-testid="stSidebar"] div.stButton > button * {
    color: var(--ink) !important;
}
[data-testid="stSidebar"] div.stButton > button[kind="primary"],
[data-testid="stSidebar"] div.stButton > button[kind="primary"] * {
    color: #ffffff !important;
}
</style>
"""
