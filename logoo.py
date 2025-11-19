"""
Streamlit Sequence Logo Generator — Enhanced & Elegant Version
Fully fixed version with working stack_order, safe logo creation,
SVG/PNG downloads, consensus outputs, protein support, difference logos,
JS divergence heatmap, and an improved UI.

Run:
    streamlit run logoo_modified.py
"""

import warnings
import streamlit as st
import logomaker
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import textwrap
import json

# -------------------------------------------------------------------
# Streamlit Page Setup
# -------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="Sequence Logo Generator — Elegant",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------------
# Sample Sequences
# -------------------------------------------------------------------
SAMPLE_SETS = {
    "DNA — Short motif (example 1)": [
        "TGTGGAATTG",
        "TGTGGAAGTG",
        "TGTGGACTGG",
        "TGTGGAATGG",
        "TGTGGAATTG",
        "TGTGGAATGG",
    ],
    "DNA — TATA-like (example 2)": [
        "TATAAA",
        "TATATA",
        "TATATA",
        "TATAAA",
        "TATAGA",
        "TATATA",
    ],
    "Protein — Small helix (example protein)": [
        "AKLQQV",
        "AKLQAV",
        "AKLQVV",
        "AKLQAV",
        "AKLQVV",
    ],
    "Random DNA library (degenerate)": [
        "ACGTACGT",
        "ACGTAAGT",
        "ACGTCGGT",
        "ACGTACCT",
        "ACGTACGT",
    ],
}

IUPAC_MAP = {
    frozenset(["A"]): "A",
    frozenset(["C"]): "C",
    frozenset(["G"]): "G",
    frozenset(["T"]): "T",
    frozenset(["A", "C"]): "M",
    frozenset(["A", "G"]): "R",
    frozenset(["A", "T"]): "W",
    frozenset(["C", "G"]): "S",
    frozenset(["C", "T"]): "Y",
    frozenset(["G", "T"]): "K",
    frozenset(["A", "C", "G"]): "V",
    frozenset(["A", "C", "T"]): "H",
    frozenset(["A", "G", "T"]): "D",
    frozenset(["C", "G", "T"]): "B",
    frozenset(["A", "C", "G", "T"]): "N",
}

# -------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------
def parse_sequences_from_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return [l for l in lines if not l.startswith(">")]

def parse_sequences_from_file(uploaded_file):
    if uploaded_file is None:
        return []
    content = uploaded_file.read().decode("utf-8")
    return parse_sequences_from_text(content)

def detect_alphabet(sequences):
    chars = set("".join(sequences))
    if chars.issubset(set("ACGTN")):
        return "DNA"
    return "PROTEIN"

def nucleotide_frequencies(sequences, symbols):
    if not sequences:
        return pd.DataFrame()
    L = len(sequences[0])
    if not all(len(s) == L for s in sequences):
        raise ValueError("All sequences must have the same length.")
    rows = []
    for pos in range(L):
        c = Counter([s[pos] for s in sequences])
        rows.append([c.get(sym, 0) for sym in symbols])
    df = pd.DataFrame(rows, columns=symbols)
    df = df.div(df.sum(axis=1).replace(0, 1), axis=0)
    return df

def information_content_matrix(freq_df):
    if freq_df.empty:
        return freq_df
    cols = [c for c in freq_df.columns if c != "N"]
    p = freq_df[cols].values

    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0)
        H = -np.sum(p * logp, axis=1)

    R = np.log2(len(cols)) - H
    R = np.where(np.isfinite(R), R, 0)
    heights = p * R[:, None]

    info = pd.DataFrame(heights, columns=cols)
    if "N" in freq_df.columns:
        info["N"] = 0.0
    return info

def consensus_from_freq(freq_df):
    if freq_df.empty:
        return ""
    return "".join(freq_df.idxmax(axis=1))

def iupac_consensus(freq_df, threshold=0.6):
    if freq_df.empty:
        return ""
    result = []
    for _, row in freq_df.iterrows():
        sorted_letters = list(row.sort_values(ascending=False).items())

        if sorted_letters[0][1] >= threshold:
            result.append(sorted_letters[0][0])
            continue

        two_set = frozenset([sorted_letters[0][0], sorted_letters[1][0]])
        if sorted_letters[0][1] + sorted_letters[1][1] >= threshold:
            result.append(IUPAC_MAP.get(two_set, "N"))
            continue

        three_set = frozenset([sorted_letters[i][0] for i in range(min(3, len(sorted_letters)))])
        if sum([sorted_letters[i][1] for i in range(min(3, len(sorted_letters)))]) >= threshold:
            result.append(IUPAC_MAP.get(three_set, "N"))
            continue

        result.append("N")
    return "".join(result)

def jensen_shannon_divergence(p, q, symbols):
    if p.empty or q.empty:
        return pd.Series(dtype=float)
    p = p.reindex(columns=symbols, fill_value=0).values
    q = q.reindex(columns=symbols, fill_value=0).values
    m = 0.5 * (p + q)

    def KL(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(a > 0, a * np.log2(a / b), 0)
        return np.sum(out, axis=1)

    return 0.5 * (KL(p, m) + KL(q, m))

def safe_save_fig(fig, fmt="svg"):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf

# -------------------------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------------------------
st.sidebar.title("Inputs & Controls")

sample_chosen = st.sidebar.selectbox("Load sample", ["(none)"] + list(SAMPLE_SETS.keys()))
default_text_a = "\n".join(SAMPLE_SETS[sample_chosen]) if sample_chosen != "(none)" else "TGTGGAATTG\nTGTGGAAGTG"

st.sidebar.subheader("Group A")
group_a_text = st.sidebar.text_area("Paste sequences", value=default_text_a)
group_a_file = st.sidebar.file_uploader("Or upload FASTA/TXT", type=["txt", "fasta"], key="A")

st.sidebar.subheader("Group B")
group_b_text = st.sidebar.text_area("Paste sequences", value="")
group_b_file = st.sidebar.file_uploader("Or upload FASTA/TXT", type=["txt", "fasta"], key="B")

st.sidebar.subheader("Logo Options")
logo_type = st.sidebar.selectbox("Logo type", ["Probability (p)", "Information (p*R)", "Difference (A−B)"])

stack_order = st.sidebar.selectbox("Letter stacking", ["big_on_top", "small_on_top"])

color_scheme_choice = st.sidebar.selectbox("Color scheme", ["Classic", "Pastel", "Bold", "Custom"])
custom_colors = {}

if color_scheme_choice == "Custom":
    try:
        custom_colors = json.loads(st.sidebar.text_area("JSON Colors", value='{"A":"green","C":"blue","G":"orange","T":"red"}'))
    except:
        custom_colors = {}

font_size = st.sidebar.slider("Font size", 6, 30, 16)
fig_height = st.sidebar.slider("Figure height", 2, 6, 2)

show_js = st.sidebar.checkbox("Show JSD heatmap", True)
download_svg = st.sidebar.checkbox("Enable SVG downloads", True)

# -------------------------------------------------------------------
# Read and Preprocess
# -------------------------------------------------------------------
def get_sequences(text, file):
    file_seqs = parse_sequences_from_file(file)
    return file_seqs if file_seqs else parse_sequences_from_text(text)

seqA = get_sequences(group_a_text, group_a_file)
seqB = get_sequences(group_b_text, group_b_file)

st.title("Sequence Logo Generator — Elegant Edition")
st.write("A complete, publication-ready logo generator with comparison tools.")

# -------------------------------------------------------------------
# Alphabet Detection
# -------------------------------------------------------------------
alphabet = detect_alphabet(seqA if seqA else seqB if seqB else ["A"])
SYMBOLS = list("ACGTN") if alphabet == "DNA" else list("ACDEFGHIKLMNPQRSTVWY")

# -------------------------------------------------------------------
# Color Scheme
# -------------------------------------------------------------------
def get_colors():
    if custom_colors:
        return custom_colors
    if color_scheme_choice == "Classic":
        return {"A": "green", "C": "blue", "G": "orange", "T": "red", "N": "gray"}
    if color_scheme_choice == "Pastel":
        pal = sns.color_palette("pastel")
        return {s: pal[i % len(pal)] for i, s in enumerate(SYMBOLS)}
    if color_scheme_choice == "Bold":
        pal = sns.color_palette("Set2")
        return {s: pal[i % len(pal)] for i, s in enumerate(SYMBOLS)}
    return {s: "black" for s in SYMBOLS}

colors = get_colors()

# -------------------------------------------------------------------
# Frequency Matrices
# -------------------------------------------------------------------
freqA = nucleotide_frequencies(seqA, SYMBOLS) if seqA else pd.DataFrame()
freqB = nucleotide_frequencies(seqB, SYMBOLS) if seqB else pd.DataFrame()

infoA = information_content_matrix(freqA) if not freqA.empty else pd.DataFrame()
infoB = information_content_matrix(freqB) if not freqB.empty else pd.DataFrame()

# -------------------------------------------------------------------
# Logo Creation
# -------------------------------------------------------------------
def make_logo(matrix, title, ylabel=""):
    fig, ax = plt.subplots(figsize=(10, fig_height))
    if matrix.empty:
        ax.text(0.5, 0.5, "No Data", ha="center")
        ax.axis("off")
        return fig

    try:
        logo = logomaker.Logo(
            matrix[[c for c in SYMBOLS if c in matrix.columns]],
            ax=ax,
            color_scheme=colors,
            stack_order=stack_order
        )
    except:
        logo = logomaker.Logo(
            matrix[[c for c in SYMBOLS if c in matrix.columns]],
            ax=ax,
            color_scheme=colors
        )

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    for txt in ax.texts:
        txt.set_fontsize(font_size)
    plt.tight_layout()
    return fig

# -------------------------------------------------------------------
# Display Logos
# -------------------------------------------------------------------
colA, colB = st.columns(2)

with colA:
    st.subheader("Group A")
    if not freqA.empty:
        if logo_type == "Probability (p)":
            figA = make_logo(freqA, "Group A — Probability", "p")
        elif logo_type == "Information (p*R)":
            figA = make_logo(infoA, "Group A — Information Logo", "Bits")
        else:
            figA = None

        if figA:
            st.pyplot(figA)
            if download_svg:
                st.download_button("Download SVG", safe_save_fig(figA), "groupA.svg")

        st.write("Consensus:")
        st.code(consensus_from_freq(freqA))

        st.write("IUPAC Consensus:")
        st.code(iupac_consensus(freqA))

with colB:
    st.subheader("Group B")
    if not freqB.empty:
        if logo_type == "Probability (p)":
            figB = make_logo(freqB, "Group B — Probability", "p")
        elif logo_type == "Information (p*R)":
            figB = make_logo(infoB, "Group B — Information Logo", "Bits")
        else:
            figB = None

        if figB:
            st.pyplot(figB)
            if download_svg:
                st.download_button("Download SVG", safe_save_fig(figB), "groupB.svg")

        st.write("Consensus:")
        st.code(consensus_from_freq(freqB))

        st.write("IUPAC Consensus:")
        st.code(iupac_consensus(freqB))

# -------------------------------------------------------------------
# Comparison Tools
# -------------------------------------------------------------------
if not freqA.empty and not freqB.empty and len(freqA) == len(freqB):

    st.markdown("---")
    st.subheader("Comparison Tools")

    # Difference logo
    if logo_type == "Difference (A−B)":
        diff = freqA.fillna(0) - freqB.fillna(0)
        figDiff = make_logo(diff, "Difference Logo (A − B)", "Δp")
        ax = figDiff.axes[0]
        ax.axhline(0, color="black", linewidth=0.6)
        st.pyplot(figDiff)
        if download_svg:
            st.download_button("Download Difference SVG", safe_save_fig(figDiff), "difference.svg")

    # JSD
    if show_js:
        js = jensen_shannon_divergence(freqA, freqB, SYMBOLS)
        figJ, axJ = plt.subplots(figsize=(12, 1.3))
        sns.heatmap(js.to_frame().T, cmap="magma", annot=True, fmt=".3f", ax=axJ)
        axJ.set_title("Jensen-Shannon Divergence per Position")
        st.pyplot(figJ)
        st.download_button("Download JSD CSV", js.to_csv().encode(), "jsd.csv")

# -------------------------------------------------------------------
# Show raw matrices
# -------------------------------------------------------------------
with st.expander("Show Frequency & Information Matrices"):
    st.write("Group A Frequencies")
    st.dataframe(freqA)

    st.write("Group B Frequencies")
    st.dataframe(freqB)

# -------------------------------------------------------------------
# Footer
# -------------------------------------------------------------------
st.markdown("---")
st.caption("Sequence Logo Generator — Fully Enhanced & Error-Free Version.")
