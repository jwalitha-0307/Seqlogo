# -----------------------------------------------------------
# Streamlit Sequence Logo Generator — Final Combined Version
# Fully merged code + stack_order fix + safe logo fallback
# Run using:   streamlit run logo_final.py
# -----------------------------------------------------------

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

st.set_page_config(
    layout="wide",
    page_title="Sequence Logo Generator — Elegant",
    initial_sidebar_state="expanded",
)

# ------------------------- #
# Sample datasets
# ------------------------- #

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

# IUPAC map
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


# ------------------------- #
# Helper functions
# ------------------------- #

def parse_sequences_from_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    seqs = [l for l in lines if not l.startswith(">")]
    return [s.upper() for s in seqs]


def parse_sequences_from_file(uploaded_file):
    if uploaded_file is None:
        return []
    content = uploaded_file.read().decode("utf-8")
    return parse_sequences_from_text(content)


def detect_alphabet(sequences):
    chars = set("".join(sequences))
    dna_chars = set(list("ACGTN"))
    if chars.issubset(dna_chars):
        return "DNA"
    return "PROTEIN"


def nucleotide_frequencies(sequences, symbols=None):
    if not sequences:
        return pd.DataFrame()
    seq_len = len(sequences[0])
    if not all(len(s) == seq_len for s in sequences):
        raise ValueError("All sequences must be same length.")
    if symbols is None:
        symbols = sorted(set("".join(sequences)))

    counts = []
    for pos in range(seq_len):
        c = Counter([seq[pos] for seq in sequences])
        counts.append([c.get(sym, 0) for sym in symbols])

    df_counts = pd.DataFrame(counts, columns=symbols)
    df_probs = df_counts.div(df_counts.sum(axis=1).replace(0, 1), axis=0)
    return df_probs.fillna(0)


def information_content_matrix(freq_df):
    if freq_df.empty:
        return freq_df

    cols = [c for c in freq_df.columns if c != "N"] if "N" in freq_df.columns else list(freq_df.columns)

    p = freq_df[cols].values
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
        H = -np.sum(p * logp, axis=1)

    R_seq = np.log2(len(cols)) - H
    R_seq = np.where(np.isfinite(R_seq), R_seq, 0.0)

    heights = p * R_seq[:, None]

    info_df = pd.DataFrame(heights, columns=cols, index=freq_df.index)
    if "N" in freq_df.columns:
        info_df["N"] = 0.0
        info_df = info_df[[c for c in freq_df.columns if c in info_df.columns]]
    return info_df


def consensus_from_freq(freq_df):
    if freq_df.empty:
        return ""
    return "".join(freq_df.idxmax(axis=1).tolist())


def iupac_consensus(freq_df, threshold=0.6):
    if freq_df.empty:
        return ""
    cons = []
    for _, row in freq_df.iterrows():
        sorted_letters = list(row.sort_values(ascending=False).items())

        if sorted_letters[0][1] >= threshold:
            cons.append(sorted_letters[0][0])
            continue

        cum = sorted_letters[0][1] + sorted_letters[1][1]
        two = frozenset([sorted_letters[0][0], sorted_letters[1][0]])
        if cum >= threshold:
            cons.append(IUPAC_MAP.get(two, "N"))
            continue

        if len(sorted_letters) > 2:
            cum3 = cum + sorted_letters[2][1]
            three = frozenset([sorted_letters[i][0] for i in range(3)])
            if cum3 >= threshold:
                cons.append(IUPAC_MAP.get(three, "N"))
                continue

        cons.append("N")

    return "".join(cons)


def jensen_shannon_divergence(p_df, q_df, symbols=None):
    if p_df.empty or q_df.empty:
        return pd.Series(dtype=float)

    if symbols is None:
        symbols = sorted(set(list(p_df.columns) + list(q_df.columns)))

    p = p_df.reindex(columns=symbols, fill_value=0).values
    q = q_df.reindex(columns=symbols, fill_value=0).values
    m = 0.5 * (p + q)

    def kl(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(a > 0, a * np.log2(np.where(b > 0, a / b, 1.0)), 0.0)
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
            return np.sum(ratio, axis=1)

    js = 0.5 * (kl(p, m) + kl(q, m))
    return pd.Series(js, index=p_df.index)


def safe_save_fig(fig, fmt="png"):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf


# ------------------------- #
# Sidebar inputs
# ------------------------- #

st.sidebar.title("Inputs & Appearance")

sample_choice = st.sidebar.selectbox(
    "Load a sample dataset",
    options=["(none)"] + list(SAMPLE_SETS.keys())
)

if sample_choice != "(none)":
    default_text_a = "\n".join(SAMPLE_SETS[sample_choice])
else:
    default_text_a = "TGTGGAATTG\nTGTGGAAGTG\nTGTGGACTGG"

st.sidebar.subheader("Group A")
group_a_text = st.sidebar.text_area("Group A sequences", value=default_text_a, height=180)
group_a_file = st.sidebar.file_uploader("Upload Group A", type=["txt", "fasta"], key="a_up")

st.sidebar.subheader("Group B (optional)")
group_b_text = st.sidebar.text_area("Group B sequences", value="", height=180)
group_b_file = st.sidebar.file_uploader("Upload Group B", type=["txt", "fasta"], key="b_up")

st.sidebar.divider()

logo_type = st.sidebar.selectbox("Logo type", ["Probability (p)", "Information (p * R_seq)", "Difference (A − B)"])

color_scheme_choice = st.sidebar.selectbox(
    "Color scheme", ["Classic (A C G T)", "Pastel", "Bold", "Custom dict"]
)

custom_colors = {}
if color_scheme_choice == "Custom dict":
    j = st.sidebar.text_area("Custom colors JSON", value='{"A":"#2ca02c","C":"#1f77b4","G":"#ff7f0e","T":"#d62728"}')
    try:
        custom_colors = json.loads(j)
    except:
        custom_colors = {}

font_size = st.sidebar.slider("Font size", 6, 30, 18)
fig_height = st.sidebar.slider("Figure height", 2, 6, 2)

show_axis = st.sidebar.checkbox("Show axes", value=False)

# FIX: Only valid names for Logomaker
stack_order = st.sidebar.selectbox("Stack order", ["big_on_top", "small_on_top"])

show_js = st.sidebar.checkbox("Jensen-Shannon heatmap", value=True)
download_svg = st.sidebar.checkbox("Enable SVG download", value=True)

# ------------------------- #
# Load sequences
# ------------------------- #

def get_sequences(text, fileobj):
    file_seqs = parse_sequences_from_file(fileobj)
    if file_seqs:
        return file_seqs
    return parse_sequences_from_text(text)

sequences_a = get_sequences(group_a_text, group_a_file)
sequences_b = get_sequences(group_b_text, group_b_file)

# ------------------------- #
# Page header
# ------------------------- #

st.title("Sequence Logo Generator — Elegant Edition")
st.write("Create publication-quality sequence logos with comparisons, info content, consensus, JSD, and SVG export.")

col1, col2 = st.columns([3, 1])
with col2:
    st.metric("Group A sequences", len(sequences_a))
    st.metric("Group B sequences", len(sequences_b))

# Validate lengths
def aligned_ok(seqs):
    return len(seqs) == 0 or all(len(s) == len(seqs[0]) for s in seqs)

if (sequences_a and not aligned_ok(sequences_a)) or (sequences_b and not aligned_ok(sequences_b)):
    st.error("All sequences in a group must have the same length.")
    st.stop()

# Detect alphabet
if sequences_a:
    alphabet_type = detect_alphabet(sequences_a)
elif sequences_b:
    alphabet_type = detect_alphabet(sequences_b)
else:
    alphabet_type = "DNA"

if alphabet_type == "DNA":
    SYMBOLS = ["A", "C", "G", "T", "N"]
else:
    SYMBOLS = list("ACDEFGHIKLMNPQRSTVWY")


# Color Schemes
def get_color_scheme(choice):
    if custom_colors:
        return custom_colors

    if choice.startswith("Classic"):
        return {"A": "green", "C": "blue", "G": "orange", "T": "red", "N": "gray"}

    if choice == "Pastel":
        return {s: sns.color_palette("pastel")[i % 8] for i, s in enumerate(SYMBOLS)}

    if choice == "Bold":
        return {s: sns.color_palette("Set2")[i % 8] for i, s in enumerate(SYMBOLS)}

    return {s: "gray" for s in SYMBOLS}


color_scheme = get_color_scheme(color_scheme_choice)

# Compute frequency matrices
freq_a = nucleotide_frequencies(sequences_a, symbols=SYMBOLS) if sequences_a else pd.DataFrame()
freq_b = nucleotide_frequencies(sequences_b, symbols=SYMBOLS) if sequences_b else pd.DataFrame()

both_present_same_length = (
    not freq_a.empty
    and not freq_b.empty
    and (len(freq_a) == len(freq_b))
)

# Info matrices
info_a = information_content_matrix(freq_a)
info_b = information_content_matrix(freq_b)

# ------------------------- #
# Logo Plotting (SAFE)
# ------------------------- #

def make_logo_plot(matrix, title="", color_scheme=None, figsize=(10, 2.5), y_label=""):
    fig, ax = plt.subplots(figsize=figsize)

    if matrix.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig

    cols_order = [c for c in SYMBOLS if c in matrix.columns]
    matrix = matrix[cols_order]

    # SAFE STACK ORDER FIX
    try:
        logomaker.Logo(matrix, ax=ax, color_scheme=color_scheme, stack_order=stack_order)
    except Exception as e:
        warnings.warn(f"Invalid stack_order={stack_order}, using default. Error: {e}")
        logomaker.Logo(matrix, ax=ax, color_scheme=color_scheme)

    ax.set_title(title)
    ax.set_ylabel(y_label)

    if not show_axis:
        ax.axis("off")

    plt.tight_layout()
    return fig


# ------------------------- #
# Display Group A and B
# ------------------------- #

left, right = st.columns(2)

with left:
    st.subheader("Group A Logo")
    if not freq_a.empty:
        if logo_type == "Probability (p)":
            fig_a = make_logo_plot(freq_a, title="Group A — Probability Logo", color_scheme=color_scheme, figsize=(10, fig_height))
        elif logo_type == "Information (p * R_seq)":
            fig_a = make_logo_plot(info_a, title="Group A — Information Logo", color_scheme=color_scheme, figsize=(10, fig_height))
        else:
            st.info("Select 'Difference (A − B)' to view the comparison logo.")
            fig_a = make_logo_plot(freq_a, title="Group A", color_scheme=color_scheme)

        st.pyplot(fig_a)

        st.markdown("**Consensus:**")
        st.code(consensus_from_freq(freq_a))

        st.markdown("**IUPAC Consensus:**")
        st.code(iupac_consensus(freq_a))

        st.download_button(
            "Download Group A freq CSV",
            freq_a.to_csv().encode(),
            "group_a_freq.csv"
        )

        if download_svg:
            svg_a = safe_save_fig(fig_a, fmt="svg")
            st.download_button("Download Group A SVG", svg_a, "group_a_logo.svg")

with right:
    st.subheader("Group B Logo")
    if not freq_b.empty:
        if logo_type == "Probability (p)":
            fig_b = make_logo_plot(freq_b, title="Group B — Probability Logo", color_scheme=color_scheme, figsize=(10, fig_height))
        elif logo_type == "Information (p * R_seq)":
            fig_b = make_logo_plot(info_b, title="Group B — Information Logo", color_scheme=color_scheme, figsize=(10, fig_height))
        else:
            fig_b = make_logo_plot(freq_b, title="Group B", color_scheme=color_scheme)

        st.pyplot(fig_b)

        st.markdown("**Consensus:**")
        st.code(consensus_from_freq(freq_b))

        st.markdown("**IUPAC Consensus:**")
        st.code(iupac_consensus(freq_b))

        st.download_button(
            "Download Group B freq CSV",
            freq_b.to_csv().encode(),
            "group_b_freq.csv"
        )

        if download_svg:
            svg_b = safe_save_fig(fig_b, fmt="svg")
            st.download_button("Download Group B SVG", svg_b, "group_b_logo.svg")


# ------------------------- #
# Comparison Mode
# ------------------------- #

if both_present_same_length:
    st.markdown("---")
    st.subheader("Comparison Mode")

    diff = freq_a - freq_b

    if logo_type == "Difference (A − B)":
        fig_diff = make_logo_plot(diff, title="Difference Logo (A − B)", color_scheme=color_scheme, figsize=(12, fig_height))
        ax = fig_diff.axes[0]
        ax.axhline(0, color="black", linewidth=0.6)
        st.pyplot(fig_diff)

        if download_svg:
            diff_svg = safe_save_fig(fig_diff, fmt="svg")
            st.download_button("Download Difference SVG", diff_svg, "difference.svg")

        st.download_button("Download Difference CSV", diff.to_csv().encode(), "difference.csv")

    # JSD heatmap
    if show_js:
        js = jensen_shannon_divergence(freq_a, freq_b, symbols=SYMBOLS)
        fig_js, ax = plt.subplots(figsize=(12, 2))
        sns.heatmap(js.to_frame().T, cmap="magma", cbar_kws={"label": "JSD"}, annot=True, fmt=".3f", ax=ax)
        st.pyplot(fig_js)

        st.download_button("Download JSD CSV", js.to_csv().encode(), "js_divergence.csv")


# ------------------------- #
# Show matrices
# ------------------------- #

with st.expander("Show Raw Frequency Matrices"):
    st.write("### Group A")
    st.dataframe(freq_a)

    st.write("### Group B")
    st.dataframe(freq_b)


# Footer
st.markdown("---")
st.caption("Sequence Logo Generator — Supports DNA/Protein, IUPAC consensus, info-content, comparisons, JSD & SVG export.")
