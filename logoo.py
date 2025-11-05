"""
Streamlit Sequence Logo Generator — Enhanced & Robust Version

This update fixes a runtime error thrown by logomaker when an invalid `stack_order`
value was supplied and hardens logo creation so the app no longer crashes —
instead it falls back to a safe default and displays a helpful message.

Run:
    streamlit run logoo.py
"""
import streamlit as st
import logomaker
import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
import textwrap
from matplotlib.colors import to_hex

st.set_page_config(
    layout="wide",
    page_title="Sequence Logo Generator — Elegant",
    initial_sidebar_state="expanded",
)

# -------------------------
# Utility data & examples
# -------------------------
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

# -------------------------
# Helper functions
# -------------------------
def parse_sequences_from_text(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # remove FASTA headers if present
    seqs = [l for l in lines if not l.startswith(">")]
    return [s.upper() for s in seqs]


def parse_sequences_from_file(uploaded_file):
    if uploaded_file is None:
        return []
    content = uploaded_file.read().decode("utf-8")
    return parse_sequences_from_text(content)


def detect_alphabet(sequences):
    # Decide DNA vs protein based on characters present
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


def information_content_matrix(freq_df, consider_missing_as_zero=True):
    """
    Compute per-letter heights = p * R_seq (bits).
    R_seq = log2(k) - H, where k is the number of informative symbols (exclude N for DNA).
    """
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
    """
    Determine IUPAC-like consensus: if top letter p >= threshold => that letter,
    else if combination of top two/three reaches threshold, map to corresponding IUPAC degenerate code.
    Falls back to 'N' if not clear.
    """
    if freq_df.empty:
        return ""
    cons = []
    for _, row in freq_df.iterrows():
        sorted_letters = list(row.sort_values(ascending=False).items())
        if sorted_letters[0][1] >= threshold:
            cons.append(sorted_letters[0][0])
            continue
        if len(sorted_letters) > 1:
            cum = sorted_letters[0][1] + sorted_letters[1][1]
            topset = frozenset([sorted_letters[0][0], sorted_letters[1][0]])
            if cum >= threshold:
                cons.append(IUPAC_MAP.get(topset, "N"))
                continue
        if len(sorted_letters) > 2:
            cum3 = sorted_letters[0][1] + sorted_letters[1][1] + sorted_letters[2][1]
            topset3 = frozenset([sorted_letters[i][0] for i in range(3)])
            if cum3 >= threshold:
                cons.append(IUPAC_MAP.get(topset3, "N"))
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


def format_seq_list_for_textarea(seqs):
    return "\n".join(seqs)


def safe_save_fig(fig, fmt="png"):
    buf = BytesIO()
    fig.savefig(buf, format=fmt, bbox_inches="tight")
    buf.seek(0)
    return buf


def sanitize_color_scheme(scheme):
    """
    Convert a color scheme (dict mapping letters->color or palette values) into a dict
    of hex color strings acceptable to logomaker/matplotlib.
    """
    if not scheme:
        return scheme
    out = {}
    for k, v in scheme.items():
        try:
            # v can be a seaborn palette color (tuple) or hex or color name
            out[k] = to_hex(v)
        except Exception:
            out[k] = str(v)
    return out

# -------------------------
# Sidebar: Inputs & options
# -------------------------
st.sidebar.title("Inputs & Appearance")

# Sample chooser
sample_choice = st.sidebar.selectbox("Load a sample dataset", options=["(none)"] + list(SAMPLE_SETS.keys()))
if sample_choice != "(none)":
    default_text_a = format_seq_list_for_textarea(SAMPLE_SETS[sample_choice])
else:
    default_text_a = "TGTGGAATTG\nTGTGGAAGTG\nTGTGGACTGG"

st.sidebar.markdown("Upload sequences or paste aligned sequences (one per line). You can load a sample to try quickly.")

st.sidebar.subheader("Group A")
group_a_text = st.sidebar.text_area("Group A sequences", value=default_text_a, height=180)
group_a_file = st.sidebar.file_uploader("Upload Group A (txt/fasta)", type=["txt", "fasta"], key="a_up")

st.sidebar.subheader("Group B (optional)")
group_b_text = st.sidebar.text_area("Group B sequences", value="", height=180)
group_b_file = st.sidebar.file_uploader("Upload Group B (txt/fasta)", type=["txt", "fasta"], key="b_up")

st.sidebar.divider()

# Visual options
st.sidebar.subheader("Visual & Logo Options")
logo_type = st.sidebar.selectbox("Logo type", ["Probability (p)", "Information (p * R_seq)", "Difference (A − B)"])
color_scheme_choice = st.sidebar.selectbox("Color scheme", ["Classic (A C G T)", "Pastel", "Bold", "Custom dict"])
custom_colors = {}
if color_scheme_choice == "Custom dict":
    st.sidebar.write("Enter color mapping as JSON (e.g. {\"A\":\"green\",\"C\":\"blue\"})")
    custom_text = st.sidebar.text_area("Custom colors (JSON)", value='{"A":"#2ca02c","C":"#1f77b4","G":"#ff7f0e","T":"#d62728"}')
    try:
        import json
        custom_colors = json.loads(custom_text)
    except Exception:
        custom_colors = {}

font_size = st.sidebar.slider("Font size for letters", 6, 30, 18)
fig_height = st.sidebar.slider("Logo height (inches)", 2, 6, 2)
show_axis = st.sidebar.checkbox("Show axes & spines", value=False)
# Present user-friendly stack order choices but map to safe defaults with fallback
stack_choice = st.sidebar.selectbox("Stack order", ["Small on top (conservative)", "Large on top (emphasize major)"])
# Map to tentative logomaker values (we'll attempt and fallback if invalid)
stack_order_map = {
    "Small on top (conservative)": "small_on_top",
    "Large on top (emphasize major)": "big_on_top",  # attempt this; if invalid, fallback
}
stack_order_user = stack_order_map.get(stack_choice, "small_on_top")

show_js = st.sidebar.checkbox("Show Jensen-Shannon divergence heatmap (when comparing)", value=True)
download_svg = st.sidebar.checkbox("Offer SVG download", value=True)

st.sidebar.divider()
st.sidebar.markdown("Help & Tips")
with st.sidebar.expander("Tips"):
    st.markdown(
        textwrap.dedent(
            """
            - Sequences must be aligned within the same group (all equal length).
            - Use Group B for comparisons (difference logos / divergence).
            - Information logos (p * R_seq) show bit heights — useful to visualize conservation.
            - IUPAC consensus uses a threshold (default 0.6) to combine bases.
            """
        )
    )

# -------------------------
# Read sequences
# -------------------------
def get_sequences(text_input, uploaded_file):
    file_seqs = parse_sequences_from_file(uploaded_file)
    if file_seqs:
        return file_seqs
    return parse_sequences_from_text(text_input)


sequences_a = get_sequences(group_a_text, group_a_file)
sequences_b = get_sequences(group_b_text, group_b_file)

# Show basic status in a header
st.title("Sequence Logo Generator — Elegant Edition (Robust)")
st.write("A polished UI to create publication-quality sequence logos with extra analysis options.")

col1, col2 = st.columns([3, 1])

with col2:
    st.metric("Group A sequences", len(sequences_a))
    st.metric("Group B sequences", len(sequences_b))

# Validate lengths
def aligned_ok(seqs):
    return len(seqs) == 0 or all(len(s) == len(seqs[0]) for s in seqs)

if (sequences_a and not aligned_ok(sequences_a)) or (sequences_b and not aligned_ok(sequences_b)):
    st.error("All sequences in a group must be the same length. Please check your inputs.")
    st.stop()

# Determine alphabet and symbols
alphabet_type = None
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

# Color schemes
def get_color_scheme(choice):
    if isinstance(custom_colors, dict) and custom_colors:
        return custom_colors
    if choice.startswith("Classic"):
        return {"A": "green", "C": "blue", "G": "orange", "T": "red", "N": "gray"}
    if choice == "Pastel":
        palette = sns.color_palette("pastel", n_colors=len(SYMBOLS))
        return {s: palette[i % len(palette)] for i, s in enumerate(SYMBOLS)}
    if choice == "Bold":
        palette = sns.color_palette("Set2", n_colors=len(SYMBOLS))
        return {s: palette[i % len(palette)] for i, s in enumerate(SYMBOLS)}
    return {s: "gray" for s in SYMBOLS}

raw_scheme = get_color_scheme(color_scheme_choice)
color_scheme = sanitize_color_scheme(raw_scheme)

# -------------------------
# Compute matrices
# -------------------------
freq_a = nucleotide_frequencies(sequences_a, symbols=SYMBOLS) if sequences_a else pd.DataFrame()
freq_b = nucleotide_frequencies(sequences_b, symbols=SYMBOLS) if sequences_b else pd.DataFrame()

both_present_same_length = (not freq_a.empty) and (not freq_b.empty) and (len(freq_a) == len(freq_b))

info_a = information_content_matrix(freq_a) if not freq_a.empty else pd.DataFrame()
info_b = information_content_matrix(freq_b) if not freq_b.empty else pd.DataFrame()

# -------------------------
# Plotting helpers (robust)
# -------------------------
def make_logo_plot(matrix, title="", color_scheme=None, figsize=(10, 2.5), y_label=""):
    """
    Create a logomaker Logo while guarding against invalid stack_order or unforeseen logomaker errors.
    If logomaker raises, fallback to a safe stack_order ('small_on_top') and show a non-fatal message.
    """
    fig, ax = plt.subplots(figsize=figsize)
    if matrix.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center")
        ax.axis("off")
        return fig
    cols_order = [c for c in SYMBOLS if c in matrix.columns]
    matrix = matrix[cols_order]

    # Try with user requested stack_order first, then fallback
    attempted_orders = [stack_order_user, "small_on_top"]
    last_exc = None
    for order in attempted_orders:
        try:
            logo = logomaker.Logo(matrix, ax=ax, color_scheme=color_scheme, stack_order=order)
            # success — break out
            break
        except Exception as e:
            # store exception and continue to fallback
            last_exc = e
            continue
    else:
        # If we reached here, all attempts failed — show message and render a plain bar plot fallback
        st.error(
            "Could not render sequence logo with logomaker. Falling back to a simple stacked bar representation. "
            "Original error: " + (str(last_exc) if last_exc else "unknown")
        )
        # create stacked bar fallback
        bottom = np.zeros(len(matrix))
        x = np.arange(len(matrix))
        for col in matrix.columns:
            ax.bar(x, matrix[col].values, bottom=bottom, color=color_scheme.get(col, "#808080"), label=col)
            bottom += matrix[col].values
        ax.set_xticks(np.arange(len(matrix)))
        ax.set_xticklabels([str(i) for i in matrix.index])
        ax.set_ylabel(y_label or "Probability")
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        return fig

    # Style and finalize the successful logo
    try:
        logo.style_spines(visible=show_axis)
        logo.style_xticks(rotation=0, anchor=0)
    except Exception:
        # On some versions of logomaker these helpers may behave differently; ignore nonfatal styling errors
        pass
    ax.set_ylabel(y_label)
    ax.set_title(title)
    for label in ax.get_yticklabels():
        label.set_fontsize(max(font_size - 4, 6))
    # try to increase letter font size where possible
    for txt in ax.texts:
        try:
            txt.set_fontsize(font_size)
        except Exception:
            pass
    plt.tight_layout()
    return fig

# -------------------------
# Display outputs
# -------------------------
left_col, right_col = st.columns(2)

with left_col:
    st.subheader("Group A Logo")
    if not freq_a.empty:
        if logo_type == "Probability (p)":
            fig_a = make_logo_plot(freq_a, title="Group A — Probability Logo", color_scheme=color_scheme, figsize=(10, fig_height), y_label="Probability")
            st.pyplot(fig_a)
        elif logo_type == "Information (p * R_seq)":
            fig_a = make_logo_plot(info_a, title="Group A — Information-content (p * R_seq)", color_scheme=color_scheme, figsize=(10, fig_height), y_label="Bits")
            st.pyplot(fig_a)
        else:
            st.info("Select 'Difference (A − B)' in the sidebar to see difference logo when both groups are present.")
        st.markdown("Consensus (most probable):")
        st.code(consensus_from_freq(freq_a))
        st.markdown("IUPAC-like consensus (threshold 0.6):")
        st.code(iupac_consensus(freq_a, threshold=0.6))
        st.download_button("Download Group A frequency CSV", data=freq_a.to_csv(index=True).encode("utf-8"), file_name="group_a_freq.csv", mime="text/csv")
        if download_svg and 'fig_a' in locals() and fig_a is not None:
            buf = safe_save_fig(fig_a, fmt="svg")
            st.download_button("Download Group A SVG", data=buf, file_name="group_a_logo.svg", mime="image/svg+xml")
    else:
        st.info("No sequences provided for Group A")

with right_col:
    st.subheader("Group B Logo")
    if not freq_b.empty:
        if logo_type == "Probability (p)":
            fig_b = make_logo_plot(freq_b, title="Group B — Probability Logo", color_scheme=color_scheme, figsize=(10, fig_height), y_label="Probability")
            st.pyplot(fig_b)
        elif logo_type == "Information (p * R_seq)":
            fig_b = make_logo_plot(info_b, title="Group B — Information-content (p * R_seq)", color_scheme=color_scheme, figsize=(10, fig_height), y_label="Bits")
            st.pyplot(fig_b)
        else:
            st.info("Select 'Difference (A − B)' in the sidebar to see difference logo when both groups are present.")
        st.markdown("Consensus (most probable):")
        st.code(consensus_from_freq(freq_b))
        st.markdown("IUPAC-like consensus (threshold 0.6):")
        st.code(iupac_consensus(freq_b, threshold=0.6))
        st.download_button("Download Group B frequency CSV", data=freq_b.to_csv(index=True).encode("utf-8"), file_name="group_b_freq.csv", mime="text/csv")
        if download_svg and 'fig_b' in locals() and fig_b is not None:
            buf = safe_save_fig(fig_b, fmt="svg")
            st.download_button("Download Group B SVG", data=buf, file_name="group_b_logo.svg", mime="image/svg+xml")
    else:
        st.info("No sequences provided for Group B")

# Comparison & fancy outputs
if both_present_same_length:
    st.markdown("---")
    st.subheader("Comparison & Fancy Outputs")

    diff = freq_a.reindex(columns=SYMBOLS, fill_value=0) - freq_b.reindex(columns=SYMBOLS, fill_value=0)
    if logo_type == "Difference (A − B)":
        fig_diff = make_logo_plot(diff, title="Difference Logo (Group A − Group B)", color_scheme=color_scheme, figsize=(12, fig_height), y_label="ΔProbability")
        ax = fig_diff.axes[0]
        try:
            ax.axhline(0, color="k", linewidth=0.6)
        except Exception:
            pass
        st.pyplot(fig_diff)
        if download_svg:
            buf = safe_save_fig(fig_diff, fmt="svg")
            st.download_button("Download Difference SVG", data=buf, file_name="difference_logo.svg", mime="image/svg+xml")
        st.download_button("Download Difference CSV", data=diff.to_csv(index=True).encode("utf-8"), file_name="difference_matrix.csv", mime="text/csv")

    if show_js:
        js = jensen_shannon_divergence(freq_a, freq_b, symbols=SYMBOLS)
        fig_js, ax_js = plt.subplots(figsize=(12, 1.2))
        sns.heatmap(js.to_frame().T, cmap="magma", cbar_kws={"label": "JSD (bits)"}, annot=True, fmt=".3f", ax=ax_js)
        ax_js.set_xlabel("Position")
        ax_js.set_ylabel("")
        ax_js.set_title("Jensen-Shannon divergence per position (A vs B)")
        st.pyplot(fig_js)
        st.download_button("Download JSD CSV", data=js.to_csv(index=True).encode("utf-8"), file_name="js_divergence.csv", mime="text/csv")

    fig_compare = plt.figure(constrained_layout=True, figsize=(12, 5))
    gs = fig_compare.add_gridspec(2, 1, height_ratios=[1, 1])
    ax1 = fig_compare.add_subplot(gs[0])
    try:
        logomaker.Logo(info_a, ax=ax1, color_scheme=color_scheme, stack_order=stack_order_user)
    except Exception:
        logomaker.Logo(info_a, ax=ax1, color_scheme=color_scheme, stack_order="small_on_top")
    ax1.set_title("Group A — Information-content")
    ax2 = fig_compare.add_subplot(gs[1], sharex=ax1)
    try:
        logomaker.Logo(info_b, ax=ax2, color_scheme=color_scheme, stack_order=stack_order_user)
    except Exception:
        logomaker.Logo(info_b, ax=ax2, color_scheme=color_scheme, stack_order="small_on_top")
    ax2.set_title("Group B — Information-content")
    st.pyplot(fig_compare)
    if download_svg:
        buf = safe_save_fig(fig_compare, fmt="svg")
        st.download_button("Download comparison SVG", data=buf, file_name="info_comparison.svg", mime="image/svg+xml")

elif (not freq_a.empty and freq_b.empty) or (freq_a.empty and not freq_b.empty):
    st.info("Only one group provided — comparison outputs are hidden. Provide a second group to enable difference logos and divergence heatmap.")

# Show frequency matrices
with st.expander("Show frequency matrices and counts (expand)"):
    st.write("Group A frequency matrix (per position):")
    if not freq_a.empty:
        st.dataframe(freq_a)
    else:
        st.write("No Group A data")
    st.write("Group B frequency matrix (per position):")
    if not freq_b.empty:
        st.dataframe(freq_b)
    else:
        st.write("No Group B data")

# Footer / notes
st.markdown("---")
st.caption(
    "Notes: Sequences must be aligned within a group. Information content uses p * R_seq (bits). "
    "IUPAC consensus combines letters when no single letter dominates. Use SVG downloads for publication-quality figures."
)
