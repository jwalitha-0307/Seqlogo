import streamlit as st
import logomaker
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from io import BytesIO

st.set_page_config(layout="wide", page_title="Sequence Logo Generator (2-sample)")

# Title and description
st.title("Sequence Logo Generator — Two-sample comparison")
st.write(
    "Upload or paste two sets of aligned DNA sequences (Group A and Group B). "
    "The app generates sequence logos for each group, an information-content logo, "
    "and a divergence heatmap to compare the two groups."
)

# Sidebar inputs
st.sidebar.header("Input / Options")

# Input method: paste or upload (for each group)
st.sidebar.subheader("Group A")
group_a_input = st.sidebar.text_area(
    "Enter Group A sequences (one per line)",
    value="TGTGGAATTG\nTGTGGAAGTG\nTGTGGACTGG",
    height=140,
)

group_a_file = st.sidebar.file_uploader("Or upload Group A (FASTA/plain sequences)", type=["txt", "fasta"], key="a_file")

st.sidebar.subheader("Group B")
group_b_input = st.sidebar.text_area(
    "Enter Group B sequences (one per line)",
    value="TGTGGAATTG\nTGTGGAATGG\nTGTGGACTGG",
    height=140,
)

group_b_file = st.sidebar.file_uploader("Or upload Group B (FASTA/plain sequences)", type=["txt", "fasta"], key="b_file")

# Appearance options
st.sidebar.subheader("Appearance")
color_scheme_choice = st.sidebar.selectbox(
    "Color scheme",
    options=["classic", "swatch1", "swatch2"],
    index=0,
    help="Classic: A=green, C=blue, G=orange, T=red. Swatches are alternate palettes."
)
show_information = st.sidebar.checkbox("Show information-content logos (R_seq)", value=True)
show_js_heatmap = st.sidebar.checkbox("Show Jensen-Shannon divergence heatmap", value=True)
download_png = st.sidebar.button("Generate & Download combined PNG")

# Helper: read uploaded file into sequence list
def parse_sequences_from_file(uploaded_file):
    if uploaded_file is None:
        return []
    content = uploaded_file.read().decode("utf-8")
    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.startswith(">")]
    return [line.upper() for line in lines]

# Get sequences (either from textarea or uploaded file)
def get_sequences(text_input, uploaded_file):
    file_seqs = parse_sequences_from_file(uploaded_file)
    if file_seqs:
        return file_seqs
    return [seq.strip().upper() for seq in text_input.splitlines() if seq.strip()]

sequences_a = get_sequences(group_a_input, group_a_file)
sequences_b = get_sequences(group_b_input, group_b_file)

# Allowed symbols (ordered)
SYMBOLS = ["A", "C", "G", "T", "N"]

# Frequency calculation
def nucleotide_frequencies(sequences, symbols=SYMBOLS):
    if not sequences:
        return pd.DataFrame()
    seq_len = len(sequences[0])
    # Validate length
    if not all(len(s) == seq_len for s in sequences):
        raise ValueError("All sequences must be the same length.")
    counts = []
    for pos in range(seq_len):
        col = [seq[pos] for seq in sequences]
        c = Counter(col)
        counts.append([c.get(sym, 0) for sym in symbols])
    df_counts = pd.DataFrame(counts, columns=symbols)
    df_probs = df_counts.div(df_counts.sum(axis=1).replace(0, 1), axis=0)
    return df_probs.fillna(0)

# Consensus
def consensus_from_freq(freq_df):
    if freq_df.empty:
        return ""
    return "".join(freq_df.idxmax(axis=1).tolist())

# Information content matrix (per-letter heights p * R_seq)
def information_content_matrix(freq_df, alphabet_size=4):
    # Determine actual alphabet_size (exclude N if not used)
    if freq_df.empty:
        return freq_df
    # We treat symbols except 'N' as informative by default
    considered = [c for c in freq_df.columns if c != "N"]
    p = freq_df[considered].values
    # compute entropy per row
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(p > 0, np.log2(p), 0.0)
        H = -np.sum(p * logp, axis=1)
    R_seq = np.log2(len(considered)) - H  # max bits = log2(alphabet)
    R_seq = np.where(np.isfinite(R_seq), R_seq, 0.0)
    # height per letter = p * R_seq
    heights = p * R_seq[:, None]
    info_df = pd.DataFrame(heights, columns=considered, index=freq_df.index)
    # if 'N' present, add zero column (or small contribution if you prefer)
    if "N" in freq_df.columns:
        info_df["N"] = 0.0
        # reorder columns to original order
        info_df = info_df[[c for c in freq_df.columns if c in info_df.columns]]
    return info_df

# Jensen-Shannon divergence per position (bits)
def jensen_shannon_divergence(p_df, q_df, symbols=SYMBOLS):
    if p_df.empty or q_df.empty:
        return pd.Series(dtype=float)
    # align indices / columns
    p = p_df.reindex(columns=symbols, fill_value=0).values
    q = q_df.reindex(columns=symbols, fill_value=0).values
    m = 0.5 * (p + q)
    def kl_div(a, b):
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(a > 0, a * np.log2(np.where(a > 0, a / b, 1.0)), 0.0)
            # when b==0 and a>0, ratio will be inf; replace with large number
            ratio = np.where(np.isfinite(ratio), ratio, 0.0)
            return np.sum(ratio, axis=1)
    kl1 = kl_div(p, m)
    kl2 = kl_div(q, m)
    js = 0.5 * (kl1 + kl2)
    return pd.Series(js, index=p_df.index)

# Coloring schemes
def get_color_scheme(choice):
    if choice == "classic":
        return {"A": "green", "C": "blue", "G": "orange", "T": "red", "N": "gray"}
    elif choice == "swatch1":
        return {"A": "#1b9e77", "C": "#d95f02", "G": "#7570b3", "T": "#e7298a", "N": "gray"}
    else:
        return {"A": "#4daf4a", "C": "#377eb8", "G": "#ff7f00", "T": "#f781bf", "N": "gray"}

# Validate sequences exist and aligned
aligned_ok_a = len(sequences_a) > 0 and all(len(s) == len(sequences_a[0]) for s in sequences_a)
aligned_ok_b = len(sequences_b) > 0 and all(len(s) == len(sequences_b[0]) for s in sequences_b)

if not sequences_a and not sequences_b:
    st.warning("Please provide sequences for at least one group (Group A and/or Group B).")
else:
    # Compute frequency matrices for provided groups
    freq_a = pd.DataFrame()
    freq_b = pd.DataFrame()
    try:
        if sequences_a:
            if not aligned_ok_a:
                raise ValueError("Not all sequences in Group A are the same length.")
            freq_a = nucleotide_frequencies(sequences_a)
        if sequences_b:
            if not aligned_ok_b:
                raise ValueError("Not all sequences in Group B are the same length.")
            freq_b = nucleotide_frequencies(sequences_b)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    # If both groups present, ensure same length for direct comparison
    same_length = (not freq_a.empty and not freq_b.empty and len(freq_a) == len(freq_b))
    if not same_length and (not freq_a.empty and not freq_b.empty):
        st.info("Both groups provided but lengths differ; logos will be shown separately. Divergence heatmap and difference logos require same length.")

    # Prepare plotting area: side-by-side logos
    cols = st.columns(2)
    color_scheme = get_color_scheme(color_scheme_choice)

    # Function to make logo figure for a frequency-like matrix
    def make_logo_figure(freq_df, title, color_scheme, show_info=False):
        if freq_df.empty:
            fig = plt.figure(figsize=(6, 3))
            plt.text(0.5, 0.5, "No data", ha="center", va="center")
            plt.axis("off")
            return fig
        # Ensure columns appear in symbol order
        cols_ordered = [c for c in SYMBOLS if c in freq_df.columns]
        freq_df = freq_df[cols_ordered]
        fig, ax = plt.subplots(figsize=(10, 2.5))
        logo = logomaker.Logo(freq_df, ax=ax, color_scheme=color_scheme)
        logo.style_spines(visible=False)
        logo.style_xticks(rotation=0, fmt='%d', anchor=0)
        ax.set_ylabel("Probability" if not show_info else "Bits (p * Rseq)")
        ax.set_xlabel("Position")
        ax.set_title(title)
        plt.tight_layout()
        return fig

    # Draw Group A logo
    with cols[0]:
        st.subheader("Group A")
        if not freq_a.empty:
            st.write(f"Sequences: {len(sequences_a)}  |  Length: {len(sequences_a[0])}")
            fig_a = make_logo_figure(freq_a, "Group A — Probability Logo", color_scheme, show_info=False)
            st.pyplot(fig_a)
            if show_information:
                info_a = information_content_matrix(freq_a)
                fig_a_info = make_logo_figure(info_a, "Group A — Information-content (p * R_seq)", color_scheme, show_info=True)
                st.pyplot(fig_a_info)
            st.markdown("Consensus (Group A):")
            st.code(consensus_from_freq(freq_a))
            st.download_button("Download Group A frequency CSV",
                               data=freq_a.to_csv(index=False).encode("utf-8"),
                               file_name="group_a_frequency_matrix.csv",
                               mime="text/csv")
        else:
            st.info("No sequences for Group A")

    # Draw Group B logo
    with cols[1]:
        st.subheader("Group B")
        if not freq_b.empty:
            st.write(f"Sequences: {len(sequences_b)}  |  Length: {len(sequences_b[0])}")
            fig_b = make_logo_figure(freq_b, "Group B — Probability Logo", color_scheme, show_info=False)
            st.pyplot(fig_b)
            if show_information:
                info_b = information_content_matrix(freq_b)
                fig_b_info = make_logo_figure(info_b, "Group B — Information-content (p * R_seq)", color_scheme, show_info=True)
                st.pyplot(fig_b_info)
            st.markdown("Consensus (Group B):")
            st.code(consensus_from_freq(freq_b))
            st.download_button("Download Group B frequency CSV",
                               data=freq_b.to_csv(index=False).encode("utf-8"),
                               file_name="group_b_frequency_matrix.csv",
                               mime="text/csv")
        else:
            st.info("No sequences for Group B")

    # If both present and same length, show comparison outputs
    if not freq_a.empty and not freq_b.empty and same_length:
        st.markdown("---")
        st.subheader("Comparison & Fancy Outputs")

        # Difference matrix (Group A - Group B)
        diff = freq_a.reindex(columns=SYMBOLS, fill_value=0) - freq_b.reindex(columns=SYMBOLS, fill_value=0)
        # Show difference logo (positive above axis, negative below) using logomaker's ability to plot values (can show negative)
        fig_diff, ax_diff = plt.subplots(figsize=(12, 3))
        # For difference, we want a color map: positive = greenish, negative = reddish; we'll map letters colors but allow negative
        # Logomaker can accept a matrix with positive/negative values
        diff_logo = logomaker.Logo(diff, ax=ax_diff, color_scheme=color_scheme, stack_order='small_on_top')
        diff_logo.ax.axhline(0, color='black', linewidth=0.5)
        ax_diff.set_title("Difference Logo (Group A − Group B)")
        ax_diff.set_ylabel("Difference in probability")
        diff_logo.style_spines(visible=False)
        st.pyplot(fig_diff)

        # Jensen-Shannon divergence heatmap per position
        if show_js_heatmap:
            js = jensen_shannon_divergence(freq_a, freq_b, symbols=SYMBOLS)
            fig_js, ax_js = plt.subplots(figsize=(12, 2))
            sns.heatmap(js.to_frame().T, cmap="magma", cbar_kws={'label': 'Jensen-Shannon divergence (bits)'}, ax=ax_js, annot=True, fmt=".3f")
            ax_js.set_xlabel("Position")
            ax_js.set_ylabel("")
            ax_js.set_title("Jensen-Shannon divergence per position (Group A vs Group B)")
            st.pyplot(fig_js)
            st.write("Higher JS divergence indicates greater distributional difference at that position.")

        # Side-by-side information-content comparison (R_seq for each group)
        info_a = information_content_matrix(freq_a)
        info_b = information_content_matrix(freq_b)
        fig_info_compare, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True)
        logomaker.Logo(info_a, ax=axes[0], color_scheme=color_scheme)
        axes[0].set_title("Group A — Information-content (p * R_seq)")
        axes[0].set_ylabel("Bits")
        logomaker.Logo(info_b, ax=axes[1], color_scheme=color_scheme)
        axes[1].set_title("Group B — Information-content (p * R_seq)")
        axes[1].set_ylabel("Bits")
        plt.tight_layout()
        st.pyplot(fig_info_compare)

        # Combined PNG download
        if download_png:
            # Compose a larger figure containing A/B/diff/js for download
            big_fig = plt.figure(figsize=(14, 10))
            gs = big_fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.6], hspace=0.6)
            ax1 = big_fig.add_subplot(gs[0, 0])
            logomaker.Logo(freq_a, ax=ax1, color_scheme=color_scheme)
            ax1.set_title("Group A — Probability Logo")
            ax2 = big_fig.add_subplot(gs[1, 0])
            logomaker.Logo(freq_b, ax=ax2, color_scheme=color_scheme)
            ax2.set_title("Group B — Probability Logo")
            ax3 = big_fig.add_subplot(gs[2, 0])
            logomaker.Logo(diff, ax=ax3, color_scheme=color_scheme)
            ax3.axhline(0, color='black', linewidth=0.5)
            ax3.set_title("Difference (A − B)")
            ax4 = big_fig.add_subplot(gs[3, 0])
            js = jensen_shannon_divergence(freq_a, freq_b, symbols=SYMBOLS)
            sns.heatmap(js.to_frame().T, cmap="magma", cbar_kws={'label': 'Jensen-Shannon divergence (bits)'}, ax=ax4, annot=True, fmt=".3f")
            ax4.set_xlabel("Position")
            ax4.set_ylabel("")
            ax4.set_title("Jensen-Shannon divergence per position")
            buf = BytesIO()
            big_fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            st.download_button("Download combined PNG", data=buf, file_name="sequence_logos_comparison.png", mime="image/png")
            plt.close(big_fig)

        # Allow downloading diff as CSV
        st.download_button("Download Difference Matrix (A - B) as CSV",
                           data=diff.to_csv(index=False).encode("utf-8"),
                           file_name="difference_matrix_A_minus_B.csv",
                           mime="text/csv")

    # If only one group present, give single-group download & info
    elif (not freq_b.empty and freq_a.empty) or (not freq_a.empty and freq_b.empty):
        st.info("Only one group provided — showing single-group outputs and downloads.")

st.markdown("---")
st.caption("Notes: • Sequences should be aligned (same length within a group). • 'Information-content' logos show letter heights multiplied by R_seq (bits). • Jensen-Shannon divergence is symmetric and measured in bits.")
