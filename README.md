
# SeqLogo — Sequence Logo Generator (Elegant & Robust)

SeqLogo is a small Streamlit application to generate publication-ready sequence logos from aligned DNA or protein sequence sets. It supports single-group logos as well as two-sample comparisons (Group A vs Group B), information-content logos (p * R_seq), difference logos (A − B), Jensen–Shannon divergence per position, and IUPAC-like consensus reporting. The app includes sample datasets, multiple color schemes, SVG/PNG downloads, and graceful fallbacks for compatibility with different logomaker versions.

This repository contains `logoo.py` — the Streamlit app — and this README describing usage, examples, and troubleshooting tips.

---

## Features

- Generate probability logos for DNA or protein alphabets.
- Compute and display information-content logos (p * R_seq in bits).
- Side-by-side comparison of two groups with:
  - Difference logo (A − B)
  - Jensen–Shannon divergence heatmap per position
  - CSV/SVG/PNG download support
- IUPAC-like consensus generation (combines bases when no single base dominates).
- Several color schemes and a custom color dict option.
- Sample datasets to get started quickly.
- Robust error handling and fallback plots if logomaker fails on certain options.

---

## Requirements

- Python 3.8+ (tested on 3.10/3.11+)
- Recommended packages:
  - streamlit
  - pandas
  - numpy
  - matplotlib
  - logomaker
  - seaborn

You can install the required packages with pip:

```bash
pip install streamlit pandas numpy matplotlib logomaker seaborn
```

If you plan to export SVGs ensure your matplotlib backend supports SVG output (the default backends do).

---

## Quickstart

1. Clone the repository (or drop `logoo.py` into a project folder):

```bash
git clone https://github.com/jwalitha-0307/Seqlogo.git
cd Seqlogo
```

2. Run the Streamlit app:

```bash
streamlit run logoo.py
```

3. Open the provided local URL (usually http://localhost:8501) in your browser.

4. In the sidebar:
   - Paste or upload aligned sequences for Group A (and optionally Group B).
   - Choose logo type (Probability, Information, or Difference).
   - Select color scheme, font size, and other visual options.
   - Generate logos and download PNG/SVG/CSV using the provided buttons.

---

## Example inputs

DNA short motif (paste into Group A):

```
TGTGGAATTG
TGTGGAAGTG
TGTGGACTGG
TGTGGAATGG
TGTGGAATTG
TGTGGAATGG
```

DNA TATA motif (example):

```
TATAAA
TATATA
TATATA
TATAAA
TATAGA
TATATA
```

Protein small helix (example):

```
AKLQQV
AKLQAV
AKLQVV
AKLQAV
AKLQVV
```

---

## Output & Downloads

- Logos are displayed inline in Streamlit and can be downloaded as PNG or (optionally) SVG.
- Frequency matrices, difference matrices, and Jensen–Shannon divergence values can be downloaded as CSV files.
- The app provides both probability logos and information-content logos (p * R_seq) for easier interpretation of conservation.

---

## Troubleshooting

- All sequences in a group must be the same length. The app validates alignment per group and will show an error if lengths differ.
- logomaker compatibility:
  - Some versions of logomaker accept different values for the `stack_order` argument. The app attempts the user-chosen order first and falls back to a safe default (`small_on_top`) if logomaker rejects it.
  - If a severe logomaker error occurs, the app will not crash; instead it will render a stacked-bar fallback and show a helpful message in the UI.
- If you see a `LogomakerError` complaining about `stack_order`, update logomaker or switch the stack order in the sidebar; the app already includes safe fallbacks.

---

## Development & Contribution

- The main app file is `logoo.py`. If you want to add features (interactive hover tooltips, different divergence metrics, automated tests, etc.), fork the repository and open a PR.
- Please include unit tests for numeric functions (frequency calculation, information content, JS divergence).
- Suggested improvements:
  - Add interactive SVG/HTML tooltips (e.g., via Plotly or custom SVG generation).
  - Add sequence alignment helper for unaligned input.
  - Add support for ambiguous IUPAC symbols as input with proper handling.

---

## License

Specify your preferred license here (e.g., MIT). Example:

```
MIT License
Copyright (c) 2025 jwalitha-0307
```

---

## Contact

If you encounter bugs or want features, open an issue in the repository or contact the maintainer: jwalitha-0307.
