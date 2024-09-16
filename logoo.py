import streamlit as st
import logomaker
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

# Streamlit app title and description
st.title("Sequence Logo Generator")
st.write("""
Upload aligned DNA sequences to generate a sequence logo.
The logo will visualize the nucleotide frequency distribution at each position.
""")

# Input for sequences
st.sidebar.header("Input Sequences")
sequence_input = st.sidebar.text_area("Enter sequences (one per line)", value="ACGT\nACGA\nACGT\nACTT\nACGG\nACGA")

# Convert input into a list of sequences
sequences = sequence_input.splitlines()

# Calculate nucleotide frequencies per position
def nucleotide_frequencies(sequences):
    sequence_length = len(sequences[0])
    freqs = {pos: Counter([seq[pos] for seq in sequences]) for pos in range(sequence_length)}
    freq_matrix = pd.DataFrame(freqs).T
    return freq_matrix.div(freq_matrix.sum(axis=1), axis=0).fillna(0)

# Generate the frequency matrix
freq_matrix = nucleotide_frequencies(sequences)

# Create a sequence logo using logomaker
if len(sequences) > 0:
    st.subheader("Generated Sequence Logo")
    
    fig, ax = plt.subplots(figsize=(10, 5))
    logo = logomaker.Logo(freq_matrix, ax=ax)
    logo.style_spines(visible=False)
    logo.style_xticks(rotation=0, fmt='%d', anchor=0)
    logo.ax.set_ylabel("Probability")
    
    st.pyplot(fig)
else:
    st.write("Please enter valid sequences.")

# Add a download button for the logo
st.sidebar.subheader("Download Options")
if st.sidebar.button('Download Logo as PNG'):
    fig.savefig('sequence_logo.png')
    with open("sequence_logo.png", "rb") as file:
        btn = st.sidebar.download_button(label="Download Logo", data=file, file_name="sequence_logo.png", mime="image/png")
