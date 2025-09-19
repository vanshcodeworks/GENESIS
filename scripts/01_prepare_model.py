import pyranges as pr
import pyfaidx
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path

def one_hot_encode(sequence):
    """Converts a DNA sequence string to a one-hot encoded NumPy array."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    unknown = [0, 0, 0, 0]
    return np.array([mapping.get(base.upper(), unknown) for base in sequence])

def main():
    """
    Main function to run the DNA processing and saving pipeline.
    """
    print("--- Starting DNA Data Preparation Pipeline ---")

    # Resolve repo root to make paths stable regardless of CWD
    repo_root = Path(__file__).resolve().parents[1]
    data_dir = repo_root / 'data'
    print("Indexing, loading, and finding overlaps...")
    genome = pyfaidx.Fasta(str(data_dir / 'Homo_sapiens.GRCh38.dna.chromosome.22.fa'))

    gff_df = pd.read_csv(
        str(data_dir / 'Homo_sapiens.GRCh38.regulatory_features.v114.gff3'),
        sep='\t', comment='#',
        names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
        dtype={'Chromosome': str}
    )
    reg_features = pr.PyRanges(gff_df)
    promoters = reg_features[reg_features.Feature == 'promoter']

    gvf_df = pd.read_csv(
        str(data_dir / 'homo_sapiens-chr22.gvf'),
        sep='\t', comment='#',
        names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
        dtype={'Chromosome': str}
    )
    variants = pr.PyRanges(gvf_df)

    variants_in_promoters = variants.join(promoters)
    results_df = variants_in_promoters.df
    print(f"Found {len(results_df)} variants in promoters to process.")
    print("-" * 20)

    # --- Step 5: Process variants with a progress bar ---
    print("Processing all found variants and one-hot encoding...")
    
    flank_size = 100
    all_sequences = []
    variant_positions = []  # keep (chrom, start, end) in X order

    for index, variant in tqdm(results_df.iterrows(), total=results_df.shape[0], desc="Encoding Sequences"):
        chrom = str(variant['Chromosome'])
        pos = int(variant['Start'])
        start_pos = pos - flank_size
        end_pos = pos + flank_size
        
        sequence_str = genome[chrom][start_pos:end_pos].seq
        
        if len(sequence_str) == flank_size * 2:
            encoded_seq = one_hot_encode(sequence_str)
            all_sequences.append(encoded_seq)
            variant_positions.append((chrom, int(variant['Start']), int(variant['End'])))

    X = np.array(all_sequences)

    # --- Step 6: Create Dummy Labels ---
    num_samples = X.shape[0]
    y = np.random.randint(0, 2, size=num_samples)
    
    print(f"Processed {num_samples} sequences.")
    print(f"Shape of sequence data (X): {X.shape}")
    print(f"Shape of labels (y): {y.shape}")
    print("-" * 20)

    # --- Step 7: Save the Processed Data ---
    print("Saving processed data to .npy files...")
    np.save(repo_root / 'processed_sequences.npy', X)
    np.save(repo_root / 'processed_labels.npy', y)
    np.save(repo_root / 'variant_positions.npy', np.array(variant_positions, dtype=object))
    print("Data saved successfully! Saved variant_positions.npy for label alignment.")

if __name__ == "__main__":
    main()