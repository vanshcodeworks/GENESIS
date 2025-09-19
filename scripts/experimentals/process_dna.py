
import pyranges as pr
import pyfaidx
import pandas as pd

def main():
    """
    Main function to run the DNA processing pipeline.
    """
    print("--- Starting DNA Processing Pipeline ---")

    # --- 1. Index the Reference Genome ---
    print("Indexing the reference genome...")
    try:
        genome = pyfaidx.Fasta('data/Homo_sapiens.GRCh38.dna.chromosome.22.fa')
        print("Genome indexed successfully.")
    except pyfaidx.FastaIndexingError as e:
        print(f"Error indexing FASTA file: {e}")
        return
    print("-" * 20)

    # --- 2. Load and Filter Regulatory Features ---
    print("Loading regulatory features...")
    try:
        gff_df = pd.read_csv(
            'data/Homo_sapiens.GRCh38.regulatory_features.v114.gff3',
            sep='\t', comment='#',
            names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute']
        )
        reg_features = pr.PyRanges(gff_df)
        promoters = reg_features[reg_features.Feature == 'promoter']
        print(f"Found {len(promoters)} promoters.")
    except FileNotFoundError:
        print("Error: Regulatory features file not found.")
        return
    print("-" * 20)
    
    # --- 3. Load Variants ---
    print("Loading variants for Chromosome 22...")
    try:
        gvf_df = pd.read_csv(
            'data/homo_sapiens-chr22.gvf',
            sep='\t', comment='#',
            names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute']
        )
        variants = pr.PyRanges(gvf_df)
        print(f"Found {len(variants)} variants on Chromosome 22.")
    except FileNotFoundError:
        print("Error: Variant file not found.")
        return
    print("-" * 20)

    # --- 4. Find Overlapping Variants ---
    print("Finding variants that overlap with promoters...")
    variants_in_promoters = variants.join(promoters)
    print(f"Found {len(variants_in_promoters)} variants located in promoters.")
    print("-" * 20)

    # --- 5. Extract DNA and Print Results ---
    flank_size = 100
    print(f"--- Extracting DNA for the first 5 promoter variants ---")
    results_df = variants_in_promoters.df
    
    if results_df.empty:
        print("No variants found in promoter regions to display.")
        return

    for index, variant in results_df.head(5).iterrows():
        chrom = variant['Chromosome']
        pos = variant['Start']
        start_pos = pos - flank_size
        end_pos = pos + flank_size
        
        sequence = genome[chrom][start_pos:end_pos].seq

        print(f"\nVariant #{index+1}:")
        print(f"  Location: Chromosome {chrom}, Position {pos}")
        print(f"  This variant is in a '{variant.Feature_b}' region.")
        print(f"  DNA Sequence ({flank_size*2}bp window): ...{sequence.upper()}...")

if __name__ == "__main__":
    main()