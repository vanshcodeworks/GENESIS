# extract_examples.py

import pandas as pd
import pyranges as pr
import pyfaidx

def main():
    """
    This script finds and extracts one real 200bp DNA sequence example
    for each target gene from the original dataset.
    """
    print("--- Extracting Real DNA Sequence Examples ---")

    # --- 1. Load all necessary data files ---
    print("Loading genomic data files...")
    try:
        # Load the reference genome
        genome = pyfaidx.Fasta('data/Homo_sapiens.GRCh38.dna.chromosome.22.fa')
        
        # Load the variant data
        gvf_df = pd.read_csv('data/homo_sapiens-chr22.gvf', sep='\t', comment='#',
                             names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
                             dtype={'Chromosome': str})
        variants = pr.PyRanges(gvf_df)

        # Load the regulatory feature data
        gff_df = pd.read_csv('data/Homo_sapiens.GRCh38.regulatory_features.v114.gff3', sep='\t', comment='#',
                             names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
                             dtype={'Chromosome': str})
        reg_features = pr.PyRanges(gff_df)

    except FileNotFoundError as e:
        print(f"✗ Error: Could not find a required data file. {e}")
        return
    
    # --- 2. Find the intersection of variants and promoters ---
    print("Finding variants located in promoter regions...")
    promoters = reg_features[reg_features.Feature == 'promoter']
    variants_in_promoters = variants.join(promoters)
    results_df = variants_in_promoters.df # Convert to pandas for easy filtering

    # --- 3. Define gene locations and extract an example for each ---
    genes = {
        'CHEK2': {'start': 28686915, 'end': 28741334},
        'TBX1':  {'start': 19850059, 'end': 19879068},
        'COMT':  {'start': 19942588, 'end': 19969679}
    }
    
    flank_size = 100
    print("\n" + "="*50)
    print("      Here are your high-confidence example sequences")
    print("="*50 + "\n")

    for gene_name, info in genes.items():
        # Filter the DataFrame to find promoter variants within the current gene
        gene_variants_df = results_df[
            (results_df['Start'] >= info['start']) &
            (results_df['End'] <= info['end'])
        ]

        if not gene_variants_df.empty:
            # Get the very first variant from the list as our example
            example_variant = gene_variants_df.iloc[0]
            chrom = example_variant['Chromosome']
            pos = example_variant['Start']

            # Extract the 200bp sequence around this variant
            start_pos = pos - flank_size
            end_pos = pos + flank_size
            sequence = genome[chrom][start_pos:end_pos].seq.upper()

            print(f"✅ {gene_name} Example (from a real promoter variant in your dataset):")
            print(f"   (Variant found at position: {chrom}:{pos})")
            print(f"   Copy the sequence below:\n")
            print(f"{sequence}\n")
            print("-" * 50 + "\n")
        else:
            print(f"⚠️ Could not find a valid example for {gene_name} in the promoter variant set.")

if __name__ == "__main__":
    main()