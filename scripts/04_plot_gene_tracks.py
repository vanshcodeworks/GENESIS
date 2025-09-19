import os
import pandas as pd
import pyranges as pr
import matplotlib.pyplot as plt

GENES = {
    'CHEK2': {'start': 28686915, 'end': 28741334},
    'TBX1':  {'start': 19850059, 'end': 19879068},
    'COMT':  {'start': 19942588, 'end': 19969679},
}

WINDOW_BP = 50_000  # +/- window around gene


def load_data():
    gff_df = pd.read_csv(
        os.path.join('data', 'Homo_sapiens.GRCh38.regulatory_features.v114.gff3'),
        sep='\t', comment='#',
        names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
        dtype={'Chromosome': str}
    )
    gvf_df = pd.read_csv(
        os.path.join('data', 'homo_sapiens-chr22.gvf'),
        sep='\t', comment='#',
        names=['Chromosome', 'Source', 'Feature', 'Start', 'End', 'Score', 'Strand', 'Frame', 'Attribute'],
        dtype={'Chromosome': str}
    )
    return gff_df, gvf_df


def plot_gene_region(ax, gene, start, end, promoters_df, variants_df):
    ax.axhline(0, color='black', linewidth=1)
    # Gene span
    ax.hlines(0.1, start, end, color='tab:blue', linewidth=8, label=f'{gene} span')

    # Promoters as boxes
    for _, row in promoters_df.iterrows():
        ax.add_patch(plt.Rectangle((row['Start'], -0.15), row['End']-row['Start'], 0.3,
                                   color='tab:orange', alpha=0.5))
    # Variants as ticks
    ax.vlines(variants_df['Start'], -0.25, -0.05, color='tab:red', alpha=0.6, linewidth=0.8)

    ax.set_xlim(start - WINDOW_BP, end + WINDOW_BP)
    ax.set_ylim(-0.4, 0.4)
    ax.set_yticks([])
    ax.set_xlabel('Genomic position (chr22)')
    ax.set_title(f'{gene} region with promoters and variants')
    ax.legend(loc='upper right')


def main():
    os.makedirs(os.path.join('reports', 'figures'), exist_ok=True)
    gff_df, gvf_df = load_data()

    promoters = pr.PyRanges(gff_df)
    promoters = promoters[promoters.Feature == 'promoter'].df

    for gene, coords in GENES.items():
        g_start = coords['start']
        g_end = coords['end']
        win_start = max(1, g_start - WINDOW_BP)
        win_end = g_end + WINDOW_BP

        # Filter promoters and variants in window
        prom_in_win = promoters[(promoters['Start'] <= win_end) & (promoters['End'] >= win_start)]
        vars_in_win = gvf_df[(gvf_df['Start'] >= win_start) & (gvf_df['Start'] <= win_end)]

        fig, ax = plt.subplots(figsize=(12, 3))
        plot_gene_region(ax, gene, g_start, g_end, prom_in_win, vars_in_win)
        out_path = os.path.join('reports', 'figures', f'{gene}_tracks.png')
        plt.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f'Saved: {out_path}')


if __name__ == '__main__':
    main()
    main()
