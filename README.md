# GENESIS: A Deep Learning Framework for Multi-Class Classification of Non-Coding Genomic Variants

Author: Vansh Goyal • Contact: vanshgoyal9528@gmail.com • Date: September 18, 2025

## Abstract
GENESIS is an end-to-end deep learning pipeline to classify non-coding variants from local DNA sequence. A PyTorch 1D CNN trained on GRCh38 chr22 achieves near-perfect performance (accuracy ~99.99%, macro F1 = 1.00) across CHEK2, TBX1, COMT vs. background promoter variants via class-weighted training and GPU acceleration.

## Background and Learning
This project is intended for research and education. It provides a concise path from genomics concepts to a deployable classifier.

- What is a gene? A gene is a genomic locus that is transcribed into RNA. Protein-coding genes contain exons that are translated into proteins; non-coding genes produce functional RNAs.
- Coding vs. non-coding genome:
  - Coding regions (exons) comprise ~1–2% of the human genome.
  - Non-coding regions (>98%) include promoters, enhancers, silencers, insulators, and other regulatory elements that control gene expression.
- Promoters: DNA segments proximal to transcription start sites that recruit transcription machinery. Variants here can modulate gene expression.
- Variants: Single-nucleotide variants (SNVs) and small indels may disrupt transcription factor binding motifs within promoters/enhancers, altering gene expression.
- Learning goal: Given a 200bp sequence centered on a variant, learn sequence patterns indicative of specific loci (CHEK2, TBX1, COMT) vs. other promoter variants.

## Methods (brief)
- Data: Ensembl GRCh38 chr22 FASTA, GVF (variants), GFF3 (regulatory features).
- Processing: pandas + PyRanges to find promoter-overlapping variants; pyfaidx to extract 200bp; one-hot encode to (4, 200).
- Labels: 4 classes: Other, CHEK2, TBX1, COMT.
- Model: 1D CNN with two Conv/Pool/Dropout blocks + 2 FC layers; CrossEntropy with class weights.
- Training: 5 epochs, Adam, GPU if available.

## Results (high level)
- Accuracy ~99.99%; macro F1 = 1.00 on the test set.
- See models/classification_report.txt and reports/figures/confusion_matrix.png.
- ROC/PR curves (OvR, micro) are generated to assess imbalanced performance:
  - reports/figures/roc_ovr.png
  - reports/figures/pr_ovr.png

## Repository structure
- src/
  - app.py (FastAPI app: uvicorn src.app:app)
- scripts/
  - 01_prepare_data.py
  - 02_train_model.py
  - 03_plot_metrics.py
  - 04_plot_gene_tracks.py
  - 05_predict.py
  - experimentals/ (legacy utilities)
- data/ (FASTA/GVF/GFF inputs)
- models/ (trained weights, metrics.json)
- reports/
  - figures/ (generated PNGs)
  - report.pdf (manuscript)
- demo/ (static web demo)
- Dockerfile, requirements.txt, LICENSE

## Installation
```bash
python -m venv .venv && . .venv/bin/activate  # or use Conda
pip install -r requirements.txt
```
Place Ensembl data in data/:
- Homo_sapiens.GRCh38.dna.chromosome.22.fa(.fai)
- homo_sapiens-chr22.gvf
- Homo_sapiens.GRCh38.regulatory_features.v114.gff3

## Reproducible pipeline
- Prepare data:
  ```bash
  python scripts/01_prepare_data.py
  ```
- Train:
  ```bash
  python scripts/02_train_model.py
  ```
- Plot results:
  ```bash
  python scripts/03_plot_metrics.py 
  python scripts/04_plot_gene_tracks.py
  ```
- Predict from CLI:
  ```bash
  python scripts/05_predict.py --seq "ACGT..."   # 200bp
  ```

## API and Demo
- Run API:
  ```bash
  uvicorn src.app:app --host 0.0.0.0 --port 8000
  ```
  The API lazily loads artifacts. If models/label_map.json or models/genesis_multiclass.pth are missing, /predict returns 503 with guidance. Train once first:
  ```bash
  python scripts/02_train_model.py
  ```
- Health check:
  ```bash
  curl http://localhost:8000/health
  ```
- Open demo/index.html. The demo uses a professional light theme and shows the API base.

## Docker
Build and run:
```bash
docker build -t genesis .
docker run --rm -p 8000:8000 -e APP_MODULE=src.app:app -v %cd%/models:/app/models genesis
```

## Resources
- GitHub: https://github.com/vanshcodeworks/GENESIS
- Report (PDF): reports/report.pdf
- External link: https://drive.google.com/your-link-here
- Learning & Background (web): demo/learning.html
- Animated Gene Learning (web): demo/genes.html

## Citation
If you use this work, please cite:
- Vansh Goyal. GENESIS: A Deep Learning Framework for Multi-Class Classification of Non-Coding Genomic Variants (2025). https://github.com/vanshcodeworks/GENESIS

## License
MIT License. See LICENSE.

## Analysis utilities
- Saliency maps (gradient-based) to highlight learned sequence motifs:
  ```bash
  python scripts/06_saliency.py       # saves reports/figures/saliency_*.png
  ```