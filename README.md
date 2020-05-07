# rnafold-rts-public
Source code for RTS analysis. https://www.biorxiv.org/content/10.1101/2020.02.10.941153v1

Important files:
* termfold.singularity - singularity recipe for container for running code (including jupyter notebooks).
* DeltaLFE calculation demonstration.ipynb - jupyter notebook demonstrating the delta-LFE computation and allowing it to be easily run on new genomes. The input will be two fasta files, one with CDS sequences and one with matching 3'-UTR sequences. The results are the native, randomized and dLFE graphs.
* config.py - configuration for database connections, etc.
