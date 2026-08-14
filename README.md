# genome-arch

A multi-scale convolutional architecture for non-coding genomic variant effect prediction, trained from scratch and
benchmarked directly against fine-tuned foundation model baselines.

## Overview

Companion project to [genome-ft](https://github.com/ankurgenomics/genome-ft). genome-ft fine-tunes an existing
genomic foundation model (Nucleotide Transformer v2) end-to-end — every parameter, no LoRA. genome-arch is the
architecture side of that work: a new model, designed and trained from random initialization, evaluated against the
fine-tuned baseline under an identical protocol.

## Why this comparison matters

Fine-tuning and architecture design are different skills. Most "foundation model" portfolios only demonstrate the
first. genome-arch demonstrates the second, on the same data and the same evaluation bar as genome-ft, so the two
are directly comparable rather than asserted separately.

## Architecture

Multi-scale block-convolution stack — short, medium, and long convolutional operators with gated mixing across
scales:

- **Short convolutions** read local motif structure near the variant.
- **Medium convolutions** read promoter/enhancer-scale elements (hundreds of bases).
- **Long convolutions** read regulatory context across the full window.
- **Gated mixing** lets the model weight local vs. distal signal per position, rather than a fixed concatenation.

The short/medium/long operator hierarchy follows the design used in Arc Institute's StripedHyena2 (Evo2). Single-GPU
trainable at the target scale — low millions of parameters, low-kilobase sequence windows — sized to represent real
architectural decisions without requiring a distributed training setup.

## Task and data

- [GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) (enhancers, promoters) — same
  dataset as genome-ft, for a direct from-scratch vs. fine-tuned comparison on identical data.
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) noncoding SNVs/indels for the primary variant-effect task, with
  UCSC phyloP/phastCons conservation as an auxiliary input channel.

## Evaluation

Leakage-free train/val/test split. Test set scored once per selected checkpoint. Multiple random seeds, variance
reported. Published numbers are cited as context, never as a controlled comparison — the only fair comparison here
is against genome-ft's own fine-tuned baseline, same data, same protocol.

## Implementation order

Data pipeline and leakage-free splits first, with a CNN baseline to sanity-check the labels. Then the architecture,
trained from scratch on GenomicBenchmarks and compared against genome-ft directly. Then the ClinVar task, with
conservation-track fusion and the full multi-seed evaluation.

## Related work

Non-coding variant effect prediction has several established approaches this project builds on and will be
evaluated against:

- **Enformer** (Avsec et al., 2021) — long-range attention over ~200kb for gene expression prediction from sequence
- **Borzoi** (Calico, 2023) — sequence-to-function across RNA-seq coverage tracks
- **DeepSEA / Sei** — chromatin profile and regulatory effect prediction from sequence
- **SpliceAI** — splice-site effect prediction from sequence context
- **BPNet / ChromBPNet** — base-resolution transcription factor binding prediction
- **Arc Institute Evo / Evo2** (StripedHyena2) — architecture inspiration for the multi-scale convolution design
- [genome-ft](https://github.com/ankurgenomics/genome-ft) — full-parameter fine-tuning of Nucleotide Transformer v2

## License

MIT
