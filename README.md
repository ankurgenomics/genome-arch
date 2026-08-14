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

Multi-scale block-convolution stack — short, medium, and long convolutional operators (the operator hierarchy is
inspired by Arc Institute's [StripedHyena2 / Evo2](https://github.com/evo-design/evo)) with gated mixing across
scales:

- **Short convolutions** read local motif structure near the variant.
- **Medium convolutions** read promoter/enhancer-scale elements (hundreds of bases).
- **Long convolutions** read regulatory context across the full window.
- **Gated mixing** lets the model weight local vs. distal signal per position, rather than a fixed concatenation.

Target: low millions of parameters, low-kilobase sequence windows — sized to represent real architectural decisions
without requiring frontier-scale compute.

## Task and data

- [GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks) (enhancers, promoters) — same
  dataset as genome-ft, for a direct from-scratch vs. fine-tuned comparison on identical data.
- [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) noncoding SNVs/indels for the primary variant-effect task, with
  UCSC phyloP/phastCons conservation as an auxiliary input channel.

## Evaluation

Leakage-free train/val/test split. Test set scored once per selected checkpoint. Multiple random seeds, variance
reported. Published numbers from Evo2/Borzoi/etc. are cited as context, never as a controlled comparison — the only
fair comparison here is against genome-ft's own fine-tuned baseline, same data, same protocol.

## Implementation order

Data pipeline and leakage-free splits first, with a CNN baseline to sanity-check the labels. Then the architecture,
trained from scratch on GenomicBenchmarks and compared against genome-ft directly. Then the ClinVar task, with
conservation-track fusion and the full multi-seed evaluation.

## Compute

Free-tier GPUs (Colab/Kaggle) for architecture development and the GenomicBenchmarks stage. A short paid burst on
spot pricing (RunPod/vast.ai) for the ClinVar stage, which needs more data and longer windows.

## Related work

- [genome-ft](https://github.com/ankurgenomics/genome-ft) — full-parameter fine-tuning of Nucleotide Transformer v2
- Arc Institute [Evo / Evo2](https://github.com/evo-design/evo) — StripedHyena2 architecture (design inspiration)

## License

MIT
