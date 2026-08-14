# genome-arch

A small multi-scale convolutional architecture for non-coding genomic variant effect prediction, trained from
scratch and benchmarked against fine-tuned foundation model baselines.

## Overview

genome-arch is a from-scratch architecture study for non-coding variant effect prediction, built as the direct
counterpart to [genome-ft](https://github.com/ankurgenomics/genome-ft) (full-parameter fine-tuning of Nucleotide
Transformer v2). Where genome-ft adapts an existing pretrained foundation model, genome-arch designs and trains a
new architecture from random initialization, then compares the two under an identical evaluation protocol.

## Motivation

Most public "foundation model" work in genomics falls into two categories: pretraining frontier-scale models (Evo2,
HyenaDNA) on massive multi-species corpora, or fine-tuning those models for a downstream task. There's less public
work directly comparing what a small, deliberately-designed architecture trained from scratch can achieve against a
fine-tuned baseline, on the same task, same data, same evaluation protocol. genome-arch is that comparison.

## Architecture

A multi-scale block-convolution stack: short, medium, and long convolutional operators (inspired by the operator
hierarchy in Arc Institute's [StripedHyena2 / Evo2](https://github.com/evo-design/evo)) stacked with lightweight
gating, operating on sequence windows in the low-kilobase range around each variant. Design goals:

- **Short convolutions** capture local motif structure near the variant.
- **Medium convolutions** capture patterns over hundreds of bases (promoter/enhancer-scale elements).
- **Long convolutions** capture longer-range regulatory context within the window.
- Gated mixing between scales, rather than a fixed concatenation, so the model can weight local vs. distal signal
  per position.

Parameter count target: low millions — enough to represent genuine architectural decisions (kernel scales, gating,
channel mixing) without requiring frontier-scale compute.

## Task and data

- **Benchmark warm-up:** [GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks)
  (`human_enhancers_cohn`, `human_nontata_promoters`) — the same dataset genome-ft uses, for a direct
  fine-tuned-vs-from-scratch comparison on identical data.
- **Primary task:** non-coding variant effect prediction on [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) SNVs
  and indels, with UCSC phyloP/phastCons conservation tracks as an auxiliary input channel alongside raw sequence.

## Evaluation protocol

Same discipline as genome-ft: leakage-free train/validation/test split, test set scored exactly once per selected
checkpoint, multiple random seeds with reported variance, no cherry-picked runs. Any published reference numbers
(Evo2, Borzoi, or other public benchmarks) are cited as context, never presented as a controlled comparison.

## Roadmap

- [ ] Data pipeline: ClinVar labels, conservation track alignment, leakage-free split, CNN baseline sanity check
- [ ] Architecture implementation, from-scratch training run on GenomicBenchmarks
- [ ] ClinVar noncoding variant task, multi-seed evaluation, results write-up

## Compute

Development and the GenomicBenchmarks stage run on free-tier GPUs (Colab/Kaggle). The ClinVar stage, which needs
more data and longer sequence windows, uses a short paid burst on spot GPU pricing (RunPod/vast.ai).

## Related work

- [genome-ft](https://github.com/ankurgenomics/genome-ft) — full-parameter fine-tuning of Nucleotide Transformer v2
- Arc Institute [Evo / Evo2](https://github.com/evo-design/evo) — StripedHyena2 architecture (design inspiration)

## License

MIT
