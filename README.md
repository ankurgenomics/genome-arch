# genome-arch

A multi-scale convolutional architecture for non-coding genomic variant effect prediction, trained from scratch and
benchmarked directly against fine-tuned foundation model baselines.

## Overview

Companion project to [genome-ft](https://github.com/ankurgenomics/genome-ft). genome-ft fine-tunes an existing
genomic foundation model (Nucleotide Transformer v2) end-to-end — every parameter, no LoRA. genome-arch is the
architecture side of that work: a new model, designed and trained from random initialization, evaluated against the
fine-tuned baseline under an identical protocol.

## The question

Does a small architecture with the right inductive biases for DNA — multi-scale locality, reverse-complement
equivariance, sparse long-range attention — recover most of a general-purpose foundation model's performance on
non-coding variant prediction, at a small fraction of the parameters and pretraining data? Or does the task
genuinely require the scale and broad multi-species pretraining that models like Evo2 bring, and inductive bias
alone can't substitute for it?

That's not a rhetorical question — it's an open, practically important one. Pretraining at frontier scale is
expensive; knowing when a well-designed small model closes most of the gap, and when it doesn't, is exactly the
kind of decision a team building genomic foundation models has to make before committing compute to one approach
over the other. genome-arch answers it on one task, with one honestly-run comparison, instead of assuming the
answer either way.

The evaluation task is scoped across species rather than a single genome, since evolutionary-scale, cross-species
diversity is where inductive bias and scale are most likely to trade off differently than they would on a single,
well-studied genome.

## Architecture

Multi-scale block-convolution stack combining ideas from two recent genomic architectures:

- **Block convolutions**, following the approach described by Radical Numerics' Omnii: instead of a single scalar
  kernel shared across the whole sequence (a standard Toeplitz convolution), each local block gets its own dense,
  learned weight matrix. This allows position-dependent transformations — a promoter-like region and an
  intron-like region can be processed differently — while staying compute-efficient because each matrix is local
  to its block.
- **Multi-scale operators**, following Arc Institute's StripedHyena2 (Evo2): short, medium, and long
  block-convolution scales, so the model reads local motif structure, promoter/enhancer-scale elements, and longer
  regulatory context in parallel.
- **Dynamic sparse attention**, following Omnii: rather than uniform attention across the window, attention is
  gated and conditioned on local sequence context, concentrating long-range compute on positions more likely to
  carry distal regulatory signal instead of attending everywhere equally.
- **Gated mixing** across scales and between the convolutional and attention paths.
- **Reverse-complement equivariance**, following Caduceus (Schiff et al., 2024): DNA is double-stranded, so a
  sequence and its reverse complement should get consistent treatment. genome-ft handled this with RC augmentation
  at training time (flipping sequences and hoping the model generalizes); genome-arch builds it into the
  architecture directly, the way Caduceus does, rather than relying on augmentation to cover it.

Omnii has no released code or weights — this is a from-scratch reimplementation of the architectural ideas
described in Radical Numerics' public preview, not a port of their implementation, and won't match their exact
design or results. StripedHyena2/Evo2's code is open and used as a secondary reference for the multi-scale
convolution mechanics. Single-GPU trainable at the target scale — low millions of parameters, low-kilobase sequence
windows — sized to represent real architectural decisions without requiring a distributed training setup.

## Task and data

Primary task: cross-species regulatory-element and non-coding variant effect prediction.

- **Multi-species whole-genome alignment** — [UCSC Comparative Genomics Lab Cactus alignments](https://cglgenomics.ucsc.edu/data/cactus/),
  providing constrained-element labels and phyloP/phastCons-style conservation scores as both targets and auxiliary
  input across species, not just human.
- **[ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/)** noncoding SNVs/indels — retained as one evaluation slice
  (the largest publicly labeled human noncoding-variant set), not the only one.
- **[GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks)** (enhancers, promoters) — kept
  only as an implementation sanity check against genome-ft's existing fine-tuned baseline on identical data, not
  the headline result.

## Evaluation

Leakage-free train/val/test split. Test set scored once per selected checkpoint. Multiple random seeds, variance
reported. Published numbers are cited as context, never as a controlled comparison. The one controlled,
same-data comparison against genome-ft's fine-tuned baseline happens on the GenomicBenchmarks sanity check, where
both models see identical data and the identical protocol; the multi-species and ClinVar results stand on their
own, since genome-ft was never evaluated on that data.

## Implementation order

Data pipeline and leakage-free splits first, with a CNN baseline to sanity-check the labels. Then a quick
GenomicBenchmarks run to confirm the architecture trains correctly, checked against genome-ft's existing fine-tuned
baseline on identical data. Then the real work: multi-species conservation and ClinVar noncoding variant effect
prediction, with the full multi-seed evaluation.

`src/model.py` implements the architecture described above — block convolutions, multi-scale gated fusion, top-k
sparse attention, and the reverse-complement handling — with passing shape/gradient/involution tests in
`tests/test_model.py` (`python -m pytest tests/`). No training has run yet; this verifies the architecture is
correctly wired, not that it performs well.

## Related work

Non-coding variant effect prediction has several established approaches this project builds on and will be
evaluated against:

- **Enformer** (Avsec et al., 2021) — long-range attention over ~200kb for gene expression prediction from sequence
- **Borzoi** (Calico, 2023) — sequence-to-function across RNA-seq coverage tracks
- **DeepSEA / Sei** — chromatin profile and regulatory effect prediction from sequence
- **SpliceAI** — splice-site effect prediction from sequence context
- **BPNet / ChromBPNet** — base-resolution transcription factor binding prediction
- **Arc Institute Evo / Evo2** (StripedHyena2) — architecture inspiration for the multi-scale convolution design
- [Omnii](https://www.radicalnumerics.ai/blog/omnii-health-preview) (Radical Numerics) — architecture inspiration
  for block convolutions and dynamic sparse attention; closed research preview, no released code or weights
- [Caduceus](https://arxiv.org/abs/2403.03234) (Schiff et al., 2024) — bi-directional, reverse-complement
  equivariant Mamba/state-space DNA model; architecture inspiration for building RC-equivariance directly into the
  model rather than relying on augmentation
- [genome-ft](https://github.com/ankurgenomics/genome-ft) — full-parameter fine-tuning of Nucleotide Transformer v2

## License

MIT
