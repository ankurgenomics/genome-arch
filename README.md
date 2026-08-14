# genome-arch

**Status: Planning / early development. No training runs, no results, no paper. This README is the plan, not a report.**

## What this is

A companion project to [genome-ft](https://github.com/ankurgenomics/genome-ft). genome-ft fine-tuned an existing
genomic foundation model (Nucleotide Transformer v2) end-to-end. This project is the other half: designing and
training a small architecture **from random initialization**, not fine-tuning a pretrained checkpoint, then
comparing the two honestly on the same evaluation protocol.

## Why

Most "foundation model experience" on the market is adapter/LoRA fine-tuning of an existing checkpoint. That's a
real, useful skill (see genome-ft), but it's a different skill from designing and training an architecture. This
project exists to build that second skill directly, not to claim it before it's true.

## Planned architecture

A small multi-scale block-convolution stack, inspired by the short/medium/long convolution design in Arc
Institute's [StripedHyena2 / Evo2](https://github.com/evo-design/evo) (architecture idea credited, not claimed as
original — the goal is genuine from-scratch implementation and training, not inventing a new architecture family).
Scoped down deliberately:

- Sequence windows in the low kilobase range, not genome-scale context
- A few million parameters, not billions
- Trained from scratch on a single task family first (see below), not pretrained on a large multi-species corpus

## Planned task and evaluation

1. **Sanity check:** train from scratch on [GenomicBenchmarks](https://github.com/ML-Bioinfo-CEITEC/genomic_benchmarks)
   (enhancers/promoters) — the same benchmark genome-ft used — for a direct fine-tuned-vs-from-scratch comparison.
2. **Main task:** non-coding variant effect prediction on [ClinVar](https://www.ncbi.nlm.nih.gov/clinvar/) noncoding
   SNVs/indels, optionally with UCSC conservation tracks (phyloP/phastCons) as an auxiliary input channel.
3. **Evaluation protocol:** same discipline as genome-ft — leakage-free train/val/test split, test set scored once,
   multiple random seeds with reported variance. No cherry-picked checkpoints, no single-seed numbers presented as
   final.

Any published reference numbers (e.g. Evo2, Borzoi, or other public benchmarks) will be cited as context only, never
as a controlled comparison, exactly as genome-ft's README already does.

## Rough milestones

- [ ] Week 1 — data pipeline: ClinVar noncoding labels, conservation tracks, leakage-free split, CNN baseline sanity check
- [ ] Week 2 — architecture implementation, from-scratch training on GenomicBenchmarks, compare against genome-ft's fine-tuned baseline
- [ ] Week 3 — extend to ClinVar noncoding variant task, multi-seed leakage-free eval, publish honest results (or honest negative results)

## Compute plan

Free tier (Google Colab / Kaggle) for architecture development and the GenomicBenchmarks stage. A short paid burst
(~$20-50 on spot GPU pricing) only for the heavier ClinVar stage. Documented here for anyone wondering how a small
project like this gets built on a real budget.

## License

MIT, same as genome-ft.
