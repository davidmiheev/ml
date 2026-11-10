# Kaggle writeups: analysis

Notes on top Kaggle solutions: the writeup, the code released with it, and what
actually made the difference.

## Index

| Competition | Year | Domain | Writeups covered | Notes |
| --- | --- | --- | --- | --- |
| *(first entries coming)* | | | | |

## Layout

One folder per competition:

```
kaggle/writeups/<competition-slug>/
  README.md          the analysis (template below)
  snippets/          small excerpts or re-implementations, used to check a claim
  figures/           plots made for the analysis
```

Upstream repositories aren't vendored. They're linked at a pinned commit, so
any reference to a file or line still points to what was read.

## Template

```markdown
# <Competition title>

<link> · <year> · metric: <metric> · <n> teams

## TL;DR
Three to five bullets: what won, and the one or two ideas that carried it.

## The problem as the winners saw it
Data shape, metric quirks, leaks or shifts, and the validation scheme that
tracked the private LB (and the ones that didn't).

## Solutions
### <place>: <team> · [writeup](...) · [code](... @ <commit>)
- Pipeline: data → model(s) → post-processing
- Key ideas, each with its reported gain
- Code notes: where the idea lives in the repo, and anything the writeup leaves out

## What actually mattered
Comparison across the top solutions: shared ideas, differing ones, and the
reported ablations ranked by impact.

## Transferable lessons
What to reuse on the next problem, and when it applies.

## Open questions
Claims I couldn't verify, and things worth trying.
```

## What each analysis checks

- **Writeup vs. code.** Writeups leave things out. The repo is the source of
  truth for hyperparameters, seeds, ensembling weights and post-processing.
- **Gains are attributed.** Each idea is tied to the CV/LB change its author
  reported. Where no number exists, the analysis says so.
- **Public vs. private shake-up.** Which choices made solutions robust to it.
