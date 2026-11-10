# ml

Machine-learning work I've collected over the years: competition notebooks,
contest solutions, fine-tuning experiments, and close readings of Kaggle
material I found worth studying.

Some of it is polished, some of it is a working log kept as it happened. Each
folder has its own README that says which is which.

## What's here

| Folder | What it is | When |
| --- | --- | --- |
| [`kaggle/writeups/`](kaggle/writeups/) | **Analysis of Kaggle winning writeups and the repos behind them.** What the top teams did, what actually moved the score, and what carries over to other problems. | ongoing |
| [`kaggle/playground-s5e1/`](kaggle/playground-s5e1/) | *Forecasting Sticker Sales* (Playground S5E1): my own notebooks next to the community notebooks I learned the most from. | Jan 2025 |
| [`yandex-coderun/`](yandex-coderun/) | Yandex CodeRun ML track: 42 short problems in one notebook, plus two full contest solutions. | 2025–2026 |
| ↳ [`happiness/`](yandex-coderun/happiness/) | *Счастье*: scheduling model rollouts to devices under bandwidth limits. Six scenarios, from an optimal solution to heap greedy with annealing and ruin-and-recreate. | Sep 2026 |
| ↳ [`semantic-switch/`](yandex-coderun/semantic-switch/) | *Смысловой переключатель*: few-shot rule induction on a frozen `rubert-tiny2`, with episode-centred metric learning, a rule codebook, and joint decoding for exact match. | Sep 2026 |
| [`experiments/distilhubert-voice-gender/`](experiments/distilhubert-voice-gender/) | Fine-tuning DistilHuBERT for binary speaker-gender classification from raw audio. | Mar 2025 |

## Kaggle writeups

The [`kaggle/writeups/`](kaggle/writeups/) folder is where most new work goes.
For each competition I go through the top writeups and the code released with
them, then write up:

- **the problem as the winners saw it**: metric quirks, leakage, how the data
  was actually shaped, and what the validation scheme needed to be;
- **the solution**: architecture, features, training tricks and post-processing,
  separated from the noise that's in every writeup;
- **what mattered**: ablations from the authors where they exist, and checks
  against their repo where they don't;
- **transferable lessons**: the parts worth reusing on the next problem.

The index and the template for these notes are in
[`kaggle/writeups/README.md`](kaggle/writeups/README.md).

## Conventions

- **No competition data in the repo.** Notebooks expect the data next to them
  or at Kaggle's `/kaggle/input/...` paths. Each folder's README says how to
  get it, usually with the
  [Kaggle CLI](https://github.com/Kaggle/kaggle-api)
  (`kaggle competitions download -c <slug>`).
- **No credentials in the repo.** Notebooks read tokens from the environment
  (`HF_TOKEN`, `~/.kaggle/kaggle.json`).
- **Other people's work is labelled.** Community notebooks and solutions kept
  for study are in `community/` folders and are credited with a link to the
  original. My notes on them are mine; the code belongs to its authors.

## License

[BSD 2-Clause](LICENSE) for my own code. Third-party notebooks keep their
authors' original terms.
