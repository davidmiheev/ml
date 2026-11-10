# DistilHuBERT: speaker-gender classification

March 2025 · Kaggle notebook (P100)

This notebook fine-tunes [`ntu-spml/distilhubert`](https://huggingface.co/ntu-spml/distilhubert)
for binary speaker-gender classification from raw 16 kHz audio, using the
Hugging Face `Trainer`. The resulting model is published as
[`DeepDavid/distilhubert-finetuned-1`](https://huggingface.co/DeepDavid/distilhubert-finetuned-1).

- **Data:** about 14k labelled `.wav` clips, split 80/20 at random (11,148
  train / 2,787 validation). Clips are truncated to 7 s.
- **Training:** 5 epochs, lr 5e-5, batch 8 with gradient accumulation. It
  took about 1 h 50 min on one P100.
- **Result:** **99.75% validation accuracy** (99.0% after the first epoch).
- **Inference:** `pipeline("audio-classification", ...)` over the test
  folder, written to a headerless `id\tgender` TSV.

The notebook also has an unused mel-spectrogram feature-extraction path.

The data was downloaded from Yandex Disk links and isn't included here. Authenticate to the Hub with `HF_TOKEN` in the
environment (or `huggingface-cli login`) before you run it.
