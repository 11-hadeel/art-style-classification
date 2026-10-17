# Art Style Classification with Vision Transformers

Classifying paintings into five art movements — **Baroque, Expressionism, Impressionism, Realism, Romanticism** — by fine-tuning and comparing several Vision Transformer models on a dataset scraped from [WikiArt](https://www.wikiart.org/).

> Author: Hadeel Taqı — Computer Systems Engineering (Bilişim Sistemleri Mühendisliği)

## Project overview

1. **Data collection** – scraped ~3,500 images per style from WikiArt (17,437 images downloaded).
2. **Preprocessing** – corrupted-image check, perceptual-hash de-duplication (17,350 unique images), resize to 224×224, augmentation (horizontal flip, rotation ±15°, color jitter), giving 52,052 images.
3. **Training** – five pretrained transformer models fine-tuned on a 60/20/20 train/val/test split.
4. **Evaluation** – accuracy, precision, recall, F1, specificity, ROC-AUC, confusion matrix, inference time.

## Results (test set)

| Model | Pretrained checkpoint | Epochs | Accuracy | F1 (weighted) |
|---|---|---|---|---|
| DeiT | `deit_base_patch16_224` (timm) | 60 | **96.80%** | 0.9679 |
| ViT | `google/vit-base-patch16-224` | 10 | 96.34% | 0.9634 |
| BEiT | `microsoft/beit-base-patch16-224` | 10 | 93.51% | 0.9349 |
| Swin | `microsoft/swin-tiny-patch4-window7-224` | 10 | 92.57% | 0.9255 |
| ViT (second run) | `google/vit-base-patch16-224` | 10 | 91.20% | 0.9112 |

DeiT extra metrics: macro specificity 0.9920, macro AUC 0.9973, inference ≈ 0.017 s/image.

## Repository structure

```
├── notebooks/
│   ├── 01_data_collection_and_preprocessing.ipynb
│   ├── 02_vit.ipynb
│   ├── 03_deit.ipynb
│   ├── 04_swin.ipynb
│   ├── 05_beit.ipynb
│   └── 06_vit_second_run.ipynb
├── results/            # confusion matrices, ROC curves, training curves
├── requirements.txt
└── README.md
```

## Dataset

Images are **not** stored in this repository (size and WikiArt terms). Download the processed dataset here:
**[Processed dataset (Google Drive)](https://drive.google.com/drive/folders/16XcA4AlwB2evKKmupt2xmmHK4pzCcsxE?usp=sharing)**
or recreate it by running `01_data_collection_and_preprocessing.ipynb`.

Expected layout: `dataset/<style_name>/*.jpg` (one folder per class).

## How to run

The notebooks were developed on **Google Colab (GPU)**.

```bash
pip install -r requirements.txt
```

1. Put the dataset in your Google Drive and update the path in the first cells (`/content/drive/MyDrive/...`).
2. Run the notebooks in order. Each model notebook splits the data (60/20/20, seed 42), trains, evaluates, and saves a `.pth` file.

## Known limitations

- **Possible data leakage:** augmentation was applied *before* the train/val/test split, so an original image and its augmented copies can land in different splits. Reported accuracies are likely optimistic. A cleaner protocol is to split the original images first and augment only the training set.
- Models were trained for different numbers of epochs (DeiT: 60, others: 10), so the comparison is not perfectly fair.
- Data scraped from WikiArt is for academic/research use only; check WikiArt's terms and the copyright status of each artwork before redistribution.

## License

*(choose one, e.g. MIT for the code)*

## Acknowledgements

[WikiArt](https://www.wikiart.org/), [timm](https://github.com/huggingface/pytorch-image-models), [Hugging Face Transformers](https://github.com/huggingface/transformers).
