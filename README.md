# Style-Based Book Recommendation System (VADES_Fuse)

A deep learning system that recommends books based on **writing style** rather than genre or topic. It combines handcrafted stylometric features with Universal Sentence Encoder (USE) semantic embeddings, trained using contrastive learning to cluster documents by author style.

---

## Overview

Most book recommendation systems rely on genre, keywords, or collaborative filtering. This project takes a different approach: if you enjoy the *way* an author writes — their sentence rhythm, vocabulary richness, punctuation habits — you might enjoy other authors with a similar style, even across genres.

The core model, **VADES_Fuse**, learns a joint embedding space where documents written in a similar style are placed close together, regardless of their topic.

---

## How It Works

### 1. Data
- Source: [Project Gutenberg](https://www.gutenberg.org/) books (HTML format), organised by author
- Each book is cleaned (Gutenberg boilerplate stripped) and chunked into overlapping 4000-character segments
- Chunks are split at the **book level** (train/val/test = 70/15/15) to prevent data leakage

### 2. Feature Extraction (300-D style vector)
Each text chunk is converted into a 300-dimensional handcrafted style vector covering:

| Feature Group | Description |
|---|---|
| Structural | Text length, word count, sentence count, avg sentence/word length |
| Punctuation | Frequency of 11 punctuation types per word |
| Function words | Frequency of 60 common function words |
| Lexical | Type-token ratio, hapax legomena rate |
| Syllables | Total syllable count, syllables per word |
| Readability | Flesch, Flesch-Kincaid, SMOG, Coleman-Liau scores |
| POS tags | 14 part-of-speech frequencies (spaCy) |
| NER | 14 named entity type frequencies (spaCy) |
| Char n-grams | Top-50 character bigram/trigram/4-gram frequencies |
| Word bigrams | Top-50 word bigram frequencies |
| Distributional | Vocabulary entropy, max word frequency |
| Extra | Digit ratio, ALL-CAPS word ratio, stop-word ratio |

### 3. Semantic Embeddings (512-D)
Each chunk is also encoded using Google's **Universal Sentence Encoder (USE)**, capturing semantic meaning at the sentence level.

### 4. Model: VADES_Fuse
A fusion encoder that concatenates the USE embedding (512-D) and style vector (300-D), then projects them to a 300-D joint embedding via a two-layer MLP.

```
[USE embedding (512)] ──┐
                         ├─► Linear(812→1024) → ReLU → Dropout → Linear(1024→300) → doc_emb (300-D)
[Style vector (300)]  ──┘
```

The model also learns a separate author embedding table (one vector per author), trained jointly.

### 5. Training: Contrastive Cosine Loss
The model is trained with a margin-based contrastive loss:
- **Positive pairs**: document ↔ its author embedding; document ↔ its own style features
- **Negative pairs**: K=5 randomly sampled other authors; K=5 permuted style features from the batch
- Loss weights: 80% style alignment, 20% author alignment (`alpha=0.8`)
- Optimizer: AdamW with `ReduceLROnPlateau` scheduler; gradient clipping at 2.0

### 6. Recommendation
At inference, a query document is embedded and compared to all other documents using cosine similarity. Top-K most similar chunks (optionally filtered to same/different author) are returned.

---

## Project Structure

```
style_based_book_recommendation_project.ipynb   # Main notebook (end-to-end pipeline)
```

The notebook expects a Google Drive directory structured as:

```
MyDrive/final_year_project_books_folder/
├── gutenberg_books/
│   ├── AuthorName/
│   │   └── book_title.html
│   └── ...
├── feat_cache/          # Auto-created: cached feature arrays (.joblib)
├── vades_final_rebuild/ # Auto-created: model checkpoints (.pth)
├── splits_final/        # Auto-created: train/val/test CSVs
└── embeddings_cache/    # Auto-created: saved embeddings for recommender use
```

---

## Requirements

```txt
torch
numpy
pandas
scikit-learn
scipy
spacy
textstat
beautifulsoup4
sentence-transformers
tensorflow
tensorflow-hub
tensorflow-text
joblib
tqdm
matplotlib
seaborn
```

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn scipy spacy textstat beautifulsoup4 \
            sentence-transformers tensorflow tensorflow-hub tensorflow-text \
            joblib tqdm matplotlib seaborn
python -m spacy download en_core_web_sm
```

> **Note:** The notebook is designed to run on **Google Colab** with GPU support and Google Drive mounted. Adjust paths if running locally.

---

## Usage

Open `style_based_book_recommendation_project.ipynb` in Google Colab and run cells top to bottom. The pipeline will:

1. Mount Google Drive and set up directories
2. Load/build the chunked book dataset (cached to CSV)
3. Extract and cache style feature vectors (300-D)
4. Compute and cache USE embeddings (512-D)
5. Split data at the book level (no leakage)
6. Train VADES_Fuse with contrastive loss for 15 epochs
7. Extract final normalised document embeddings
8. Run evaluation (nearest-centroid author attribution accuracy)
9. Generate visualisations: T-SNE, confusion matrix, MSE analysis
10. Save all embeddings to `embeddings_cache/` for use in a separate recommender notebook

To get recommendations after training:

```python
results = recommend_similar_docs(query_idx=1000, top_k=5, same_author=False)
for r in results:
    print(f"[{r['score']:.3f}] {r['author']}: {r['text_preview']}")
```

---

## Evaluation

The model is evaluated on two tasks:

**Author Attribution (nearest-centroid)**
Author centroids are computed from the training set; test chunks are classified by nearest centroid (Euclidean distance). Accuracy is compared against a random baseline (1/num_authors).

**Stylistic Feature Prediction (SVR)**
An SVR is trained to predict each style feature from the learned document embeddings using 5-fold cross-validation. Results are reported as MSE ± std per feature group, measuring how much style information is encoded in the embedding space.

---

## Visualisations

The notebook produces four saved figures:

| File | Description |
|---|---|
| `tsne_visualization_vades_fuse.png` | 2D T-SNE of test document embeddings coloured by author, with learned author centroids overlaid |
| `confusion_matrix.png` | Normalised confusion matrix for author attribution on the test set |
| `style_mse_analysis.png` | MSE distributions (histogram, box plot, per-author, CDF) for style reconstruction |
| `training_losses.png` | Training and validation loss curves across epochs |

---

## Key Design Decisions

- **Book-level splitting** ensures no chunks from the same book appear in both train and test sets, preventing inflated accuracy from near-duplicate passages.
- **Style features are fit-scaled on train only** (mean/std saved to `style_scaler.joblib`) to avoid leakage into val/test.
- **Alpha=0.8** weighting prioritises style alignment over author identity in the loss, keeping the embedding space style-driven.
- **USE over BERT**: USE produces fixed 512-D sentence-level embeddings efficiently without fine-tuning, suitable for long document chunks.

---

## Acknowledgements

- Book data sourced from [Project Gutenberg](https://www.gutenberg.org/) (public domain)
- Universal Sentence Encoder via [TensorFlow Hub](https://tfhub.dev/google/universal-sentence-encoder/4)
- Readability metrics via the [`textstat`](https://pypi.org/project/textstat/) library
- NLP pipeline via [spaCy](https://spacy.io/) (`en_core_web_sm`)
