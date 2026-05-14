# mdm-2026

Student material for **Multimedia Data Management M** — University of Bologna, academic year 2025–2026.

This repository contains the slides, notebooks, and supporting material for **Exercise 1.E** (multimodal fusion / reference architecture) and **Exercise 1.F** (classic vs modern Flickr8k retrieval).

---

## What's in this repo

```text
mdm-2026/
├── README.md
└── content/
    ├── exercise/
    │   └── 0.03.Exercises.pdf            # Official exercise sheet (canonical source)
    ├── lab/
    │   ├── 1E/
    │   │   └── 1E.pdf                    # Exercise 1.E slides — fixed reference architecture
    │   └── 1F/
    │       ├── 1F.pdf                    # Exercise 1.F slides — classic vs modern retrieval
    │       ├── evaluation_queries.json   # Shared queries + relevance rules
    │       ├── Lab_1F-1_Classic_Retrieval_Baseline.ipynb
    │       └── Lab_1F-2_Modern_CLIP_FAISS.ipynb
    └── papers/
        ├── A_Multimodal_Framework_for_Depression_Detection.pdf   # Background, optional
        └── SigLIP2.pdf                                            # Background, optional
```

---

## Read first

1. `content/exercise/0.03.Exercises.pdf` — the official course exercise sheet.
2. `content/lab/1E/1E.pdf` — Exercise 1.E slides. Defines the fixed reference architecture both Part F notebooks map onto (offline ingestion, online query path, media/content DB, structured store, semi-structured store, feature/descriptor DB, query processor, results/visualization).

   **Your goal in Exercise 1.E** — take your own project and map each of its parts onto the reference architecture above. The point is to work concretely on your specific project, not in the abstract, so you build a clear sense of where each component (stores, descriptors, query engine, results) lives in the architecture. Exercise 1.F then turns that mapping into runnable code over Flickr8k.
3. `content/lab/1F/1F.pdf` — Exercise 1.F slides. Walks through the two retrieval pipelines.

Papers under `content/papers/` are background only — useful for motivation but not required to run the notebooks.

---

## Run Exercise 1.F

Exercise 1.F compares two retrieval pipelines over Flickr8k:

- **Pipeline 1 (classic):** structured metadata + caption TF-IDF + handcrafted HSV/edge descriptors + required **k-NN** over handcrafted visual descriptors. No deep learning.
- **Pipeline 2 (modern):** CLIP ViT-B/32 embeddings + **FAISS `IndexFlatIP`**.

### 1. Clone and enter the repo

```bash
git clone https://github.com/ghasempouri1984/mdm-2026.git
cd mdm-2026
```

### 2. Install dependencies

Python 3.10 or newer is recommended. From the repo root:

```bash
pip install jupyterlab ipykernel datasets transformers torch faiss-cpu scikit-learn pandas numpy matplotlib pillow opencv-python
```

Notes:

- A GPU is optional. `Lab_1F-2_Modern_CLIP_FAISS.ipynb` auto-detects CUDA and falls back to CPU.
- On Windows, install `faiss-cpu` from PyPI (the GPU build is not required).
- The first run downloads Flickr8k via Hugging Face Datasets, so an internet connection is needed once.

### 3. Launch the notebooks

From the repo root:

```bash
jupyter lab
```

VS Code's notebook UI works too — open the repo folder, then open the notebook files.

### 4. Run the notebooks in this order

1. **`content/lab/1F/Lab_1F-1_Classic_Retrieval_Baseline.ipynb`** — keep `FAST_DEV = True`, run all cells. Builds TF-IDF + handcrafted HSV/edge descriptors, runs k-NN visual retrieval, computes P@K / R@K, and saves `classic_results_fast_dev.json`.
2. **`content/lab/1F/Lab_1F-2_Modern_CLIP_FAISS.ipynb`** — keep `FAST_DEV = True`, run all cells. Loads CLIP ViT-B/32, builds a FAISS `IndexFlatIP` index, and overlays its results onto the classic results from step 1.

Running Notebook 2 before Notebook 1 still works, but the classic-vs-modern comparison plot will be skipped because it reads the JSON produced by Notebook 1.

### 5. FAST_DEV vs full run

Both notebooks expose:

```python
FAST_DEV = True
FAST_DEV_LIMIT = 200
```

- `FAST_DEV=True` uses a 200-image subset. Results are illustrative — they are for debugging and demo, not benchmark numbers.
- `FAST_DEV=False` runs on the full Flickr8k dataset. Expect a longer runtime and a larger Hugging Face cache.

### 6. Where outputs are written

Outputs land next to the notebooks under `content/lab/1F/outputs/`, with figures under `content/lab/1F/outputs/figures/`. JSON files are tagged with the mode:

- `classic_results_fast_dev.json` / `classic_results_full.json`
- `modern_clip_results_fast_dev.json` / `modern_clip_results_full.json`
- `dataset_signature_*.json` and `clip_image_embeddings_*.npy` (cached embeddings)

### 7. Expected warnings

`evaluation_queries.json` currently ships with placeholder `reference_image_id` values. For image and fused queries the notebooks deterministically fall back to the first sorted relevant image in the loaded subset, and you will see a warning such as:

```text
Reference image dog_snow_reference not found in this run; using <id> for dog_snow_001.
```

This is expected. Visual and fused metrics should be treated as non-final until the placeholders are replaced with real Flickr8k filenames.

---

## Background references

- **Architecture baseline:** the reference architecture from Exercise 1.E (`content/lab/1E/1E.pdf`). Both Part F notebooks map their code onto this architecture.
- **Modern pipeline:** CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) + FAISS `IndexFlatIP`.
- **Optional background:** `content/papers/SigLIP2.pdf` describes a newer alternative vision-language encoder; it is not the implemented Pipeline 2.
- **Optional background:** `content/papers/A_Multimodal_Framework_for_Depression_Detection.pdf` is motivational context only; it does not define Pipeline 1.
- **Dataset:** Flickr8k (~8k images + 5 captions each), pulled automatically from Hugging Face Datasets (`jxie/flickr8k`).
