# ORE_in_CNNs_and_humans

**Code for the project investigating the Other-Race Effect (ORE) in CNNs and humans.**

This repository accompanies our research exploring how convolutional neural networks (CNNs) trained on different visual experiences compare with human behavioral data on the Other-Race Effect — a perceptual bias where recognition accuracy differs across faces of different races.

Our analyses combine model lesioning, activations, and behavioral correlates to better understand representational similarities and gaps between machines and human perception. :contentReference[oaicite:2]{index=2}

---

## 📁 Repository structure

| Folder | Description |
|--------|-------------|
| `activations/` | Scripts for extracting and saving model activations |
| `experiment_data/` | Behavioral measurements and associated files |
| `figures/` | Figures used in analyses and drafts |
| `lesioning/` | Code for model lesioning experiments |
| `utils/` | Utility scripts and helper functions |
| `plots.ipynb` | Jupyter notebook with key analysis plots |

---

## 📦 Requirements

Dependencies include:

- Python 3.x  
- `numpy`, `matplotlib`, `pandas`  
- Deep learning framework (e.g., PyTorch or TensorFlow)  
- Jupyter notebook

Install via:

```bash
pip install -r requirements.txt
