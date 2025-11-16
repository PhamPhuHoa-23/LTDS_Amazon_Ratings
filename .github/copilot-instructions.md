Project snapshot
- This repo implements a small recommendation-data workflow (NumPy-first). Key folders:
  - `src/` : reusable modules (`data_processing.py`, `models.py`, `visualization.py`).
  - `notebooks/` : numbered notebooks (01..03). Run notebooks in ascending order; earlier notebooks persist outputs for later ones in `data/processed/`.
  - `data/` : `raw/` (source CSV) and `processed/` (generated .npy/.npz files).

Quick architecture & dataflow
- Raw CSV → `notebooks/01_data_exploration.ipynb` (explore, derive simple aggregates) → save outputs to `data/processed/exploration_outputs.npz`.
- Preprocessing and feature engineering live in `src/data_processing.py`. Notebooks and scripts should call functions there rather than reimplementing logic.
- `src/models.py` contains simple recommenders (popularity, CF, SVD, weighted). These are NumPy-only implementations; they expect integer indices for users/products and dense/sparse NumPy arrays.
- `src/visualization.py` contains plotting helpers used by notebooks.

Project conventions the agent must follow
- NumPy-first: prefer vectorized NumPy operations; avoid introducing Pandas unless explicitly requested.
- Persistence: use `np.savez_compressed(...)` or `np.save(...)` into `data/processed/` for artifacts that later notebooks consume. Example file names used by the repo:
  - `data/processed/preprocessed_data.npz`
  - `data/processed/id_mappings.npz`
  - `data/processed/exploration_outputs.npz` (added by notebook 01)
- Prints: keep console output minimal and factual. Do not add decorative separators (no repeated `=` lines, no unicode checkmarks). Use concise Vietnamese sentences for notebook prints and short Vietnamese comments in `src/` (keep English technical terms such as "SVD", "CF", "cosine" where appropriate).
- Notebooks: keep them idempotent and runnable top-to-bottom. If generating artifacts needed by later notebooks, save them under `data/processed/` with stable filenames and document that save in the notebook.

Developer workflows & useful commands
- Create a reproducible environment: `pip install -r requirements.txt` (project uses numpy, matplotlib, seaborn).
- Run a single module's quick sanity checks:
  - `python src/models.py` (prints concise example recommendations)
  - `python src/visualization.py` (runs a minimal demo plot)
- Notebooks: open `notebooks/01_data_exploration.ipynb` in Jupyter or VS Code Notebook. Run in order 01 → 02 → 03 to reuse saved outputs.

Files to inspect when changing behavior
- `src/data_processing.py` — load/sample/clean/filter/create mappings; follow existing function signatures when calling from notebooks.
- `src/models.py` — implementations assume inputs are integer indices and dense NumPy arrays; when adding new models preserve the index-mapping pattern.
- `notebooks/01_data_exploration.ipynb` — now saves `exploration_outputs.npz`; modify only to extend saved keys if later notebooks require them.

When you make changes
- Update or add to `data/processed/` only via code (not by committing dataset dumps). Prefer small, focused `.npz` files containing only the arrays necessary for later steps.
- Keep inline comments short and in Vietnamese. Avoid long essay-style comments in notebooks; notebooks should communicate intent via short markdown headings and short Vietnamese explanations.

If unsure
- Check `notebooks/01_data_exploration.ipynb` to see what outputs it saves and what names later notebooks expect.
- Prefer minimal, local changes — do not refactor global APIs without updating all callers (notebooks and scripts).

Ask the repo owner if you need:
- permission to add heavier dependencies (pandas, scipy) or to change the storage format (e.g., Parquet)
- clarification about which notebook(s) are canonical for preprocessed artifacts

End of instructions