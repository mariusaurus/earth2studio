# Analysing and Plotting TC Tracks

## Notebooks

- **`tracks_slayground_notebook.py`** – Analysing and plotting complete
  tracks from a full ensemble run for a case study on a given storm
- **`plot_tracks_n_fields_notebook.py`** – Plotting tracks and fields for
  individual ensemble members

Both scripts are [JupyText](https://jupytext.readthedocs.io/) Python files
that can be run directly or converted to Jupyter notebooks:

```bash
# Convert to a Jupyter notebook
jupytext --to notebook tracks_slayground_notebook.py
jupytext --to notebook plot_tracks_n_fields_notebook.py
```

## Additional Information

- This directory includes Python scripts with methods for plotting and
  analysing results, which are called from the notebooks
- Each notebook specifies at the beginning what data is required and how to
  produce it using the TC tracking pipeline
