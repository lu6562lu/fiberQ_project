# fiberQ_project

This project processes borehole and fiber seismic data to extract energy, spectral, and source-related features.

## Folder structure

- `scripts/`: core modules (e.g. ESD, picking, resampling, FFT)
- `notebooks/`: exploratory workflows and main analysis
- `data/`: station and fiber mapping files
- `results/`: plots and outputs (excluded by .gitignore)

## Example usage

```python
from scripts.esd_module import calculate_esd_for_stations
from scripts.picking_module import process_event
