# Data

The SDWPF wind-farm dataset is **not** included in this repository.

1. Download the `sdwpf_full` files from the Hugging Face mirror
   <https://huggingface.co/datasets/AI4Climate/WindFarm_raw/tree/main> (folder `sdwpf_full`):
   - `sdwpf_2001_2112_full.parquet` (about 280 MB, 11.4 million rows)
   - `sdwpf_turb_location_elevation.csv` (134 turbines)
2. Put both files in this folder, or keep them elsewhere and pass `--data-dir /path/to/them`
   (or set the `SDWPF_DATA_DIR` environment variable).

```
data/sdwpf_2001_2112_full.parquet
data/sdwpf_turb_location_elevation.csv
```

The data comes from the Spatial Dynamic Wind Power Forecasting (SDWPF) dataset released with the
KDD Cup 2022 challenge: SCADA records of 134 turbines plus weather values, January 2020 to
December 2021. Please check the terms of use on the original sources before reusing it, and cite
the dataset paper (see the main README).

Facts checked against the file itself:

| | |
|---|---|
| Rows | 11,361,190 (18 columns) |
| Turbines | 134 (`TurbID` 1 to 134) |
| Period | 2020-01-01 00:10 to 2022-01-01 00:10 |
| Sampling | **every 15 minutes during 2020, every 10 minutes during 2021** (April 2020 is missing) |
| Rows with missing sensor values | 496,998 (about 4.4 %) |
| `Patv` (active power, kW) | -9.31 to 1561.38; negative values are the turbine drawing power while idle |

The scripts in `src/` use only `TurbID`, `Tmstamp`, the five inputs `Wspd`, `Wdir`, `Etmp`,
`Itmp`, `RelH` and the target `Patv`.
