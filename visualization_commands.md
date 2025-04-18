# Visualization Commands for Regional Climate Analysis

> **Note on `--consistent-range` flag**: When using this flag, do not specify the `--experiment` parameter. The command will automatically generate plots for both SSP245 and SSP585 scenarios with consistent y-axis ranges.

## 1. Tambacounda, Senegal (Rainy Season: June-October) - HadGEM3-GC31-LL (UK)

### Compound Metric Heatmaps
```bash
# CWHD (Consecutive Wet-Hot Days) for June-October
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric cwhd --months 6-10 --model hadgem3_gc31_ll
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric cwhd --months 6-10 --model hadgem3_gc31_ll

# Hot-Dry Frequency for all months (drought/heat stress patterns year-round)
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric hot_dry_frequency --months all --model hadgem3_gc31_ll
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric hot_dry_frequency --months all --model hadgem3_gc31_ll
```

### Temperature Time Series
```bash
# Warm days percentage for August (peak rainy season) with consistent y-axis
python generate_plots.py --viz-type timeseries --variable temperature --month 8 --metric tm90p --consistent-range --model hadgem3_gc31_ll
```

### Precipitation Time Series
```bash
# Maximum 5-day precipitation for August with consistent y-axis
python generate_plots.py --viz-type timeseries --variable precipitation --month 8 --metric rx5day --consistent-range --model hadgem3_gc31_ll
```

## 2. Alicante, Spain (Fall: September-November) - EC-Earth3-CC (Europe)

### Compound Metric Heatmaps
```bash
# WSPI (Warm-Season Precipitation Intensity) for September-November
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric wspi --months 9-11 --model ec_earth3_cc
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric wspi --months 9-11 --model ec_earth3_cc

# Consecutive Dry Days for all months (drought patterns year-round)
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric cdd --months all --model ec_earth3_cc
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric cdd --months all --model ec_earth3_cc
```

### Temperature Time Series
```bash
# Maximum temperature for October (cold drop events) with consistent y-axis
python generate_plots.py --viz-type timeseries --variable temperature --month 10 --metric tm_max --consistent-range --model ec_earth3_cc
```

### Precipitation Time Series
```bash
# Maximum 1-day precipitation for October (flash floods) with consistent y-axis
python generate_plots.py --viz-type timeseries --variable precipitation --month 10 --metric rx1day --consistent-range --model ec_earth3_cc
```

## 3. Bergen, Norway (Winter: December-February) - NorESM2-MM (Norway)

### Compound Metric Heatmaps
```bash
# WWPD (Warm Winter Precipitation Days) for December-February
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric wpd --months 12,1,2 --model noresm2_mm
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric wpd --months 12,1,2 --model noresm2_mm

# Cold Spell Days for all months (seasonal cold period patterns)
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric cold_spell_days --months all --model noresm2_mm
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric cold_spell_days --months all --model noresm2_mm
```

### Temperature Time Series
```bash
# Cold days percentage for January with consistent y-axis
python generate_plots.py --viz-type timeseries --variable temperature --month 1 --metric tm10p --consistent-range --model noresm2_mm
```

### Precipitation Time Series
```bash
# Total precipitation for January (snow-to-rain transition) with consistent y-axis
python generate_plots.py --viz-type timeseries --variable precipitation --month 1 --metric prcptot --consistent-range --model noresm2_mm
```

## Location Maps for Regional Context
```bash
# Generate location maps for reference
python generate_plots.py --viz-type location --variable temperature --experiment ssp245 --month 8 --model hadgem3_gc31_ll # Tambacounda (rainy season)
python generate_plots.py --viz-type location --variable temperature --experiment ssp245 --month 10 --model ec_earth3_cc   # Alicante (fall)
python generate_plots.py --viz-type location --variable temperature --experiment ssp245 --month 1 --model noresm2_mm     # Bergen (winter)
```

## Batch Visualization Commands
```bash
# For efficiency, these commands can be combined into a shell script

# Tambacounda visualizations
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric cwhd --months 6-10 --model hadgem3_gc31_ll
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric cwhd --months 6-10 --model hadgem3_gc31_ll
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric hot_dry_frequency --months all --model hadgem3_gc31_ll
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric hot_dry_frequency --months all --model hadgem3_gc31_ll
python generate_plots.py --viz-type timeseries --variable temperature --month 8 --metric tm90p --consistent-range --model hadgem3_gc31_ll
python generate_plots.py --viz-type timeseries --variable precipitation --month 8 --metric rx5day --consistent-range --model hadgem3_gc31_ll
python generate_plots.py --viz-type location --variable temperature --experiment ssp245 --month 8 --model hadgem3_gc31_ll
```

```bash
# Alicante visualizations
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric wspi --months 9-11 --model ec_earth3_cc
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric wspi --months 9-11 --model ec_earth3_cc
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric cdd --months all --model ec_earth3_cc
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric cdd --months all --model ec_earth3_cc
python generate_plots.py --viz-type timeseries --variable temperature --month 10 --metric tm_max --consistent-range --model ec_earth3_cc
python generate_plots.py --viz-type timeseries --variable precipitation --month 10 --metric rx1day --consistent-range --model ec_earth3_cc
python generate_plots.py --viz-type location --variable temperature --experiment ssp245 --month 8 --model ec_earth3_cc
```

```bash
# Bergen visualizations
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric wpd --months 12,1,2 --model noresm2_mm
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric wpd --months 12,1,2 --model noresm2_mm
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp245 --metric cold_spell_days --months all --model noresm2_mm
python generate_plots.py --viz-type heatmap --variable temperature --experiment ssp585 --metric cold_spell_days --months all --model noresm2_mm
python generate_plots.py --viz-type timeseries --variable temperature --month 1 --metric tm10p --consistent-range --model noresm2_mm
python generate_plots.py --viz-type timeseries --variable precipitation --month 1 --metric prcptot --consistent-range --model noresm2_mm
python generate_plots.py --viz-type location --variable temperature --experiment ssp245 --month 8 --model noresm2_mm
```