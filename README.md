# Traffic-monitoring-Challenge

## 🚀 Raspberry Pi Setup Guide

### Step 0: Clone repository
```bash
cd ~
git clone https://github.com/mikeandrusyak/Traffic-monitoring-Challenge.git Traffic-monitoring-Challenge
cd Traffic-monitoring-Challenge
```

### Step 1: Install System Packages
First, install system packages using the provided script:

```bash
chmod +x scripts/install_system_packages.sh
./scripts/install_system_packages.sh
```

### Step 2: Create Virtual Environment
Create virtual environment that can access system packages:

```bash
python3 -m venv .venv --system-site-packages
source .venv/bin/activate
```

### Step 3: Install Pip Packages
Install remaining packages through pip:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify Installation
Test that everything works:

```bash
python3 -c "
import cv2
import numpy as np
from picamera2 import Picamera2
import RPi.GPIO
print('All packages imported successfully!')
print('OpenCV version:', cv2.__version__)
print('NumPy version:', np.__version__)
"
```

### Why This Approach?

- **System packages via APT**: Raspberry Pi optimized versions
- **Python packages via pip**: Latest versions that work with system packages
- **--system-site-packages**: Allows virtual environment to access system packages

### Original Problem

The original requirements.txt contained:
- `arandr==0.1.11` - This is a GUI tool, not a Python package
- Many system packages that should be installed via APT
- Exact versions that may not be available on all platforms

This approach separates system dependencies from Python dependencies for better compatibility.

---

## 📊 Data Wrangling

### Goal
The main goal of the data processing stage is to extract objects (vehicles) from raw data that can be used for traffic analysis. Raw data contains many artifacts: fragmented tracks, static objects, detector noise, so classification and merging of fragments into unified objects is required.

### Data Exploration
The **`notebooks/raw_data_analysis.ipynb`** notebook is used for initial exploration and analysis of raw data. It allows you to:
- Load data from database or CSV file
- Visualize object trajectories
- Analyze category distribution
- Check fragment merging results
- Experiment with processing parameters

### Automatic Processing of All Sessions
The **`data_pipeline.py`** script is used to process all sessions.

#### Run with default parameters:
```bash
python data_pipeline.py
```

#### Run with database loading:
```bash
python data_pipeline.py --from-database
```

#### Configure processing parameters:
```bash
python data_pipeline.py \
  --input data/raw_traffic_data.csv \
  --output data/processed_traffic_data.csv \
  --categories Noise Partial Static Ghost \
  --time-gap 1.5 \
  --space-gap 40 \
  --size-sim 0.2
```

**Parameters:**
- `--input` — path to input CSV file (default: `data/raw_traffic_data.csv`)
- `--output` — path to output CSV file (default: `data/processed_traffic_data.csv`)
- `--from-database` — load data from database instead of CSV
- `--categories` — categories to merge (default: `Noise Partial Static Ghost`)
- `--time-gap` — maximum time gap between fragments in seconds (default: `1.5`)
- `--space-gap` — maximum distance between fragments in pixels (default: `40`)
- `--size-sim` — maximum difference in object size (default: `0.2` = 20%)

---

### Track Classification Logic (`classify_tracks`)

The `classify_tracks` function from the `utils.transformer` module classifies objects based on geometric and temporal metrics.

#### Categories:

1. **Ghost** — technical noise
   - Very short tracks (< 30 frames)
   - Almost zero path length (< 10% ROI)
   - Reason: detector errors, glares, shadows

2. **Static** — static objects
   - Very low movement speed (`movement_efficiency < 0.0015`)
   - Or long track with short traveled path (> 200 frames, but < 30% ROI)
   - Reason: parked cars, traffic jams, objects at the beginning/end of ROI

3. **Perfect** — ideal passages
   - Full path through ROI (> 95%)
   - Stable width and height (CV < 0.45)
   - Normal movement speed
   - These are target objects for analysis

4. **Flickering** — unstable objects
   - Strong width jumps (CV ≥ 0.45)
   - Reason: object overlaps, tracker issues

5. **Partial** — stable fragments
   - Partial path (30-95% ROI)
   - Stable dimensions (CV < 0.45)
   - Normal speed
   - These are candidates for merging into full tracks

6. **Noise** — other noise
   - All objects that didn't fall into previous categories
   - Potential candidates for merging

**Key metrics:**
- `path_completeness` — fraction of traveled path through ROI (0.0 - 1.0)
- `movement_efficiency` — movement speed (path per frame)
- `w_cv`, `h_cv` — coefficient of variation for width/height (dimension stability)
- `frames_count` — number of frames the object was tracked
- `y_start`, `y_end` — vertical position at first and last frame
- `w_mean`, `w_std` — average width and standard deviation
- `h_mean`, `h_std` — average height and standard deviation
- `w_start`, `w_end` — width at first and last frame (transition points)
- `h_start`, `h_end` — height at first and last frame (transition points)
- `x_mean`, `x_std` — average horizontal position and standard deviation
- `t_start`, `t_end` — timestamp of first and last appearance

---

### Fragment Merging Logic (`find_merging_pairs`)

The `find_merging_pairs` function finds fragments of the same object and merges them into a unified track.

#### Conditions for merging two fragments A → B:

1. **Temporal proximity**
   - Fragment B should appear shortly after A ends
   - Time gap: `-1.0` ≤ `gap_time` ≤ `time_gap_limit` (default 1.5 sec)
   - Negative gap (-1.0) allows slight overlap

2. **Same movement direction**
   - Both fragments must move in the same direction (up or down)
   - `direction_a * direction_b > 0`

3. **Same traffic lane**
   - Left lane (moving down): `x + width ≤ 140`
   - Right lane (moving up): `x + width > 140`
   - Both fragments must be in the same lane

4. **Spatial proximity**
   - Vertical distance (main axis): `dist_y < space_gap_limit` (default 40 pixels)
   - Horizontal distance: `dist_x < 20` pixels

5. **Size similarity**
   - Average width: `width_diff < size_sim_limit` (default 20%)
   - Width at transition point: `transition_width_diff < 20%`
   - Height at transition point: `transition_height_diff < 20%`
   - This helps distinguish one object from two consecutive ones

#### Building chains:
After finding pairs (A→B, B→C, C→D), the `build_merge_chains` function builds chains: `[A, B, C, D]`

#### Consolidation:
The `apply_merges_to_summary` function:
- Merges all fragments of a chain into one record with **Merged** category
- Assigns `unified_id` in chronological order
- Recalculates metrics for the merged track
- Discards merges with `path_completeness ≤ 0.3` (splits back into fragments)

---

### Processing Results

After processing, we get `data/processed_traffic_data.csv` with columns:
- `unified_id` — unique ID for analysis (only for Perfect, Partial, Merged)
- `vehicle_id` — list of original IDs (for Merged this is a list of merged fragments)
- `category` — final object category
- Metrics: `path_completeness`, `w_cv`, `h_cv`, `frames_count`, etc.

**Objects for final analysis:** only those with `unified_id` (Perfect, Partial, Merged with sufficient `path_completeness`)

