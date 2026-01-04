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

## Traffic Tracking logic


### 🕹️ Usage
To start the monitoring system, you first need to activate the virtual environment to ensure the script uses the correct installed libraries. Then, run the main camera script:

Bash
#### 1. Activate the virtual environment
```source .venv/bin/activate```

#### 2. Run the main tracking application
```python main_cam_2.py```
Controls:

The system will open windows showing the Mask, Merged Detection, and Tracker.

Press **Esc** to exit the application.


### Detection Pipeline

The main execution loop processes frames in a 11-step pipeline designed to handle varying lighting conditions and object sizes.

#### 1. Image Capture & Preprocessing

* **Source**: Captures `YUV420` lores frames (640x480) via `Picamera2`.
* **ROI**: Crops the image to the Region of Interest defined by `ROI_TOP`, `ROI_BOTTOM`, `ROI_LEFT`, and `ROI_RIGHT` to exclude irrelevant background.
* **Enhancement**: Applies **CLAHE** (Contrast Limited Adaptive Histogram Equalization) with `clipLimit=2.0` to improve contrast before detection.

#### 2. Adaptive Background Subtraction (Day/Night)

The system automatically switches between two Gaussian Mixture-based Background/Foreground Segmentation models (`MOG2`) based on frame brightness:

* **Day Mode** (`brightness > 80`):
* `history=600`, `varThreshold=32`
* Optimized for faster adaptation to lighting changes.


* **Night Mode** (`brightness <= 80`):
* `history=1500`, `varThreshold=18`
* Higher sensitivity and longer history to detect fainter objects in low light.



#### 3. Noise Reduction & Merging

* **Morphology**: Applies `MORPH_OPEN` and `MORPH_CLOSE` with a small elliptical kernel to remove pixel noise.
* **Object Merging**: A crucial step to prevent large vehicles (like buses) from being split into multiple IDs. It draws a rectangle over detected contours and applies a larger rectangular kernel (`20x20`) to fuse adjacent components.

#### 4. Filtering & Logging

* **Area Thresholds**: Only objects with area between `AREA_MIN` (800) and `AREA_MAX` (40000) are passed to the tracker.
* **Async Database Write**: The system implements a Non-blocking Producer-Consumer pattern with Batching to handle high-frequency data insertion into Google Cloud SQL without affecting the video processing frame rate.

#### 4.1. The Mechanism

Main Thread (Producer): Analyzes frames ~30 times per second. When an object is detected, it instantly pushes the data package into a thread-safe `FIFO Queue`. This operation is near-instantaneous (~O(1)).

Daemon Thread (Consumer): A background thread (`db_writer_thread`) constantly monitors the queue. It implements a Batching Strategy to minimize network round-trips:

Batch Size: Accumulates up to 200 records.

Time Timeout: Flushes data every 0.2 seconds if the batch isn't full.

Efficient Insert: Instead of executing hundreds of individual SQL queries, the system constructs a single Multi-Row INSERT statement, significantly reducing I/O overhead.

#### 4.2. Tech Stack & Connection

Database: PostgreSQL hosted on Google Cloud SQL.

Driver: Uses `pg8000` combined with the `google-cloud-sql-connector` for secure, certificate-based authentication without managing static SSL keys.

Credentials: Connection strings and secrets are loaded safely via environment variables (`.env`).

#### 4.3. Stored Data Schema

Data is normalized and stored in the traffic_data table.

* `vehicle_id`: Unique ID assigned by the tracker
* `frame_id`: Sequential counter to sync video with data
* `date_time`:	Precise datetime of the detection
* `x, y`: Top-left coordinates of the bounding box
* `width, heigth`: Dimensions of the object
* `area`: Calculated pixel area (used for filtering)

**Key Configuration Parameters:**

* `distance_per_frame`: **35 px** (Max travel distance between frames)
* `history`: **600** (Day) / **1500** (Night) (Frames used for background modeling)
* `varThreshold`: **32** (Day) / **18** (Night) (Sensitivity)

### Tracker Logic (`Tracker2`)

The project utilizes a custom `Tracker2` class based on Euclidean distance centroid tracking.

**How it works:**

1. **Centroid Calculation**: For every detected bounding box in a frame, the center point  is calculated.
2. **Distance Comparison**: The system calculates the Euclidean distance between the center points of new objects and existing objects from the previous frame.
3. **ID Assignment**:
* If the distance is less than the `distance_per_frame` threshold (set to **35 pixels**), the object is considered the same, and the ID is maintained.
* If no existing object is found within the threshold, a new ID is assigned (`id_count` increments).


4. **Cleanup**: IDs that are no longer detected are removed from the dictionary to keep memory usage low.

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

---

## Data Analysis

### Goal
The data analysis stage transforms processed traffic data into actionable insights by computing derived metrics, classifying vehicles, and generating visualizations for traffic monitoring.

### Analysis Notebook
The **`notebooks/summary_data_analysis.ipynb`** notebook performs the complete analysis pipeline:
- Loads processed data from `data/processed_traffic_data.csv` or database
- Filters data by date range (e.g., December 2025)
- Computes derived metrics and classifications
- Generates visualization plots

### Feature Engineering (`utils/new_columns_fiorenzo.py`)

#### Speed and Direction Calculation
The `add_speed_direction_to_summary` function computes:
- `velocity_x_px_seconds`, `velocity_y_px_seconds` — velocity in pixels/second
- `velocity_x_km_h`, `velocity_y_km_h` — velocity in km/h (using pixel-to-meter ratio)
- `direction` — movement direction: `up`, `down`, or `static`

**Pixel-to-meter conversion:**
- Total ROI height: 300 pixels = 25 meters
- Ratio: 12 pixels/meter

#### Day/Night Classification
The `add_day_night_to_summary` function determines `solar_phase` (Day/Night) based on:
- Sunset times from `utils/sunset.csv` (source: timeanddate.com)
- Compares vehicle timestamp against sunset time for that date

#### Vehicle Type Classification
The `classify_vehicle_types` function classifies vehicles as **Car** or **Truck** using different strategies:

**Day classification** (height-based):
- `h_mean < 120 pixels` → Car
- `h_mean >= 120 pixels` → Truck

**Night classification** (multi-criteria OR logic):
- `h_mean >= 120 pixels` → Truck
- `w_mean >= 110 pixels` → Truck
- `duration >= threshold` → Truck (threshold calculated from day data)
- Otherwise → Car

#### Additional Computed Metrics
- `duration` — track duration in seconds
- `size_mean` — area (width × height in pixels²)
- `h_w_mean_ratio` — aspect ratio (height/width)
- `h_mean_meters`, `w_mean_meters`, `size_mean_meters` — dimensions in meters

---

### Visualization (`utils/plots_fiorenzo.py`)

#### Classification Visualization
`visualize_classification` — 3×3 scatter matrix showing height, width, and duration by vehicle class (Car/Truck). Supports filtering by solar phase.

#### Vehicle Count Histogram
`vehicle_count_over_time_histogram` — 2×2 matrix of horizontal bar charts showing daily vehicle counts:
- Rows: Day / Night
- Columns: Car / Truck
- Includes gap indicators for periods with script errors

#### Speed Analysis
`average_speed_by_weekday_and_hour` — Grid of 7 line plots (one per weekday) showing average speed by hour. Includes speed limit reference line and highlights speeding zones.

#### Speeding Analysis
`speeding_vehicles_histogram` — Three separate horizontal bar charts showing daily counts of:
- Speeding vehicles (> speed limit)
- Slow vehicles (10-20 km/h)
- Very slow vehicles (< 10 km/h)

#### Data Gap Handling
All plots include visual indicators (hatched shading) for periods affected by data collection interruptions due to script errors.

---

### Analysis Results

After analysis, the enriched dataset contains additional columns:
- `velocity_y_km_h` — vertical speed in km/h
- `direction` — movement direction (up/down/static)
- `solar_phase` — Day or Night
- `Class` — vehicle type (Car/Truck)
- Dimension metrics in both pixels and meters

**Output visualizations** are saved to the `plots/` directory:
- `classification_visualization.png`
- `vehicle_count_over_time.png`
- `avg_speed_by_weekday_hour.png`
- `speeding_vehicles.png`, `slow_vehicles.png`, `very_slow_vehicles.png`

---
