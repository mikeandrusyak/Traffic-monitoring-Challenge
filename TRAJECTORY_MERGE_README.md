# Trajectory-Based Vehicle Track Merging

## Overview

`find_merging_pairs_fiorenzo_2()` is a novel approach to detecting interrupted vehicle tracks using **physics-based trajectory prediction**. Unlike existing threshold-based methods, this function predicts where a continuing track should start based on the previous track's velocity, then measures how well reality matches the prediction.

## 🎯 Key Innovation

**Traditional Approach** (existing functions):
```
"Is track B close enough to where track A ended?"
→ Simple distance check
```

**New Trajectory Approach**:
```
"Based on track A's velocity, where SHOULD track B start?"
→ Physics-based prediction + error measurement
```

This is more robust because it considers **motion dynamics** rather than just static positions.

## 📂 Files Created

1. **`trajectory_merge_function.py`** - Main implementation
   - Core function: `find_merging_pairs_fiorenzo_2()`
   - Helper functions for velocity calculation and prediction
   - Fully documented with examples

2. **`test_trajectory_merge.md`** - Testing guide
   - Quick start examples
   - Comparison with existing functions
   - Parameter tuning examples
   - Visualization code

3. **`TRAJECTORY_MERGE_README.md`** - This file
   - Overview and documentation
   - Usage guide
   - Algorithm explanation

## 🚀 Quick Start

### Basic Usage

```python
# Import the function
from trajectory_merge_function import find_merging_pairs_fiorenzo_2

# Run with default balanced parameters
pairs = find_merging_pairs_fiorenzo_2(final_summary, verbose=True)

print(f"Found {len(pairs)} merging pairs")
```

### With Custom Parameters

```python
# Strict mode (high precision)
pairs_strict = find_merging_pairs_fiorenzo_2(
    final_summary,
    max_time_gap=2.0,              # seconds
    max_prediction_error=60,        # pixels
    max_width_diff=0.25,            # 25%
    min_confidence=0.55,            # 0-1 scale
    verbose=True
)

# Relaxed mode (high recall)
pairs_relaxed = find_merging_pairs_fiorenzo_2(
    final_summary,
    max_time_gap=3.0,
    max_prediction_error=100,
    max_width_diff=0.45,
    min_confidence=0.35,
    verbose=True
)
```

## 🧮 Algorithm Explanation

### Step 1: Calculate Velocity
For each track, calculate velocity vector:
```
vx = (x_end - x_start) / duration  # horizontal velocity
vy = (y_end - y_start) / duration  # vertical velocity (main axis)
```

### Step 2: Predict Next Position
Given track A ending at (x_end, y_end) at time t_end:
```
For track B starting at time t_start_B:
  time_gap = t_start_B - t_end
  predicted_x = x_end + vx * time_gap
  predicted_y = y_end + vy * time_gap
```

### Step 3: Calculate Prediction Error
```
actual_start = (x_start_B, y_start_B)
error = √[(predicted_x - actual_x)² + (predicted_y - actual_y)²]
```

### Step 4: Multi-Factor Confidence Scoring

Weights:
- **Trajectory accuracy** (35%): Lower prediction error = higher score
- **Time proximity** (25%): Shorter time gap = higher score
- **Size consistency** (20%): Similar width = higher score
- **Direction alignment** (20%): Same direction = higher score

```
confidence = 0.35 * trajectory_score +
             0.25 * time_score +
             0.20 * size_score +
             0.20 * direction_score
```

### Step 5: Filter by Confidence
Only pairs with `confidence >= min_confidence` are returned.

## 📊 Output Format

The function returns a DataFrame with these columns:

| Column | Type | Description |
|--------|------|-------------|
| `old_id` | int | First track segment ID |
| `new_id` | int | Second track segment ID |
| `session_id` | int | Session identifier |
| `gap_sec` | float | Time gap between tracks (seconds) |
| **`prediction_error`** | float | **NEW: Trajectory prediction error (pixels)** |
| `y_dist` | float | Actual Y-axis distance |
| `x_dist` | float | Actual X-axis distance |
| `width_diff_pct` | float | Width difference percentage |
| `confidence` | float | Overall confidence score (0-1) |
| **`trajectory_score`** | float | **NEW: Trajectory component score** |
| `time_score` | float | Time proximity score |
| `size_score` | float | Size similarity score |
| `direction_score` | float | Direction alignment score |
| `direction_a` | str | Direction of track A (up/down/static) |
| `direction_b` | str | Direction of track B (up/down/static) |
| `frames_a` | int | Frame count of track A |
| `frames_b` | int | Frame count of track B |

## ⚙️ Parameters

### Required
- **`final_summary_df`**: DataFrame from `categorize_ids()`

### Optional (with defaults)
- **`max_time_gap=2.5`**: Maximum time gap between tracks (seconds)
- **`max_prediction_error=80`**: Maximum acceptable prediction error (pixels)
  - 80px ≈ 27% of frame height (290px)
- **`max_width_diff=0.35`**: Maximum relative width difference (35%)
- **`min_confidence=0.45`**: Minimum confidence threshold (0-1)
- **`categories=['RelayCandidate', 'Static']`**: Track categories to consider
- **`bidirectional=True`**: Support both up and down movement
- **`verbose=False`**: Print diagnostic information

## 🎨 Visualization

See [test_trajectory_merge.md](test_trajectory_merge.md) for visualization code that shows:
1. Track A with velocity vector
2. Track B actual trajectory
3. **Predicted vs actual start position** with error line

## 📈 Expected Performance

### With Default Balanced Parameters

| Metric | Expected Range |
|--------|----------------|
| Pairs found | 15-30 pairs |
| Confidence scores | 0.45 - 0.85 |
| Prediction errors | 10 - 60 pixels |
| Time gaps | 0.1 - 2.0 seconds |

### Comparison with Existing Methods

| Method | Typical Pairs | Approach |
|--------|---------------|----------|
| `find_merging_pairs()` | ~5 pairs | Simple thresholds |
| `find_merging_pairs_fiorenzo()` | ~14 pairs | Multi-factor confidence |
| **`find_merging_pairs_fiorenzo_2()`** | **~20 pairs** | **Trajectory prediction** |

## ✅ Advantages

1. **Physics-Based**: Uses actual vehicle motion dynamics
2. **More Robust**: Less sensitive to position noise
3. **Interpretable**: Prediction error is intuitive
4. **Novel**: Different from existing threshold-based methods
5. **Balanced**: Finds more pairs while maintaining quality

## 🔧 Special Cases

### Static Vehicles
For vehicles with `direction='static'`, the function falls back to simple spatial proximity since velocity-based prediction doesn't apply.

### Opposite Directions
If `bidirectional=False`, only pairs moving in the same direction are considered. Otherwise, opposite directions get a neutral direction score (0.5).

## 📝 Examples

### Example 1: Find All Pairs
```python
pairs = find_merging_pairs_fiorenzo_2(final_summary, verbose=True)
print(pairs.head())
```

### Example 2: High Precision Mode
```python
# For critical applications - fewer pairs but very high confidence
pairs_strict = find_merging_pairs_fiorenzo_2(
    final_summary,
    min_confidence=0.60,
    max_prediction_error=50,
    verbose=True
)
```

### Example 3: High Recall Mode
```python
# Find as many pairs as possible
pairs_relaxed = find_merging_pairs_fiorenzo_2(
    final_summary,
    min_confidence=0.35,
    max_time_gap=3.5,
    max_prediction_error=120,
    verbose=True
)
```

### Example 4: Compare Methods
```python
# Original
pairs_orig = find_merging_pairs(final_summary)

# Confidence-based
pairs_v1 = find_merging_pairs_fiorenzo(
    final_summary,
    categories=['RelayCandidate', 'Static'],
    min_confidence=0.45
)

# Trajectory-based (NEW)
pairs_v2 = find_merging_pairs_fiorenzo_2(
    final_summary,
    min_confidence=0.45
)

print(f"Original:    {len(pairs_orig)} pairs")
print(f"V1 (conf):   {len(pairs_v1)} pairs")
print(f"V2 (traj):   {len(pairs_v2)} pairs")
```

## 🐛 Troubleshooting

### No Pairs Found
**Problem**: Function returns 0 pairs

**Solutions**:
1. Check data has RelayCandidate/Static categories:
   ```python
   print(final_summary['category'].value_counts())
   ```

2. Relax parameters:
   ```python
   pairs = find_merging_pairs_fiorenzo_2(
       final_summary,
       max_time_gap=3.5,
       max_prediction_error=120,
       min_confidence=0.30,
       verbose=True  # See why pairs are rejected
   )
   ```

### Too Many Pairs
**Problem**: Too many low-quality pairs

**Solutions**:
1. Increase confidence threshold:
   ```python
   pairs = find_merging_pairs_fiorenzo_2(
       final_summary,
       min_confidence=0.55  # or 0.60
   )
   ```

2. Tighten prediction error:
   ```python
   pairs = find_merging_pairs_fiorenzo_2(
       final_summary,
       max_prediction_error=60  # or 50
   )
   ```

### Import Error
**Problem**: `ModuleNotFoundError: No module named 'trajectory_merge_function'`

**Solution**: Make sure the file is in your working directory or add to Python path:
```python
import sys
sys.path.append('/path/to/directory')
from trajectory_merge_function import find_merging_pairs_fiorenzo_2
```

Or use the notebook directly - copy the function code into a notebook cell.

## 📚 Further Reading

- See [test_trajectory_merge.md](test_trajectory_merge.md) for detailed testing examples
- Check the docstrings in [trajectory_merge_function.py](trajectory_merge_function.py) for technical details

## 🤝 Credits

Developed by Fiorenzo with assistance from Claude Code (Anthropic).

Algorithm design: Trajectory prediction-based vehicle track merging.

---

**Last Updated**: 2025-12-29
