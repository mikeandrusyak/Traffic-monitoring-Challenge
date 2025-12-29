# Testing find_merging_pairs_fiorenzo_2()

## Quick Start

Add this cell to your Jupyter notebook after the existing merge functions:

```python
# Load the trajectory-based merge function
from trajectory_merge_function import find_merging_pairs_fiorenzo_2

# Test with default balanced parameters
print("="*70)
print("TESTING TRAJECTORY PREDICTION-BASED MERGE DETECTION")
print("="*70)

pairs_traj = find_merging_pairs_fiorenzo_2(
    final_summary,
    verbose=True
)

print(f"\n✅ Found {len(pairs_traj)} pairs using trajectory prediction!")
```

## Comparison with Existing Functions

```python
# Compare all three functions
print("\n" + "="*70)
print("COMPARISON: All Three Functions")
print("="*70)

# Original function
pairs_original = find_merging_pairs(final_summary)
print(f"Original function:     {len(pairs_original):2d} pairs")

# First improvement (confidence-based)
pairs_fiorenzo_1 = find_merging_pairs_fiorenzo(
    final_summary,
    categories=['RelayCandidate', 'Static'],
    min_confidence=0.45,  # Match new function's default
    verbose=False
)
print(f"Fiorenzo v1 (conf):    {len(pairs_fiorenzo_1):2d} pairs")

# New trajectory-based function
pairs_fiorenzo_2 = find_merging_pairs_fiorenzo_2(
    final_summary,
    min_confidence=0.45,
    verbose=False
)
print(f"Fiorenzo v2 (traj):    {len(pairs_fiorenzo_2):2d} pairs")

print("\n" + "="*70)
print("Detailed Results Comparison")
print("="*70)

# Show unique pairs found by each method
if len(pairs_fiorenzo_2) > 0:
    print("\n🎯 Top 5 pairs by trajectory method:")
    print(pairs_fiorenzo_2[['old_id', 'new_id', 'confidence', 'prediction_error',
                            'gap_sec', 'direction_a', 'direction_b']].head(5))

    # Find pairs unique to trajectory method
    traj_pairs_set = set(zip(pairs_fiorenzo_2['old_id'], pairs_fiorenzo_2['new_id']))
    fior1_pairs_set = set(zip(pairs_fiorenzo_1['old_id'], pairs_fiorenzo_1['new_id']))

    unique_to_traj = traj_pairs_set - fior1_pairs_set
    unique_to_fior1 = fior1_pairs_set - traj_pairs_set
    common_pairs = traj_pairs_set & fior1_pairs_set

    print(f"\n📊 Overlap Analysis:")
    print(f"  Common to both methods: {len(common_pairs)}")
    print(f"  Unique to trajectory:   {len(unique_to_traj)}")
    print(f"  Unique to fiorenzo v1:  {len(unique_to_fior1)}")

    if unique_to_traj:
        print(f"\n  💡 Pairs found ONLY by trajectory method:")
        for old_id, new_id in list(unique_to_traj)[:3]:
            row = pairs_fiorenzo_2[(pairs_fiorenzo_2['old_id']==old_id) &
                                    (pairs_fiorenzo_2['new_id']==new_id)].iloc[0]
            print(f"     {old_id:4d} → {new_id:4d}: conf={row['confidence']:.3f}, "
                  f"pred_err={row['prediction_error']:.1f}px, gap={row['gap_sec']:.2f}s")
```

## Parameter Tuning Examples

```python
# Test different strictness levels
print("\n" + "="*70)
print("PARAMETER TUNING")
print("="*70)

configs = [
    ("Strict (high precision)", {
        'max_time_gap': 2.0,
        'max_prediction_error': 60,
        'max_width_diff': 0.25,
        'min_confidence': 0.55
    }),
    ("Balanced (default)", {
        'max_time_gap': 2.5,
        'max_prediction_error': 80,
        'max_width_diff': 0.35,
        'min_confidence': 0.45
    }),
    ("Relaxed (high recall)", {
        'max_time_gap': 3.0,
        'max_prediction_error': 100,
        'max_width_diff': 0.45,
        'min_confidence': 0.35
    })
]

for name, params in configs:
    result = find_merging_pairs_fiorenzo_2(final_summary, **params, verbose=False)
    print(f"{name:25s}: {len(result):2d} pairs")
```

## Visualize Trajectory Predictions

```python
# Visualize a pair with trajectory prediction overlay
def visualize_trajectory_prediction(session_df, final_summary, merge_row, figsize=(16, 6)):
    """
    Enhanced visualization showing trajectory prediction.
    """
    import matplotlib.pyplot as plt

    old_id = merge_row['old_id']
    new_id = merge_row['new_id']

    # Get track data
    old_track = session_df[session_df['vehicle_id'] == old_id].sort_values('date_time')
    new_track = session_df[session_df['vehicle_id'] == new_id].sort_values('date_time')

    # Get summary data for velocity calculation
    track_a_summary = final_summary[final_summary['vehicle_id'] == old_id].iloc[0]

    # Calculate velocity
    from trajectory_merge_function import _calculate_velocity, _predict_position
    vx, vy, direction = _calculate_velocity(track_a_summary)

    # Predict position
    gap_time = merge_row['gap_sec']
    track_a_dict = track_a_summary.to_dict()
    track_a_dict['_vx'] = vx
    track_a_dict['_vy'] = vy
    predicted_x, predicted_y = _predict_position(track_a_dict, gap_time)

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    ROI_HEIGHT = 290
    ROI_WIDTH = 200

    # Plot 1: Track A with velocity vector
    ax1 = axes[0]
    ax1.set_xlim(0, ROI_WIDTH)
    ax1.set_ylim(0, ROI_HEIGHT)
    ax1.invert_yaxis()
    ax1.set_title(f'Track A (ID: {old_id})\nVelocity: vy={vy:.1f} px/s', fontweight='bold')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    ax1.grid(True, alpha=0.3)

    ax1.plot(old_track['x'], old_track['y'], 'o-', color='blue', alpha=0.6, linewidth=2)
    ax1.scatter(old_track['x'].iloc[-1], old_track['y'].iloc[-1],
               color='red', s=200, marker='X', label='End', zorder=5)

    # Draw velocity vector
    scale = 20  # Scale factor for visualization
    ax1.arrow(old_track['x'].iloc[-1], old_track['y'].iloc[-1],
             0, vy * scale,
             head_width=10, head_length=10, fc='orange', ec='orange',
             label=f'Velocity', linewidth=2)
    ax1.legend()

    # Plot 2: Track B
    ax2 = axes[1]
    ax2.set_xlim(0, ROI_WIDTH)
    ax2.set_ylim(0, ROI_HEIGHT)
    ax2.invert_yaxis()
    ax2.set_title(f'Track B (ID: {new_id})\nActual Start', fontweight='bold')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    ax2.grid(True, alpha=0.3)

    ax2.plot(new_track['x'], new_track['y'], 'o-', color='darkgreen', alpha=0.6, linewidth=2)
    ax2.scatter(new_track['x'].iloc[0], new_track['y'].iloc[0],
               color='green', s=200, marker='o', label='Start', zorder=5)
    ax2.legend()

    # Plot 3: Prediction overlay
    ax3 = axes[2]
    ax3.set_xlim(0, ROI_WIDTH)
    ax3.set_ylim(0, ROI_HEIGHT)
    ax3.invert_yaxis()
    ax3.set_title(f'Trajectory Prediction\nError: {merge_row["prediction_error"]:.1f}px',
                  fontweight='bold')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    ax3.grid(True, alpha=0.3)

    # Plot both tracks
    ax3.plot(old_track['x'], old_track['y'], 'o-', color='blue', alpha=0.5,
             linewidth=2, label='Track A')
    ax3.plot(new_track['x'], new_track['y'], 'o-', color='darkgreen', alpha=0.5,
             linewidth=2, label='Track B')

    # Show predicted vs actual start position
    ax3.scatter(predicted_x, predicted_y, color='red', s=250, marker='*',
               label='Predicted start', zorder=6, edgecolors='darkred', linewidths=2)
    ax3.scatter(new_track['x'].iloc[0], new_track['y'].iloc[0],
               color='green', s=200, marker='o', label='Actual start', zorder=5)

    # Draw error line
    ax3.plot([predicted_x, new_track['x'].iloc[0]],
            [predicted_y, new_track['y'].iloc[0]],
            'r--', linewidth=2, label='Prediction error', alpha=0.7)

    ax3.legend()

    # Metadata
    metadata = f"""
Trajectory Method
-----------------
Confidence: {merge_row['confidence']:.3f}
Pred Error: {merge_row['prediction_error']:.1f}px

Traj Score: {merge_row['trajectory_score']:.3f}
Time Score: {merge_row['time_score']:.3f}
Size Score: {merge_row['size_score']:.3f}
Dir Score:  {merge_row['direction_score']:.3f}

Time gap:   {merge_row['gap_sec']:.2f}s
Y-distance: {merge_row['y_dist']:.1f}px
Width diff: {merge_row['width_diff_pct']:.1f}%
Direction:  {merge_row['direction_a']} → {merge_row['direction_b']}
    """.strip()

    fig.text(0.99, 0.5, metadata, ha='right', va='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
             fontfamily='monospace', fontsize=9)

    plt.tight_layout()
    plt.show()

# Visualize top 3 pairs
if len(pairs_fiorenzo_2) > 0:
    print("\n" + "="*70)
    print("VISUALIZING TOP 3 PAIRS WITH TRAJECTORY PREDICTION")
    print("="*70)

    for idx in range(min(3, len(pairs_fiorenzo_2))):
        print(f"\nPair {idx+1}: {pairs_fiorenzo_2.iloc[idx]['old_id']} → "
              f"{pairs_fiorenzo_2.iloc[idx]['new_id']}")
        visualize_trajectory_prediction(session_df, final_summary, pairs_fiorenzo_2.iloc[idx])
```

## Expected Output Summary

With default balanced parameters, you should see approximately **15-25 pairs** depending on your dataset.

Key advantages over existing methods:
- ✅ Physics-based trajectory prediction
- ✅ More robust to position noise
- ✅ Interpretable prediction error metric
- ✅ Better handling of moving vs static vehicles
- ✅ Separate scoring for trajectory, time, size, and direction

## Troubleshooting

If you get zero pairs:
1. Check that your data has RelayCandidate or Static categories
2. Try relaxing parameters (see "Relaxed" config above)
3. Run with `verbose=True` to see diagnostic information

If you get too many pairs:
1. Increase `min_confidence` (try 0.55 or 0.60)
2. Decrease `max_prediction_error` (try 60 or 50)
3. Decrease `max_time_gap` (try 2.0)
