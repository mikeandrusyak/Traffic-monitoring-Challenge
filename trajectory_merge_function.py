import pandas as pd
import numpy as np

# ============================================================================
# TRAJECTORY PREDICTION-BASED MERGE DETECTION
# ============================================================================
# Author: Fiorenzo (with Claude Code assistance)
# Novel approach: Uses physics-based trajectory prediction instead of
# simple threshold matching
# ============================================================================

def _calculate_velocity(track_row):
    """
    Calculate velocity vector and direction from track data.

    Parameters
    ----------
    track_row : dict or pd.Series
        Track data with y_start, y_end, x_mean (or x_start/x_end), t_start, t_end

    Returns
    -------
    tuple
        (vx, vy, direction) where:
        - vx: horizontal velocity (pixels/second)
        - vy: vertical velocity (pixels/second)
        - direction: 'down', 'up', or 'static'
    """
    # Calculate time duration
    duration = (track_row['t_end'] - track_row['t_start']).total_seconds()

    if duration <= 0:
        return (0.0, 0.0, 'static')

    # Calculate Y velocity (main movement axis)
    dy = track_row['y_end'] - track_row['y_start']
    vy = dy / duration

    # Calculate X velocity (use x_std as proxy for lateral movement)
    # For tracks with little lateral movement, x_std will be small
    vx = track_row.get('x_std', 0) / duration if duration > 0 else 0

    # Determine direction
    if abs(vy) < 1.0:  # Less than 1 pixel/second ~ static
        direction = 'static'
    elif vy > 0:
        direction = 'down'
    else:
        direction = 'up'

    return (vx, vy, direction)


def _predict_position(track_a, time_gap):
    """
    Predict where track B should start based on track A's trajectory.

    Parameters
    ----------
    track_a : dict
        Track A data with velocities calculated
    time_gap : float
        Time gap in seconds between end of A and start of B

    Returns
    -------
    tuple
        (x_predicted, y_predicted) - predicted start position of track B
    """
    # Get track A's endpoint
    x_end = track_a['x_mean']  # Use mean X as endpoint approximation
    y_end = track_a['y_end']

    # Get velocities
    vx = track_a.get('_vx', 0)
    vy = track_a.get('_vy', 0)

    # Predict position after time_gap using linear extrapolation
    x_predicted = x_end + (vx * time_gap)
    y_predicted = y_end + (vy * time_gap)

    return (x_predicted, y_predicted)


def _calculate_prediction_error(predicted_pos, actual_pos):
    """
    Calculate Euclidean distance between predicted and actual positions.

    Parameters
    ----------
    predicted_pos : tuple
        (x_predicted, y_predicted)
    actual_pos : tuple
        (x_actual, y_actual)

    Returns
    -------
    float
        Euclidean distance in pixels
    """
    dx = predicted_pos[0] - actual_pos[0]
    dy = predicted_pos[1] - actual_pos[1]

    return np.sqrt(dx**2 + dy**2)


def _calculate_trajectory_confidence(track_a, track_b, time_gap,
                                    max_prediction_error, max_time_gap, max_width_diff):
    """
    Calculate multi-factor confidence score emphasizing trajectory prediction.

    Scoring weights:
    - Trajectory accuracy: 35% (NEW - physics-based prediction)
    - Time proximity: 25%
    - Size consistency: 20%
    - Direction alignment: 20%

    Parameters
    ----------
    track_a, track_b : dict
        Track data dictionaries
    time_gap : float
        Time gap in seconds
    max_prediction_error : float
        Maximum acceptable prediction error (pixels)
    max_time_gap : float
        Maximum time gap (seconds)
    max_width_diff : float
        Maximum relative width difference (0-1)

    Returns
    -------
    dict
        Dictionary with 'confidence' and individual component scores
    """
    # 1. TRAJECTORY SCORE (35%) - NEW APPROACH
    # For static tracks, fall back to simple spatial proximity
    if track_a['_direction'] == 'static' or track_b['_direction'] == 'static':
        # Simple spatial distance for static objects
        dist_y = abs(track_b['y_start'] - track_a['y_end'])
        dist_x = abs(track_b['x_mean'] - track_a['x_mean'])
        spatial_error = np.sqrt(dist_y**2 + dist_x**2)
        trajectory_score = max(0, 1 - (spatial_error / max_prediction_error))
    else:
        # Predict where B should start based on A's velocity
        predicted_pos = _predict_position(track_a, time_gap)
        actual_pos = (track_b['x_mean'], track_b['y_start'])

        prediction_error = _calculate_prediction_error(predicted_pos, actual_pos)
        trajectory_score = max(0, 1 - (prediction_error / max_prediction_error))

    # 2. TIME PROXIMITY SCORE (25%)
    time_score = max(0, 1 - (time_gap / max_time_gap)) if max_time_gap > 0 else 0

    # 3. SIZE CONSISTENCY SCORE (20%)
    if track_a['w_mean'] > 0:
        width_diff = abs(track_a['w_mean'] - track_b['w_mean']) / track_a['w_mean']
        size_score = max(0, 1 - (width_diff / max_width_diff)) if max_width_diff > 0 else 0
    else:
        size_score = 0

    # 4. DIRECTION ALIGNMENT SCORE (20%)
    dir_a = track_a['_direction']
    dir_b = track_b['_direction']

    if dir_a == dir_b:
        direction_score = 1.0  # Perfect alignment
    elif dir_a == 'static' or dir_b == 'static':
        direction_score = 0.5  # Neutral (one is static)
    else:
        direction_score = 0.0  # Opposite directions

    # Weighted combination
    confidence = (
        trajectory_score * 0.35 +
        time_score * 0.25 +
        size_score * 0.20 +
        direction_score * 0.20
    )

    return {
        'confidence': confidence,
        'trajectory_score': trajectory_score,
        'time_score': time_score,
        'size_score': size_score,
        'direction_score': direction_score
    }


def find_merging_pairs_fiorenzo_2(
    final_summary_df,
    max_time_gap=2.5,
    max_prediction_error=80,
    max_width_diff=0.35,
    min_confidence=0.45,
    categories=None,
    bidirectional=True,
    verbose=False
):
    """
    Find vehicle track merging pairs using trajectory prediction.

    NOVEL APPROACH: This function predicts where track B should start based on
    track A's velocity vector, then measures how well the prediction matches reality.
    This physics-based approach is more robust than simple distance thresholds.

    How it works:
    1. Calculate velocity vector (vx, vy) for each track
    2. For each candidate pair (A→B), predict B's start position using A's velocity
    3. Measure prediction error (Euclidean distance)
    4. Score based on prediction accuracy + time proximity + size + direction
    5. Return pairs above confidence threshold

    Parameters
    ----------
    final_summary_df : pd.DataFrame
        Vehicle tracking summary with aggregated metrics from categorize_ids()
    max_time_gap : float, default=2.5
        Maximum time gap between track end and track start (seconds)
    max_prediction_error : float, default=80
        Maximum trajectory prediction error tolerance (pixels)
        (~27% of 290px frame height)
    max_width_diff : float, default=0.35
        Maximum relative width difference (0.35 = 35%)
    min_confidence : float, default=0.45
        Minimum confidence threshold to include pair (0-1 scale)
    categories : list, optional
        Track categories to consider. Defaults to ['RelayCandidate', 'Static']
    bidirectional : bool, default=True
        Support vehicles moving both up and down
    verbose : bool, default=False
        Print detailed diagnostic information

    Returns
    -------
    pd.DataFrame
        Merge pairs with columns:
        - old_id, new_id, session_id: Track identifiers
        - gap_sec: Time gap between tracks
        - prediction_error: Trajectory prediction error (pixels) - NEW
        - y_dist, x_dist: Actual spatial distances
        - width_diff_pct: Width difference percentage
        - confidence: Overall confidence score (0-1)
        - trajectory_score: Trajectory prediction component - NEW
        - time_score, size_score, direction_score: Other components
        - direction_a, direction_b: Movement directions
        - frames_a, frames_b: Frame counts

    Examples
    --------
    >>> # Basic usage with default parameters
    >>> pairs = find_merging_pairs_fiorenzo_2(final_summary)
    >>> print(f"Found {len(pairs)} pairs")

    >>> # Strict mode (higher precision, fewer pairs)
    >>> pairs_strict = find_merging_pairs_fiorenzo_2(
    ...     final_summary,
    ...     max_prediction_error=50,
    ...     min_confidence=0.60
    ... )

    >>> # Relaxed mode (more pairs, slightly lower precision)
    >>> pairs_relaxed = find_merging_pairs_fiorenzo_2(
    ...     final_summary,
    ...     max_time_gap=3.0,
    ...     max_prediction_error=100,
    ...     min_confidence=0.35
    ... )

    >>> # Verbose mode for debugging
    >>> pairs = find_merging_pairs_fiorenzo_2(final_summary, verbose=True)
    """
    # Input validation
    required_columns = ['session_id', 'vehicle_id', 'category', 't_start', 't_end',
                       'y_start', 'y_end', 'x_mean', 'x_std', 'w_mean', 'frames_count']
    missing_cols = [col for col in required_columns if col not in final_summary_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Set default categories
    if categories is None:
        categories = ['RelayCandidate', 'Static']

    # Filter candidates
    candidates = final_summary_df[final_summary_df['category'].isin(categories)].copy()

    # Calculate velocities for all candidates
    for idx in candidates.index:
        vx, vy, direction = _calculate_velocity(candidates.loc[idx])
        candidates.loc[idx, '_vx'] = vx
        candidates.loc[idx, '_vy'] = vy
        candidates.loc[idx, '_direction'] = direction

    # Sort by session and time
    candidates = candidates.sort_values(['session_id', 't_start']).reset_index(drop=True)
    records = candidates.to_dict('records')

    if verbose:
        print("="*70)
        print("TRAJECTORY PREDICTION-BASED MERGE DETECTION")
        print("="*70)
        print(f"Total tracks in summary: {len(final_summary_df)}")
        print(f"Candidate tracks for merging: {len(candidates)}")
        print(f"Categories considered: {categories}")
        print(f"\nCategory distribution:")
        print(candidates['category'].value_counts())
        print(f"\nDirection distribution:")
        print(candidates['_direction'].value_counts())

    # Find all potential pairs
    all_pairs = []

    for i in range(len(records)):
        track_a = records[i]

        for j in range(i + 1, len(records)):
            track_b = records[j]

            # Must be same session
            if track_a['session_id'] != track_b['session_id']:
                break

            # Calculate time gap
            gap_time = (track_b['t_start'] - track_a['t_end']).total_seconds()

            # B must start after A ends, within time limit
            if not (0 <= gap_time <= max_time_gap):
                if gap_time > max_time_gap:
                    break  # No more candidates in this session
                continue

            # Direction check (if not bidirectional)
            if not bidirectional:
                if track_a['_direction'] != track_b['_direction']:
                    continue

            # Calculate actual distances (for reference)
            dist_y = abs(track_b['y_start'] - track_a['y_end'])
            dist_x = abs(track_b['x_mean'] - track_a['x_mean'])

            # Calculate trajectory-based confidence
            confidence_data = _calculate_trajectory_confidence(
                track_a, track_b, gap_time,
                max_prediction_error, max_time_gap, max_width_diff
            )

            # Only keep pairs above minimum confidence
            if confidence_data['confidence'] < min_confidence:
                continue

            # Calculate prediction error for output
            if track_a['_direction'] != 'static' and track_b['_direction'] != 'static':
                predicted_pos = _predict_position(track_a, gap_time)
                actual_pos = (track_b['x_mean'], track_b['y_start'])
                prediction_error = _calculate_prediction_error(predicted_pos, actual_pos)
            else:
                # For static tracks, use spatial distance as "error"
                prediction_error = np.sqrt(dist_y**2 + dist_x**2)

            # Calculate width difference percentage
            if track_a['w_mean'] > 0:
                width_diff_pct = (abs(track_a['w_mean'] - track_b['w_mean']) / track_a['w_mean']) * 100
            else:
                width_diff_pct = 0

            all_pairs.append({
                'old_id': int(track_a['vehicle_id']),
                'new_id': int(track_b['vehicle_id']),
                'session_id': int(track_a['session_id']),
                'gap_sec': round(gap_time, 3),
                'prediction_error': round(prediction_error, 2),
                'y_dist': round(dist_y, 2),
                'x_dist': round(dist_x, 2),
                'width_diff_pct': round(width_diff_pct, 2),
                'confidence': round(confidence_data['confidence'], 3),
                'trajectory_score': round(confidence_data['trajectory_score'], 3),
                'time_score': round(confidence_data['time_score'], 3),
                'size_score': round(confidence_data['size_score'], 3),
                'direction_score': round(confidence_data['direction_score'], 3),
                'direction_a': track_a['_direction'],
                'direction_b': track_b['_direction'],
                'frames_a': int(track_a['frames_count']),
                'frames_b': int(track_b['frames_count'])
            })

    # Create result DataFrame
    result_df = pd.DataFrame(all_pairs)

    # Sort by confidence (highest first)
    if len(result_df) > 0:
        result_df = result_df.sort_values('confidence', ascending=False).reset_index(drop=True)

    # Verbose output
    if verbose:
        print("\n" + "="*70)
        print("RESULTS")
        print("="*70)
        print(f"Total merge pairs found: {len(result_df)}")

        if len(result_df) > 0:
            print("\n" + "-"*70)
            print("STATISTICS")
            print("-"*70)

            print("\n📊 Confidence scores:")
            print(f"  Range: {result_df['confidence'].min():.3f} - {result_df['confidence'].max():.3f}")
            print(f"  Mean:  {result_df['confidence'].mean():.3f}")
            print(f"  Median: {result_df['confidence'].median():.3f}")

            print("\n🎯 Prediction errors (pixels):")
            print(f"  Range: {result_df['prediction_error'].min():.1f} - {result_df['prediction_error'].max():.1f}")
            print(f"  Mean:  {result_df['prediction_error'].mean():.1f}")
            print(f"  Median: {result_df['prediction_error'].median():.1f}")

            print("\n⏱️  Time gaps (seconds):")
            print(f"  Range: {result_df['gap_sec'].min():.3f} - {result_df['gap_sec'].max():.3f}")
            print(f"  Mean:  {result_df['gap_sec'].mean():.3f}")

            print("\n📏 Y-distances (pixels):")
            print(f"  Range: {result_df['y_dist'].min():.1f} - {result_df['y_dist'].max():.1f}")
            print(f"  Mean:  {result_df['y_dist'].mean():.1f}")

            print("\n📐 Width differences (%):")
            print(f"  Range: {result_df['width_diff_pct'].min():.1f} - {result_df['width_diff_pct'].max():.1f}")
            print(f"  Mean:  {result_df['width_diff_pct'].mean():.1f}")

            print("\n🏆 Top 10 pairs by confidence:")
            print(result_df[['old_id', 'new_id', 'confidence', 'prediction_error', 'gap_sec',
                           'y_dist', 'direction_a', 'direction_b']].head(10).to_string(index=False))
        else:
            print("\nℹ️  No pairs found with current parameters.")
            print("Try relaxing the constraints:")
            print("  - Increase max_time_gap (current: {:.1f}s)".format(max_time_gap))
            print("  - Increase max_prediction_error (current: {:.0f}px)".format(max_prediction_error))
            print("  - Lower min_confidence (current: {:.2f})".format(min_confidence))

    return result_df


# Print confirmation when module is loaded
print("✅ find_merging_pairs_fiorenzo_2() loaded successfully!")
print("   Novel trajectory prediction-based merge detection ready to use.")
