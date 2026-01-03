import numpy as np
import pandas as pd


def classify_tracks(metrics):
    """
    Classifies tracks based on geometric and temporal metrics.
    """
    # Calculate movement efficiency (path per frame)
    # This helps identify objects that were stationary (Static) at any point in the ROI
    metrics['movement_efficiency'] = metrics['path_completeness'] / metrics['frames_count']

    # --- Classification conditions ---

    # 1. GHOST: Technical noise (very short tracks)
    is_ghost = (metrics['frames_count'] < 30) | (metrics['path_completeness'] < 0.1)

    # 2. STATIC: Stationary object (at start, end, or in traffic jam)
    # If there's too little movement per frame
    is_static = (metrics['movement_efficiency'] < 0.0015) | \
                ((metrics['frames_count'] > 200) & (metrics['path_completeness'] < 0.3))

    # 3. PERFECT: Ideal passage (stable width, full path, normal speed)
    is_perfect = (
            (metrics['path_completeness'] > 0.95) &
            (metrics['w_cv'] < 0.45) &
            (metrics['h_cv'] < 0.45) &
            (metrics['movement_efficiency'] >= 0.0015)
    )

    # 4. FLICKERING: Unstable object (strong width jumps)
    is_flickering = (metrics['w_cv'] >= 0.45)

    # 5. PARTIAL: Stable fragments (vehicles that appeared/disappeared mid-frame)
    is_partial = (
            (metrics['path_completeness'] > 0.3) &
            (metrics['w_cv'] < 0.45) &
            (metrics['movement_efficiency'] >= 0.0015)
    )

    # Priority order (from most important/simplest to general)
    conditions = [
        is_ghost,
        is_static,
        is_perfect,
        is_flickering,
        is_partial
    ]

    choices = [
        'Ghost',
        'Static',
        'Perfect',
        'Flickering',
        'Partial'
    ]

    # All others become candidates for merging (Noise)
    metrics['category'] = np.select(conditions, choices, default='Noise')

    return metrics


def categorize_ids(df):
    """
    Aggregates raw data into metrics for each vehicle and classifies them.
    """
    # Convert time and sort for calculation stability
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(['session_id', 'date_time', 'frame_id'])

    # Group by session and ID
    grouped = df.groupby(['session_id', 'vehicle_id'])

    # ROI height (according to your settings 460 - 170)
    ROI_H = 290

    # Data aggregation
    metrics = grouped.agg(
        y_start=('y', 'first'),
        y_end=('y', 'last'),
        w_mean=('width', 'mean'),
        w_std=('width', 'std'),
        w_start=('width', 'first'),  # Width of first frame (green box)
        w_end=('width', 'last'),  # Width of last frame (red box)
        h_mean=('heigth', 'mean'),
        h_std=('heigth', 'std'),
        h_start=('heigth', 'first'),  # Height of first frame (green box)
        h_end=('heigth', 'last'),  # Height of last frame (red box)
        frames_count=('frame_id', 'count'),
        t_start=('date_time', 'min'),
        t_end=('date_time', 'max'),
        x_mean=('x', 'mean'),
        x_std=('x', 'std')
    ).reset_index()

    # Calculate path completeness (0.0 - 1.0)
    # Direction-aware: measure from green box start to red box end
    # Moving down (y_end > y_start): from top of green to bottom of red
    # Moving up (y_end < y_start): from bottom of green to top of red
    moving_down = (metrics['x_mean'] + metrics['w_mean']) <= 140
    path_down = (metrics['y_end'] + metrics['h_end']) - metrics['y_start']
    path_up = (metrics['y_start'] + metrics['h_start']) - metrics['y_end']
    metrics['path_completeness'] = (moving_down * path_down + ~moving_down * path_up).abs() / ROI_H

    # Calculate size stability (Coefficient of Variation)
    # Use fillna(0) for single-frame objects
    metrics['w_cv'] = (metrics['w_std'] / metrics['w_mean']).fillna(0)
    metrics['h_cv'] = (metrics['h_std'] / metrics['h_mean']).fillna(0)

    # Run classification
    final_summary = classify_tracks(metrics)

    return final_summary


def find_merging_pairs(summary_df, category_filter=None, time_gap_limit=1.5, space_gap_limit=40, size_sim_limit=0.2):
    """
    summary_df: result of metrics calculation (final_summary)
    time_gap_limit: max time gap (seconds)
    space_gap_limit: max distance between points (pixels)
    size_sim_limit: max width difference (relative, 0.2 = 20%)
    """

    # Filter by categories if specified
    if category_filter is not None:
        candidates = summary_df[summary_df['category'].isin(category_filter)]
    else:
        candidates = summary_df.copy()

    # Sort by appearance time
    candidates = candidates.sort_values('t_start')

    merges = []
    used_as_new = set()  # Track IDs that are already used as new_id

    # Convert to list of dictionaries for fast iteration
    records = candidates.to_dict('records')

    for i in range(len(records)):
        id_a = records[i]
        # Allow id_a even if it was previously a new_id (enables chains)

        for j in range(i + 1, len(records)):
            id_b = records[j]
            # Prevent id_b from being merged multiple times as new_id
            if id_b['vehicle_id'] in used_as_new: continue

            # 1. Session check (must be in the same session)
            if id_a['session_id'] != id_b['session_id']: continue

            # 2. Time gap (A ended, B started soon after)
            gap_time = (id_b['t_start'] - id_a['t_end']).total_seconds()

            # OPTIMISATION: stop the internal cycle if the time limit has been exceeded
            if gap_time > (time_gap_limit + 2.0):
                break

            # We're looking for B that comes AFTER A, but not later than the limit
            if -1.0 <= gap_time <= time_gap_limit:

                # 2.5. Direction check (must move in the same direction)
                direction_a = id_a['y_end'] - id_a['y_start']
                direction_b = id_b['y_end'] - id_b['y_start']

                # Skip if directions are opposite (one positive, one negative)
                if direction_a * direction_b < 0:
                    continue

                # 2.6. Lane check (must be in the same lane)
                # Left lane: x + w <= 140 (moving down)
                # Right lane: x + w > 140 (moving up)
                lane_a = (id_a['x_mean'] + id_a['w_mean']) <= 140
                lane_b = (id_b['x_mean'] + id_b['w_mean']) <= 140

                # Skip if in different lanes
                if lane_a != lane_b:
                    continue

                # 3. Spatial proximity (end of A to start of B)
                # Use Y as it's the main axis of movement
                dist_y = abs(id_b['y_start'] - id_a['y_end'])
                dist_x = abs(id_b['x_mean'] - id_a['x_mean'])

                # 4. Size similarity checks
                # Check average width similarity
                width_diff = abs(id_a['w_mean'] - id_b['w_mean']) / id_a['w_mean']

                # Check transition sizes (end of A should match start of B)
                # This helps distinguish one vehicle from two consecutive vehicles
                transition_width_diff = abs(id_a['w_end'] - id_b['w_start']) / max(id_a['w_end'], 1)
                transition_height_diff = abs(id_a['h_end'] - id_b['h_start']) / max(id_a['h_end'], 1)

                if (dist_y < space_gap_limit and dist_x < 20 and
                        width_diff < size_sim_limit and
                        transition_width_diff < size_sim_limit and
                        transition_height_diff < size_sim_limit):
                    merges.append({
                        'old_id': int(id_a['vehicle_id']),
                        'new_id': int(id_b['vehicle_id']),
                        'gap_sec': round(gap_time, 2),
                        'y_dist': round(dist_y, 1),
                        'size_diff_pct': round(width_diff * 100, 1),
                        'transition_w_diff_pct': round(transition_width_diff * 100, 1),
                        'transition_h_diff_pct': round(transition_height_diff * 100, 1),
                        # Coordinates for old_id
                        'old_x_start': round(id_a['x_mean'], 1),
                        'old_y_start': round(id_a['y_start'], 1),
                        'old_x_end': round(id_a['x_mean'], 1),
                        'old_y_end': round(id_a['y_end'], 1),
                        'old_h_start': round(id_a['h_start'], 1),
                        'old_h_end': round(id_a['h_end'], 1),
                        'old_w_mean': round(id_a['w_mean'], 1),
                        'old_w_start': round(id_a['w_start'], 1),
                        'old_w_end': round(id_a['w_end'], 1),
                        # Coordinates for new_id
                        'new_x_start': round(id_b['x_mean'], 1),
                        'new_y_start': round(id_b['y_start'], 1),
                        'new_x_end': round(id_b['x_mean'], 1),
                        'new_y_end': round(id_b['y_end'], 1),
                        'new_h_start': round(id_b['h_start'], 1),
                        'new_h_end': round(id_b['h_end'], 1),
                        'new_w_mean': round(id_b['w_mean'], 1),
                        'new_w_start': round(id_b['w_start'], 1),
                        'new_w_end': round(id_b['w_end'], 1),
                        # Time stamps
                        'old_t_start': id_a['t_start'],
                        'old_t_end': id_a['t_end'],
                        'new_t_start': id_b['t_start'],
                        'new_t_end': id_b['t_end']
                    })
                    # Mark id_b as used, but id_b can still be old_id for next merge
                    # This enables chains: 1->2, 2->3, 3->4
                    used_as_new.add(id_b['vehicle_id'])
                    break

    return pd.DataFrame(merges)


def build_merge_chains(merge_results):
    """
    Build chains of merges from merge_results DataFrame.
    Example: if we have pairs 448→451 and 451→454, return chain [448, 451, 454]
    """
    # Build a mapping: old_id -> new_id
    merge_map = {}
    all_new_ids = set()

    for _, row in merge_results.iterrows():
        merge_map[row['old_id']] = row['new_id']
        all_new_ids.add(row['new_id'])

    # Find chain starts (IDs that are old_id but never new_id)
    chain_starts = set(merge_map.keys()) - all_new_ids

    # Build chains
    chains = []
    for start_id in chain_starts:
        chain = [start_id]
        current_id = start_id

        # Follow the chain
        while current_id in merge_map:
            next_id = merge_map[current_id]
            chain.append(next_id)
            current_id = next_id

        # Only include chains with 2+ IDs
        if len(chain) >= 2:
            chains.append(chain)

    # Get time information for each chain (time of first ID)
    chain_times = []
    for chain in chains:
        # Find time of the first ID in the chain
        first_id = chain[0]
        # Look for this ID in merge_results
        first_row = merge_results[merge_results['old_id'] == first_id].iloc[0] if len(
            merge_results[merge_results['old_id'] == first_id]) > 0 else None
        if first_row is not None:
            chain_times.append((chain, first_row['old_t_start']))
        else:
            # Fallback: use current time if not found
            chain_times.append((chain, pd.Timestamp.now()))

    # Sort chains by time (earliest first)
    chain_times.sort(key=lambda x: x[1])
    chains = [c[0] for c in chain_times]

    return chains


def apply_merges_to_summary(summary_df, chains, unified_id_start=1,
                            categories_for_unified_id=None,
                            consolidate=True, ROI_H=290):
    """
    Apply merge chains to final_summary, creating unified_id column and Merged category.
    Also assigns unique unified_id to non-merged IDs from specified categories.
    Optionally consolidates merged IDs into single rows.

    Parameters:
    -----------
    summary_df : pd.DataFrame
        The final_summary dataframe with vehicle metrics
    chains : list of lists
        Merge chains from build_merge_chains()
    unified_id_start : int
        Starting number for unified IDs (default: 1)
    categories_for_unified_id : list of str, optional
        Categories that should receive unified_id (even if not merged)
        Default: ['Merged', 'Perfect', 'Partial']
        Merged IDs get one unified_id per chain and category='Merged'
        Non-merged IDs get individual unified_id and keep original category
    consolidate : bool
        If True, consolidates multiple rows with same unified_id into one row (default: True)
    ROI_H : float
        Height of ROI for path_completeness calculation when consolidating (default: 290)

    Returns:
    --------
    pd.DataFrame
        Updated summary with unified_id column, optionally consolidated
    """
    # Default categories
    if categories_for_unified_id is None:
        categories_for_unified_id = ['Merged', 'Perfect', 'Partial']

    # Create a copy to avoid modifying original
    result_df = summary_df.copy()

    # Initialize unified_id column with NaN
    result_df['unified_id'] = pd.NA

    # Collect all candidates for unified_id assignment with their timestamps
    # Format: (timestamp, type, data) where type is 'chain' or 'individual'
    unified_candidates = []

    # Add chains
    for chain in chains:
        first_id = chain[0]
        first_row = result_df[result_df['vehicle_id'] == first_id]
        if len(first_row) > 0:
            unified_candidates.append((first_row.iloc[0]['t_start'], 'chain', chain))

    # Add individual IDs from specified categories (not in chains)
    ids_in_chains = set()
    for chain in chains:
        ids_in_chains.update(chain)

    individual_candidates = result_df[
        (~result_df['vehicle_id'].isin(ids_in_chains)) &
        (result_df['category'].isin(categories_for_unified_id))
        ].copy()

    for _, row in individual_candidates.iterrows():
        unified_candidates.append((row['t_start'], 'individual', row['vehicle_id']))

    # Sort all candidates by timestamp (chronological order)
    unified_candidates.sort(key=lambda x: x[0])

    # Assign unified_id in chronological order
    current_unified_id = unified_id_start

    for timestamp, candidate_type, data in unified_candidates:
        if candidate_type == 'chain':
            # Assign same unified_id to all IDs in chain
            chain = data
            for vid in chain:
                result_df.loc[result_df['vehicle_id'] == vid, 'unified_id'] = current_unified_id
                result_df.loc[result_df['vehicle_id'] == vid, 'category'] = 'Merged'
        else:
            # Assign unified_id to individual ID
            vid = data
            result_df.loc[result_df['vehicle_id'] == vid, 'unified_id'] = current_unified_id
            # Keep original category for individuals

        current_unified_id += 1

    # Step 3: Consolidate if requested
    if consolidate:
        result_df = _consolidate_merged_ids(result_df, summary_df, ROI_H)

    return result_df


def _consolidate_merged_ids(summary_df, original_summary_df, ROI_H=290):
    """
    Internal helper: Consolidate rows with same unified_id and category='Merged' into single rows.
    Non-merged IDs with unified_id keep individual rows but get vehicle_id as list.
    IDs without unified_id remain as is with vehicle_id as list.

    Parameters:
    -----------
    summary_df : pd.DataFrame
        DataFrame with unified_id and updated categories
    original_summary_df : pd.DataFrame
        Original summary_df before applying merges (to restore original categories)
    ROI_H : float
        Height of ROI for path_completeness calculation
    """
    # Separate merged, non-merged with unified_id, and without unified_id
    merged_df = summary_df[summary_df['category'] == 'Merged'].copy()
    non_merged_with_unified = summary_df[
        (summary_df['category'] != 'Merged') &
        (summary_df['unified_id'].notna())
        ].copy()
    without_unified = summary_df[summary_df['unified_id'].isna()].copy()

    consolidated_rows = []

    # Consolidate merged IDs (multiple rows -> one row per unified_id)
    if len(merged_df) > 0:
        for unified_id, group in merged_df.groupby('unified_id'):
            # Sort by time to determine direction
            group = group.sort_values('t_start')

            # Determine direction based on x position (lane detection)
            # Left lane (x + w <= 140): moving down
            # Right lane (x + w > 140): moving up
            x_mean_avg = group['x_mean'].mean()
            w_mean_avg = group['w_mean'].mean()
            moving_down = (x_mean_avg + w_mean_avg) <= 140

            # Get first and last box dimensions
            first_row = group.iloc[0]
            last_row = group.iloc[-1]

            # Calculate path_completeness using the same formula as categorize_ids
            # Direction-aware: measure from green box start to red box end
            path_down = (last_row['y_end'] + last_row['h_end']) - first_row['y_start']
            path_up = (first_row['y_start'] + first_row['h_start']) - last_row['y_end']
            path_completeness = abs(moving_down * path_down + ~moving_down * path_up) / ROI_H

            # Store y coordinates from first and last rows (for consolidated record)
            y_start = first_row['y_start']
            y_end = last_row['y_end']

            # Aggregate metrics
            w_mean_avg = group['w_mean'].mean()
            w_std_avg = group['w_std'].mean()
            h_mean_avg = group['h_mean'].mean()
            h_std_avg = group['h_std'].mean()

            # Recalculate CV
            w_cv = w_std_avg / w_mean_avg if w_mean_avg > 0 else 0
            h_cv = h_std_avg / h_mean_avg if h_mean_avg > 0 else 0

            # Sum frames_count
            frames_count_total = group['frames_count'].sum()

            # Recalculate movement_efficiency
            movement_efficiency = path_completeness / frames_count_total if frames_count_total > 0 else 0

            consolidated_row = {
                'session_id': group['session_id'].iloc[0],
                'vehicle_id': group['vehicle_id'].tolist(),  # List of merged IDs
                'y_start': y_start,
                'y_end': y_end,
                'w_mean': w_mean_avg,
                'w_std': w_std_avg,
                'w_start': first_row['w_start'],
                'w_end': last_row['w_end'],
                'h_mean': h_mean_avg,
                'h_std': h_std_avg,
                'h_start': first_row['h_start'],
                'h_end': last_row['h_end'],
                'frames_count': frames_count_total,
                't_start': group['t_start'].min(),
                't_end': group['t_end'].max(),
                'x_mean': group['x_mean'].mean(),
                'x_std': group['x_std'].mean(),
                'path_completeness': path_completeness,
                'w_cv': w_cv,
                'h_cv': h_cv,
                'movement_efficiency': movement_efficiency,
                'category': 'Merged',
                'unified_id': unified_id
            }

            consolidated_rows.append(consolidated_row)

    # Convert vehicle_id to list for all non-consolidated rows (for consistency)
    non_merged_with_unified['vehicle_id'] = non_merged_with_unified['vehicle_id'].apply(lambda x: [x])
    without_unified['vehicle_id'] = without_unified['vehicle_id'].apply(lambda x: [x])

    # Combine all parts
    if consolidated_rows:
        consolidated_df = pd.DataFrame(consolidated_rows)

        # Filter consolidated merges: only keep those with path_completeness > 0.3
        # Split back those with insufficient path_completeness
        valid_merges = consolidated_df[consolidated_df['path_completeness'] > 0.3].copy()
        invalid_merges = consolidated_df[consolidated_df['path_completeness'] <= 0.3].copy()

        # Convert invalid merges back to individual rows
        unmerged_rows = []
        if len(invalid_merges) > 0:
            for _, merged_row in invalid_merges.iterrows():
                vehicle_ids = merged_row['vehicle_id']
                # Get original rows for these IDs from the original summary
                for vid in vehicle_ids:
                    original_row = original_summary_df[original_summary_df['vehicle_id'] == vid]
                    if len(original_row) > 0:
                        # Restore original row with original category, convert vehicle_id to list
                        restored = original_row.iloc[0].copy()
                        restored['vehicle_id'] = [restored['vehicle_id']]
                        restored['unified_id'] = pd.NA
                        # Original category is already restored from original_summary_df
                        unmerged_rows.append(restored)

        # Combine valid merges with unmerged and other categories
        if len(unmerged_rows) > 0:
            unmerged_df = pd.DataFrame(unmerged_rows)
            result_df = pd.concat([valid_merges, unmerged_df, non_merged_with_unified, without_unified],
                                  ignore_index=True)
        else:
            result_df = pd.concat([valid_merges, non_merged_with_unified, without_unified], ignore_index=True)
    else:
        result_df = pd.concat([non_merged_with_unified, without_unified], ignore_index=True)

    # Sort by session_id and t_start
    result_df = result_df.sort_values(['session_id', 't_start']).reset_index(drop=True)

    return result_df
