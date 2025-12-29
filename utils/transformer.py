import numpy as np
import pandas as pd
from tqdm.auto import tqdm

def classify_tracks(metrics):
    """
    Classifies tracks based on geometric and temporal metrics.
    """
    # Calculate movement efficiency (path per frame)
    # This helps identify objects that were stationary (Static) at any point in the ROI
    metrics['movement_efficiency'] = metrics['path_completeness'] / metrics['frames_count']

    # --- Classification conditions ---

    # 1. STATIC: Stationary object (at start, end, or in traffic jam)
    # If there's too little movement per frame
    is_static = (metrics['movement_efficiency'] < 0.0015) | \
                ((metrics['frames_count'] > 200) & (metrics['path_completeness'] < 0.3))

    # 2. PERFECT: Ideal passage (stable width, full path, normal speed)
    is_perfect = (
        (metrics['path_completeness'] > 0.85) & 
        (metrics['w_cv'] < 0.30) & 
        (metrics['movement_efficiency'] >= 0.0015)
    )

    # 3. FLICKERING: Unstable object (strong width jumps)
    is_flickering = (metrics['w_cv'] > 0.45)

    # 4. PARTIAL: Stable fragments (vehicles that appeared/disappeared mid-frame)
    is_partial = (
        (metrics['path_completeness'].between(0.3, 0.85)) & 
        (metrics['w_cv'] < 0.30)
    )

    # Priority order (from most important/simplest to general)
    conditions = [
        is_static,
        is_perfect,
        is_flickering,
        is_partial
    ]

    choices = [
        'Static', 
        'Perfect', 
        'Flickering', 
        'Partial'
    ]

    # All others become candidates for merging (RelayCandidate)
    metrics['category'] = np.select(conditions, choices, default='RelayCandidate')
    
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
        h_mean=('heigth', 'mean'),
        h_std=('heigth', 'std'),
        frames_count=('frame_id', 'count'),
        t_start=('date_time', 'min'),
        t_end=('date_time', 'max'),
        x_mean=('x', 'mean'),
        x_std=('x', 'std')
    ).reset_index()

    # Calculate path completeness (0.0 - 1.0)
    metrics['path_completeness'] = (metrics['y_end'] - metrics['y_start']).abs() / ROI_H

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

    for i in tqdm(range(len(records)), desc="Search for pairs for merging", unit="ID"):
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
                
                # 3. Spatial proximity (end of A to start of B)
                # Use Y as it's the main axis of movement
                dist_y = abs(id_b['y_start'] - id_a['y_end'])
                dist_x = abs(id_b['x_mean'] - id_a['x_mean'])
                
                # 4. Size similarity (width shouldn't jump)
                size_diff = abs(id_a['w_mean'] - id_b['w_mean']) / id_a['w_mean']

                if dist_y < space_gap_limit and dist_x < 20 and size_diff < size_sim_limit:
                    merges.append({
                        'old_id': int(id_a['vehicle_id']),
                        'new_id': int(id_b['vehicle_id']),
                        'gap_sec': round(gap_time, 2),
                        'y_dist': round(dist_y, 1),
                        'size_diff_pct': round(size_diff * 100, 1),
                        # Coordinates for old_id
                        'old_x_start': round(id_a['x_mean'], 1),
                        'old_y_start': round(id_a['y_start'], 1),
                        'old_x_end': round(id_a['x_mean'], 1),
                        'old_y_end': round(id_a['y_end'], 1),
                        # Coordinates for new_id
                        'new_x_start': round(id_b['x_mean'], 1),
                        'new_y_start': round(id_b['y_start'], 1),
                        'new_x_end': round(id_b['x_mean'], 1),
                        'new_y_end': round(id_b['y_end'], 1),
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
        first_row = merge_results[merge_results['old_id'] == first_id].iloc[0] if len(merge_results[merge_results['old_id'] == first_id]) > 0 else None
        if first_row is not None:
            chain_times.append((chain, first_row['old_t_start']))
        else:
            # Fallback: use current time if not found
            chain_times.append((chain, pd.Timestamp.now()))
    
    # Sort chains by time (earliest first)
    chain_times.sort(key=lambda x: x[1])
    chains = [c[0] for c in chain_times]
    
    return chains