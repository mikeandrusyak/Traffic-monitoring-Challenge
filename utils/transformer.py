import numpy as np
import pandas as pd

def classify_tracks(metrics):
    # 1. GHOST: Very short tracks (technical noise)
    is_ghost = (metrics['frames_count'] < 10)

    # 2. PERFECT: Passed almost the entire path with high WIDTH stability
    is_perfect = (
        (metrics['path_completeness'] > 0.85) & 
        (metrics['w_cv'] < 0.20)
    )

    # 3. ENTRY/EXIT: Ideal passages where HEIGHT (heigth) changed regularly
    # (objects 153, 238, 435 will fall into this category)
    is_entry_exit = (
        (metrics['path_completeness'] > 0.85) & 
        (metrics['w_cv'] < 0.30) & 
        (metrics['h_cv'] > 0.35)
    )

    # 4. STATIC: Objects that are standing or "accumulating" frames without movement
    is_static = (
        (metrics['frames_count'] > 100) & 
        (metrics['path_completeness'] < 0.15)
    )

    # 5. FLICKERING: Chaotic width changes (sign of poor mask)
    is_flickering = (metrics['w_cv'] > 0.45)

    # 6. PARTIAL: Stable path fragments (30-85% ROI)
    is_partial = (
        (metrics['path_completeness'].between(0.3, 0.85)) & 
        (metrics['w_cv'] < 0.30)
    )

    # Define conditions in priority order
    conditions = [
        is_ghost,
        is_perfect,
        is_entry_exit,
        is_static,
        is_flickering,
        is_partial
    ]

    choices = [
        'Ghost', 
        'Perfect', 
        'EntryExit', 
        'Static', 
        'Flickering', 
        'Partial'
    ]

    # Everything that didn't match conditions becomes RelayCandidate
    metrics['category'] = np.select(conditions, choices, default='RelayCandidate')
    
    return metrics

def categorize_ids(df):
    # Global sorting to avoid chaos in sessions
    df['date_time'] = pd.to_datetime(df['date_time'])
    df = df.sort_values(['date_time', 'frame_id'])

    # Group by session and ID
    grouped = df.groupby(['session_id', 'vehicle_id'])
    ROI_H = 290 
    
    metrics = grouped.agg(
        y_start=('y', 'first'),
        y_end=('y', 'last'),
        w_mean=('width', 'mean'),
        w_std=('width', 'std'),
        h_mean=('heigth', 'mean'), # Remember the typo 'heigth' in your database
        h_std=('heigth', 'std'),
        frames_count=('frame_id', 'count'),
        t_start=('date_time', 'min'),
        t_end=('date_time', 'max'),
        x_mean=('x', 'mean'),
        x_std=('x', 'std')
    ).reset_index()

    # Calculate key coefficients
    metrics['path_completeness'] = (metrics['y_end'] - metrics['y_start']).abs() / ROI_H
    metrics['w_cv'] = (metrics['w_std'] / metrics['w_mean']).fillna(0)
    metrics['h_cv'] = (metrics['h_std'] / metrics['h_mean']).fillna(0)

    return classify_tracks(metrics)