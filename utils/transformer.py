import pandas as pd
import numpy as np

def classify_tracks(metrics):
    conditions = [
        # PERFECT: Over 85% of the way there and stable size
        (metrics['path_completeness'] > 0.85) & (metrics['size_cv'] < 0.15),
        
        # GHOST: Very short (less than 10 frames)
        (metrics['frames_count'] < 10),
        
        # FLICKERING: Strong jumps in width
        (metrics['size_cv'] > 0.4),
        
        # PARTIAL: I have come a long way, but not all the way
        (metrics['path_completeness'].between(0.3, 0.85)) & (metrics['size_cv'] < 0.25)
    ]
    
    choices = ['Perfect', 'Ghost', 'Flickering', 'Partial']
    
    metrics['category'] = np.select(conditions, choices, default='Unclassified/RelayCandidate')
    return metrics

def categirze_ids(df):
    # Group by ID and session
    grouped = df.groupby(['session_id', 'vehicle_id'])
    
    # ROI height (from code 460 - 170 = 290)
    ROI_H = 290 
    
    metrics = grouped.agg(
        y_start=('y', 'first'),
        y_end=('y', 'last'),
        w_mean=('width', 'mean'),
        w_std=('width', 'std'),
        frames_count=('frame_id', 'count'),
        t_start=('date_time', 'min'),
        t_end=('date_time', 'max'),
        x_mean=('x', 'mean')
    ).reset_index()

    # 1. The completeness of the path (Path Completeness)
    metrics['path_completeness'] = (metrics['y_end'] - metrics['y_start']).abs() / ROI_H

    # 2. Size stability (Coefficient of Variation)
    # fillna(0) for tracks with 1 frame, where std cannot be calculated
    metrics['size_cv'] = (metrics['w_std'] / metrics['w_mean']).fillna(0)

    final_summary = classify_tracks(metrics)
    
    return final_summary
