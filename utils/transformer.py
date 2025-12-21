import pandas as pd
import numpy as np

def classify_tracks(metrics):
    conditions = [
        # PERFECT: Пройшов > 85% шляху та стабільний розмір
        (metrics['path_completeness'] > 0.85) & (metrics['size_cv'] < 0.15),
        
        # GHOST: Дуже короткий (менше 10 кадрів)
        (metrics['frames_count'] < 10),
        
        # FLICKERING: Сильні стрибки ширини
        (metrics['size_cv'] > 0.4),
        
        # PARTIAL: Пройшов значну частину, але не весь шлях
        (metrics['path_completeness'].between(0.3, 0.85)) & (metrics['size_cv'] < 0.25)
    ]
    
    choices = ['Perfect', 'Ghost', 'Flickering', 'Partial']
    
    metrics['category'] = np.select(conditions, choices, default='Unclassified/RelayCandidate')
    return metrics

def categirze_ids(df):
    # Групуємо за ID та сесією
    grouped = df.groupby(['session_id', 'vehicle_id'])
    
    # ROI height (з вашого коду 460 - 170 = 290)
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

    # 1. Повнота шляху (Path Completeness)
    metrics['path_completeness'] = (metrics['y_end'] - metrics['y_start']).abs() / ROI_H

    # 2. Стабільність розміру (Coefficient of Variation)
    # fillna(0) для треків з 1 кадром, де std неможливо порахувати
    metrics['size_cv'] = (metrics['w_std'] / metrics['w_mean']).fillna(0)

    final_summary = classify_tracks(metrics)
    
    return final_summary
