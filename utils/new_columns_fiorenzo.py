import pandas as pd
import os
from pathlib import Path

project_root = Path().absolute().parent
path_sep = os.path.sep

sunset_times = pd.read_csv(str(project_root) + path_sep + 'utils' + path_sep + 'sunset.csv')
sunset_times['Day'] = pd.to_datetime(sunset_times['Day'], format='%d.%m.%Y %H:%M') # Data from https://www.timeanddate.com

total_y_pixel = 300
total_y_meters = 25

pixel_meters_ratio = total_y_pixel / total_y_meters


def calculate_velocity_px_per_s(track_row):
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
    duration = track_row['duration']

    if duration <= 0:
        return 0.0, 0.0, 'static'

    # Calculate Y velocity (main movement axis)
    dy = track_row['y_end'] - track_row['y_start']
    vy = dy / duration

    # Calculate X velocity (use x_std as proxy for lateral movement)
    # For tracks with little lateral movement, x_std will be small
    vx = track_row.get('x_std', 0) / duration

    # Determine direction
    if abs(vy) < 1.0:  # Less than 1 pixel/second ~ static
        direction = 'static'
    elif vy > 0:
        direction = 'down'
    else:
        direction = 'up'
        vy = -1.0 * vy

    return vx, vy, direction

def calculate_velocity_km_per_s(track_row):
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
    duration = track_row['duration']

    if duration <= 0:
        return 0.0, 0.0, 'static'

    # Calculate Y velocity (main movement axis)
    dy_pixels = track_row['y_end'] - track_row['y_start']
    dy_meters = dy_pixels / pixel_meters_ratio
    vy = dy_meters / duration
    vy = vy * 60 * 60 / 1000

    # Calculate X velocity (use x_std as proxy for lateral movement)
    # For tracks with little lateral movement, x_std will be small
    vx = track_row.get('x_std', 0) / duration if duration > 0 else 0

    # Determine direction
    if abs(vy) < (1.0 / pixel_meters_ratio):  # Less than 1 pixel/second ~ static
        direction = 'static'
    elif vy > 0:
        direction = 'down'
    else:
        direction = 'up'
        vy = -1.0 * vy

    return vx, vy, direction

def add_speed_direction_to_summary(data_summary):
    '''
    Adds speed (pixel per second) and direction (static/down/up) columns to the summary dataframe.
    '''

    if 't_start' not in data_summary.columns or 't_end' not in data_summary.columns:
        raise ValueError("Input dataframe must contain 't_start' and 't_end' columns.")
    if 'y_start' not in data_summary.columns or 'y_end' not in data_summary.columns:
        raise ValueError("Input dataframe must contain 'y_start' and 'y_end' columns.")
    if 'x_std' not in data_summary.columns:
        raise ValueError("Input dataframe must contain 'x_std' column.")
    for idx in data_summary.index:
        vx_pixels, vy_pixels, direction = calculate_velocity_px_per_s(data_summary.loc[idx])
        vx_meters, vy_meters, direction = calculate_velocity_km_per_s(data_summary.loc[idx])
        data_summary.loc[idx, 'velocity_x_px_seconds'] = vx_pixels
        data_summary.loc[idx, 'velocity_y_px_seconds'] = vy_pixels
        data_summary.loc[idx, 'velocity_x_km_h'] = vx_meters
        data_summary.loc[idx, 'velocity_y_km_h'] = vy_meters
        data_summary.loc[idx, 'direction'] = direction
    data_summary = data_summary[(data_summary['velocity_y_km_h'] > 0) & (data_summary['velocity_y_km_h'] < 35)]
    return data_summary

def define_day_or_night(track_row):
    day_or_night = 'error'
    for idx in range(0, len(sunset_times)):
        if track_row['t_end'].date() == sunset_times.loc[idx]['Day'].date():
            if track_row['t_end'].time() < sunset_times.loc[idx]['Day'].time():
                day_or_night = 'Day'
            else:
                day_or_night = 'Night'
    return day_or_night

def add_day_night_to_summary(data_summary):
    for idx in data_summary.index:
        day_or_night = define_day_or_night(data_summary.loc[idx])
        data_summary.loc[idx, 'solar_phase'] = day_or_night
    return data_summary


def classify_vehicle_types(summary_df,
                           day_height_threshold=120,
                           night_height_threshold=120,
                           night_width_threshold=0,
                           night_duration_threshold=None,
                           verbose=True):
    """
    Classifies vehicles as 'Car' or 'Truck' using different strategies for day and night.

    Day classification:
    - Uses h_mean (height) threshold only
    - h_mean < day_height_threshold -> Car
    - h_mean >= day_height_threshold -> Truck

    Night classification:
    - Uses three thresholds (OR logic - any condition makes it a Truck):
    - h_mean >= night_height_threshold -> Truck
    - w_mean >= night_width_threshold -> Truck
    - duration >= night_duration_threshold -> Truck
    - If none of the above -> Car

    Parameters:
    -----------
    summary_df : pd.DataFrame
        DataFrame with vehicle metrics
        Must contain columns: 'h_mean', 'w_mean', 'duration', 'solar_phase'

    day_height_threshold : float, default=120
        Height threshold in pixels for day classification
        Vehicles with h_mean >= threshold are classified as Truck

    night_height_threshold : float, default=120
        Height threshold in pixels for night classification

    night_width_threshold : float, default=0
        Width threshold in pixels for night classification

    night_duration_threshold : float, optional
        Duration threshold in seconds for night classification
        If None, will be calculated as midpoint between mean car and truck duration from day data

    verbose : bool, default=True
        If True, prints classification statistics and thresholds

    Returns:
    --------
    pd.DataFrame
        Copy of input dataframe with added 'Class' column ('Car' or 'Truck')
    """
    import numpy as np

    # Validate input
    required_cols = ['h_mean', 'w_mean', 'duration', 'solar_phase']
    missing_cols = [col for col in required_cols if col not in summary_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create copy to avoid modifying original
    df = summary_df.copy()

    # Initialize Class column
    df['Class'] = None

    # Step 1: Classify DAY vehicles using height threshold
    day_mask = df['solar_phase'] == 'Day'
    df.loc[day_mask, 'Class'] = df.loc[day_mask, 'h_mean'].apply(
        lambda h: 'Truck' if h >= day_height_threshold else 'Car'
    )

    if verbose:
        print("=" * 70)
        print("VEHICLE CLASSIFICATION RESULTS - DAY")
        print("=" * 70)
        day_count = day_mask.sum()
        print(f"\nVehicles classified during Day: {day_count}")
        print(f"\nClassification threshold used for Day:")
        print(f"  Height threshold:        {day_height_threshold:>8.2f} pixels")

        print(f"\nClass distribution for Day:")
        day_class_counts = df[day_mask]['Class'].value_counts()
        for cls in ['Car', 'Truck']:
            count = day_class_counts.get(cls, 0)
            pct = (count / day_count * 100) if day_count > 0 else 0
            print(f"  {cls:<6s}: {count:>4d} ({pct:>5.1f}%)")

    # Step 2: Calculate duration threshold from day data if not provided
    if night_duration_threshold is None:
        day_cars = df[(df['solar_phase'] == 'Day') & (df['Class'] == 'Car')]
        day_trucks = df[(df['solar_phase'] == 'Day') & (df['Class'] == 'Truck')]

        if len(day_cars) > 0 and len(day_trucks) > 0:
            mean_car_duration = day_cars['duration'].mean()
            mean_truck_duration = day_trucks['duration'].mean()
            # Use midpoint between mean durations as threshold
            night_duration_threshold = (mean_car_duration + mean_truck_duration) / 2
        else:
            # Fallback if no day data available
            night_duration_threshold = df['duration'].median()
            mean_car_duration = np.nan
            mean_truck_duration = np.nan

        if verbose:
            print(f"\nDuration threshold calculated from Day data:")
            print(f"  Mean Car duration:       {mean_car_duration:>8.2f} seconds")
            print(f"  Mean Truck duration:     {mean_truck_duration:>8.2f} seconds")
            print(f"  Calculated threshold:    {night_duration_threshold:>8.2f} seconds (midpoint)")

    print("\n" + "=" * 70)

    # Step 3: Classify NIGHT vehicles using three thresholds (OR logic)
    night_mask = df['solar_phase'] == 'Night'

    def classify_night_vehicle(row):
        if row['h_mean'] < night_height_threshold:
            return 'Car'
        if row['w_mean'] < night_width_threshold:
            return 'Car'
        if row['duration'] < night_duration_threshold:
            return 'Car'
        return 'Truck'

    df.loc[night_mask, 'Class'] = df.loc[night_mask].apply(classify_night_vehicle, axis=1)

    if verbose:
        print("=" * 70)
        print("VEHICLE CLASSIFICATION RESULTS - NIGHT")
        print("=" * 70)
        night_count = night_mask.sum()
        print(f"\nVehicles classified during Night: {night_count}")
        print(f"\nClassification thresholds used for Night (OR logic):")
        print(f"  Height threshold:        {night_height_threshold:>8.2f} pixels")
        print(f"  Width threshold:         {night_width_threshold:>8.2f} pixels")
        print(f"  Duration threshold:      {night_duration_threshold:>8.2f} seconds")

        # Count how many trucks were classified by each criterion
        night_df = df[night_mask]
        trucks_by_height = ((night_df['h_mean'] >= night_height_threshold)).sum()
        trucks_by_width = ((night_df['w_mean'] >= night_width_threshold) &
                          (night_df['h_mean'] < night_height_threshold)).sum()
        trucks_by_duration = ((night_df['duration'] >= night_duration_threshold) &
                             (night_df['h_mean'] < night_height_threshold) &
                             (night_df['w_mean'] < night_width_threshold)).sum()

        print(f"\nTrucks classified by criterion:")
        print(f"  By height:               {trucks_by_height:>4d}")
        print(f"  By width (not height):   {trucks_by_width:>4d}")
        print(f"  By duration (not h/w):   {trucks_by_duration:>4d}")

        print(f"\nClass distribution for Night:")
        night_class_counts = df[night_mask]['Class'].value_counts()
        for cls in ['Car', 'Truck']:
            count = night_class_counts.get(cls, 0)
            pct = (count / night_count * 100) if night_count > 0 else 0
            print(f"  {cls:<6s}: {count:>4d} ({pct:>5.1f}%)")
        print("\n" + "=" * 70)

    # Overall summary
    if verbose:
        print("=" * 70)
        print("OVERALL CLASSIFICATION SUMMARY")
        print("=" * 70)
        print(f"\nTotal vehicles classified: {len(df)}")

        print(f"\nOverall class distribution:")
        class_counts = df['Class'].value_counts()
        for cls in ['Car', 'Truck']:
            count = class_counts.get(cls, 0)
            pct = (count / len(df) * 100) if len(df) > 0 else 0
            print(f"  {cls:<6s}: {count:>4d} ({pct:>5.1f}%)")

        print("\n" + "=" * 70)

    return df