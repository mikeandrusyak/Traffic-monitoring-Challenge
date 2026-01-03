import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from scipy.stats import gaussian_kde
import pandas as pd
import os
from pathlib import Path

project_root = Path().absolute().parent
path_sep = os.path.sep

plots_dir = str(project_root) + path_sep + 'plots' + path_sep


def get_data_gap_intervals(data_summary):
    """
    Detects time gaps between session IDs where data collection was interrupted due to script errors.

    The function identifies gaps by finding the end time of each session and the start time of the next session.
    These gaps represent periods when the data collection script crashed and had to be restarted.

    Parameters:
    -----------
    data_summary : pd.DataFrame
        DataFrame with vehicle metrics
        Must contain columns: 'session_id', 't_start', 't_end'

    Returns:
    --------
    list of tuples
        List of (gap_start, gap_end) datetime tuples representing periods with no data
        Returns empty list if 'session_id' column is not present or only one session exists
    """
    if 'session_id' not in data_summary.columns:
        return []

    df = data_summary.copy()
    df['t_start'] = pd.to_datetime(df['t_start'])
    df['t_end'] = pd.to_datetime(df['t_end'])

    # Get unique sessions sorted by their start time
    session_times = df.groupby('session_id').agg({
        't_start': 'min',
        't_end': 'max'
    }).sort_values('t_start')

    if len(session_times) <= 1:
        return []

    gaps = []
    sessions = session_times.reset_index()

    for i in range(len(sessions) - 1):
        current_session_end = sessions.loc[i, 't_end']
        next_session_start = sessions.loc[i + 1, 't_start']

        # Only add gap if there's actually a time difference
        if next_session_start > current_session_end:
            gaps.append((current_session_end, next_session_start))

    return gaps


def add_data_gap_shading(ax, data_summary, alpha=0.3, color='gray', label_first=True):
    """
    Adds shaded regions to a matplotlib axes to indicate data gaps due to script errors.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add shading to

    data_summary : pd.DataFrame
        DataFrame with vehicle metrics (must contain 'session_id', 't_start', 't_end')

    alpha : float, default=0.3
        Transparency of the shaded regions

    color : str, default='gray'
        Color of the shaded regions

    label_first : bool, default=True
        If True, only the first gap region gets a legend label

    Returns:
    --------
    list of matplotlib.patches.Polygon
        List of the shaded region patches added to the axes
    """
    gaps = get_data_gap_intervals(data_summary)
    patches = []

    for i, (gap_start, gap_end) in enumerate(gaps):
        label = 'Script Error (No Data)' if (i == 0 and label_first) else None
        patch = ax.axvspan(gap_start, gap_end, alpha=alpha, color=color,
                          label=label, hatch='///', edgecolor='darkgray')
        patches.append(patch)

    return patches


def get_dates_with_gaps(data_summary):
    """
    Returns a set of dates that were affected by data collection gaps (script errors).

    A date is considered affected if a gap started or ended on that date,
    meaning the data for that date is incomplete.

    Parameters:
    -----------
    data_summary : pd.DataFrame
        DataFrame with vehicle metrics (must contain 'session_id', 't_start', 't_end')

    Returns:
    --------
    set of datetime.date
        Set of dates that had incomplete data due to script errors
    """
    gaps = get_data_gap_intervals(data_summary)
    affected_dates = set()

    for gap_start, gap_end in gaps:
        # Add all dates that fall within or touch the gap
        current = gap_start.date()
        end_date = gap_end.date()
        while current <= end_date:
            affected_dates.add(current)
            current = current + pd.Timedelta(days=1)

    return affected_dates


def add_gap_indicators_to_bar_chart(ax, unique_dates, data_summary, label_first=True, horizontal=False):
    """
    Adds visual indicators (hatched background) to bar chart positions for dates affected by script errors.

    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to add indicators to

    unique_dates : list
        List of dates (datetime.date objects) corresponding to bar positions

    data_summary : pd.DataFrame
        DataFrame with vehicle metrics (must contain 'session_id', 't_start', 't_end')

    label_first : bool, default=True
        If True, only the first indicator gets a legend label

    horizontal : bool, default=False
        If True, uses axhspan instead of axvspan for horizontal bar charts

    Returns:
    --------
    list of matplotlib.patches.Rectangle
        List of the indicator patches added to the axes
    """
    affected_dates = get_dates_with_gaps(data_summary)
    patches = []
    first_added = False

    for i, date in enumerate(unique_dates):
        if date in affected_dates:
            label = 'Script Error (Incomplete Data)' if (not first_added and label_first) else None
            if horizontal:
                patch = ax.axhspan(i - 0.4, i + 0.4, alpha=0.2, color='gray',
                                  hatch='///', edgecolor='darkgray', label=label, zorder=0)
            else:
                patch = ax.axvspan(i - 0.4, i + 0.4, alpha=0.2, color='gray',
                                  hatch='///', edgecolor='darkgray', label=label, zorder=0)
            patches.append(patch)
            first_added = True

    return patches

def visualize_classification(data_summary, show_plot=False, day_only=False, night_only=False):
    """
    Creates a 3x3 matrix of scatter plots showing vehicle dimensions by class (Car/Truck).

    Parameters:
    -----------
    data_summary : pd.DataFrame
        DataFrame with vehicle metrics including classification information
        Must contain columns: 'h_mean', 'w_mean', 'duration', 'Class'
        Optional column: 'solar_phase' (required if day_only or night_only is True)

    show_plot : bool, default=False
        If True, displays the plot interactively

    day_only : bool, default=False
        If True, shows only vehicles during day (solar_phase == 'Day')

    night_only : bool, default=False
        If True, shows only vehicles during night (solar_phase == 'Night')

    Note: If both day_only and night_only are False, all vehicles are shown.
          If both are True, all vehicles are shown.

    Returns:
    --------
    None
        Saves plot to 'plots/classification_visualization.png'
    """
    # Create copy to avoid modifying original
    df = data_summary.copy()

    # Handle filtering by solar phase
    filter_label = "All"
    if day_only and night_only:
        pass  # Show all if both are True
    elif day_only:
        if 'solar_phase' not in df.columns:
            raise ValueError("Column 'solar_phase' is required when day_only=True")
        df = df[df['solar_phase'] == 'Day']
        filter_label = "Day Only"
    elif night_only:
        if 'solar_phase' not in df.columns:
            raise ValueError("Column 'solar_phase' is required when night_only=True")
        df = df[df['solar_phase'] == 'Night']
        filter_label = "Night Only"

    if len(df) == 0:
        print(f"No data available for the selected filter ({filter_label}).")
        return

    # Dimensions to plot
    dims = ['h_mean', 'w_mean', 'duration']
    dim_labels = {
        'h_mean': 'Height Mean (pixels)',
        'w_mean': 'Width Mean (pixels)',
        'duration': 'Duration (s)'
    }
    n = len(dims)

    # Separate by class
    cars = df[df['Class'] == 'Car']
    trucks = df[df['Class'] == 'Truck']

    # Create figure with 3x3 subplots
    fig, axes = plt.subplots(n, n, figsize=(14, 12))

    # Color scheme
    car_color = '#636EFA'
    truck_color = '#EF553B'

    # Fill the matrix
    for i, y_col in enumerate(dims):
        for j, x_col in enumerate(dims):
            ax = axes[i, j]

            # Scatter plot for both classes
            ax.scatter(cars[x_col], cars[y_col], alpha=0.6, s=40, c=car_color,
                       edgecolors='white', linewidth=0.3, label='Car')
            ax.scatter(trucks[x_col], trucks[y_col], alpha=0.6, s=40, c=truck_color,
                       edgecolors='white', linewidth=0.3, label='Truck')

            # Set labels
            if i == n - 1:  # Bottom row
                ax.set_xlabel(dim_labels[x_col], fontsize=10, fontweight='bold')
            if j == 0:  # Left column
                ax.set_ylabel(dim_labels[y_col], fontsize=10, fontweight='bold')

            ax.grid(True, alpha=0.3)

            # Add legend only to top-right subplot
            if i == 0 and j == n - 1:
                ax.legend(fontsize=9, loc='upper right')

    plt.suptitle(f'Vehicle Classification Matrix ({filter_label})', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir + 'classification_visualization.png')
    if show_plot:
        plt.show()

    print("\n" + "=" * 70)
    print(f"CLASSIFICATION SUMMARY ({filter_label})")
    print("=" * 70)
    print(f"\nTotal vehicles: {len(df)}")
    print(f"  Cars:   {len(cars)} ({len(cars) / len(df) * 100:.1f}%)")
    print(f"  Trucks: {len(trucks)} ({len(trucks) / len(df) * 100:.1f}%)")
    print("\n" + "=" * 70)

def interactive_dimension_plot_by_cat(data_summary, show_plot=False):


    # 1. Parameters
    dims = ['w_mean', 'h_mean', 'size_mean', 'duration', 'h_w_mean_ratio']
    categories = data_summary['category'].unique()
    n = len(dims)

    # 2. Create subplot grid
    fig = make_subplots(
        rows=n, cols=n,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.03, vertical_spacing=0.03,
        column_titles=dims, row_titles=dims
    )

    # Color palette for all possible categories
    color_map = {
        'Perfect': '#636EFA',
        'Partial': '#EF553B',
        'Merged': '#00CC96',
        'Ghost': '#AB63FA',
        'Noise': '#FFA15A',
        'Static': '#19D3F3',
    }
    # Fallback colors for any unknown categories
    fallback_colors = ['#FF6692', '#B6E880', '#FF97FF', '#FECB52']

    # 3. Fill the matrix
    for i, y_col in enumerate(dims):
        for j, x_col in enumerate(dims):
            for k, cat in enumerate(categories):
                df_sub = data_summary[data_summary['category'] == cat]

                # Get color for this category
                if cat in color_map:
                    color = color_map[cat]
                else:
                    color = fallback_colors[k % len(fallback_colors)]

                # If not enough data for KDE (less than 2 points), skip the curve
                if len(df_sub) < 2: continue

                if i == j:  # DIAGONAL: Smooth KDE curves
                    # Calculate KDE
                    x_range = np.linspace(data_summary[x_col].min(), data_summary[x_col].max(), 100)
                    try:
                        kde = gaussian_kde(df_sub[x_col])
                        y_kde = kde(x_range)

                        fig.add_trace(
                            go.Scatter(
                                x=x_range, y=y_kde,
                                name=cat, line=dict(color=color, width=2),
                                fill='tozeroy', opacity=0.3,  # Fill under the curve
                                showlegend=(i == 0 and j == 0),
                                legendgroup=cat
                            ),
                            row=i + 1, col=j + 1
                        )
                    except:
                        pass # In case of zero variance

                else:  # OFF-DIAGONAL: Scatter plots
                    fig.add_trace(
                        go.Scatter(
                            x=df_sub[x_col], y=df_sub[y_col],
                            mode='markers', name=cat, marker_color=color,
                            opacity=0.5, marker_size=4,
                            showlegend=False, legendgroup=cat,
                            hovertext=df_sub['vehicle_id'].apply(lambda x: f"ID: {x}")
                        ),
                        row=i + 1, col=j + 1
                    )

    # 4. Configure fixed axes
    for i, col in enumerate(dims):
        margin = (data_summary[col].max() - data_summary[col].min()) * 0.05
        r = [data_summary[col].min() - margin, data_summary[col].max() + margin]

        for k in range(1, n + 1):
            fig.update_xaxes(range=r, row=k, col=i + 1)
            if i != k - 1:  # Don't touch Y axis for diagonal, as it has density scale
                fig.update_yaxes(range=r, row=i + 1, col=k)

    fig.update_layout(
        title_text="Classification Plot: Viable Vehicles (Perfect + Partial)",
        width=1200, height=1100,
        template="plotly_white"
    )
    fig.write_html(plots_dir + 'int_dimension_by_category.html')
    if show_plot:
        fig.show()

def plot_size_distribution(data_summary, show_plot=False):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Size mean distribution
    axes[0, 0].hist(data_summary['size_mean'], bins=30, alpha=0.7, edgecolor='black', color='#636EFA')
    axes[0, 0].set_xlabel('Size Mean (pixels²)', fontsize=10)
    axes[0, 0].set_ylabel('Frequency', fontsize=10)
    axes[0, 0].set_title('Distribution of Vehicle Area', fontsize=11, fontweight='bold')
    median_size = data_summary['size_mean'].median()
    axes[0, 0].axvline(median_size, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_size:.0f}')
    axes[0, 0].axvline(data_summary['size_mean'].quantile(0.65), color='orange',
                       linestyle='--', linewidth=2,
                       label=f'65th %ile: {data_summary["size_mean"].quantile(0.65):.0f}')
    axes[0, 0].legend(fontsize=9)
    axes[0, 0].grid(True, alpha=0.3)

    # Width distribution
    axes[0, 1].hist(data_summary['w_mean'], bins=30, alpha=0.7, edgecolor='black', color='#00CC96')
    axes[0, 1].set_xlabel('Width Mean (pixels)', fontsize=10)
    axes[0, 1].set_ylabel('Frequency', fontsize=10)
    axes[0, 1].set_title('Distribution of Vehicle Width', fontsize=11, fontweight='bold')
    median_w = data_summary['w_mean'].median()
    axes[0, 1].axvline(median_w, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_w:.1f}')
    axes[0, 1].axvline(data_summary['w_mean'].quantile(0.65), color='orange',
                       linestyle='--', linewidth=2,
                       label=f'65th %ile: {data_summary["w_mean"].quantile(0.65):.1f}')
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(True, alpha=0.3)

    # Height distribution
    axes[1, 0].hist(data_summary['h_mean'], bins=30, alpha=0.7, edgecolor='black', color='#AB63FA')
    axes[1, 0].set_xlabel('Height Mean (pixels)', fontsize=10)
    axes[1, 0].set_ylabel('Frequency', fontsize=10)
    axes[1, 0].set_title('Distribution of Vehicle Height', fontsize=11, fontweight='bold')
    median_h = data_summary['h_mean'].median()
    axes[1, 0].axvline(median_h, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_h:.1f}')
    axes[1, 0].axvline(data_summary['h_mean'].quantile(0.60), color='orange',
                       linestyle='--', linewidth=2,
                       label=f'60th %ile: {data_summary["h_mean"].quantile(0.60):.1f}')
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(True, alpha=0.3)

    # Aspect ratio
    h_w_mean_ratio = data_summary['h_mean'] / data_summary['w_mean']
    axes[1, 1].hist(h_w_mean_ratio, bins=30, alpha=0.7, edgecolor='black', color='#FFA15A')
    axes[1, 1].set_xlabel('Aspect Ratio (height/width)', fontsize=10)
    axes[1, 1].set_ylabel('Frequency', fontsize=10)
    axes[1, 1].set_title('Distribution of Aspect Ratio', fontsize=11, fontweight='bold')
    median_ar = h_w_mean_ratio.median()
    axes[1, 1].axvline(median_ar, color='red', linestyle='--', linewidth=2,
                       label=f'Median: {median_ar:.2f}')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plots_dir + 'size_distribution.png')
    if show_plot:
        plt.show()

    # Print statistics
    print("=" * 70)
    print("VEHICLE SIZE STATISTICS (Viable Vehicles Only)")
    print("=" * 70)
    print("\nDescriptive statistics:")
    print(data_summary[['w_mean', 'h_mean', 'size_mean']].describe().round(2))
    print("\nAspect ratio statistics:")
    print(h_w_mean_ratio.describe().round(3))
    print("\n" + "=" * 70)


def interactive_dimension_plot_by_class(data_summary, show_plot=False, day_only=False, night_only=False):
    """
    Creates an interactive scatter matrix plot showing vehicle dimensions by class.

    Parameters:
    -----------
    data_summary : pd.DataFrame
        DataFrame with vehicle metrics including classification information
        Must contain columns: 'w_mean', 'h_mean', 'size_mean', 'duration', 'h_w_mean_ratio', 'Class', 'vehicle_id'
        Optional column: 'solar_phase' (required if day_only or night_only is True)

    day_only : bool, default=False
        If True, shows only vehicles during day (solar_phase == 'Day')

    night_only : bool, default=False
        If True, shows only vehicles during night (solar_phase == 'Night')

    Note: If both day_only and night_only are False, all vehicles are shown.
          If both are True, a warning is printed and all vehicles are shown.

    Returns:
    --------
    None
        Displays interactive plotly figure
    """
    # Create copy to avoid modifying original
    df = data_summary.copy()

    # Handle filtering by solar phase
    filter_label = "All"
    if day_only and night_only:
        print("Warning: Both day_only and night_only are True. Showing all vehicles.")
    elif day_only:
        if 'solar_phase' not in df.columns:
            raise ValueError("Column 'solar_phase' is required when day_only=True")
        df = df[df['solar_phase'] == 'Day']
        filter_label = "Day Only"
    elif night_only:
        if 'solar_phase' not in df.columns:
            raise ValueError("Column 'solar_phase' is required when night_only=True")
        df = df[df['solar_phase'] == 'Night']
        filter_label = "Night Only"

    if len(df) == 0:
        print(f"No data available for the selected filter ({filter_label}).")
        return

    # 1. Parameters
    dims = ['w_mean', 'h_mean', 'size_mean', 'duration', 'h_w_mean_ratio']
    # Define fixed order and colors for consistency
    class_colors = {'Car': '#636EFA', 'Truck': '#EF553B'}
    classes = [cls for cls in ['Car', 'Truck'] if cls in df['Class'].unique()]
    n = len(dims)

    if len(classes) == 0:
        print("No valid classes found in data. Expected 'Car' or 'Truck' in 'Class' column.")
        return

    # 2. Create subplot grid
    fig = make_subplots(
        rows=n, cols=n,
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.03, vertical_spacing=0.03,
        column_titles=dims, row_titles=dims
    )

    # 3. Fill the matrix
    for i, y_col in enumerate(dims):
        for j, x_col in enumerate(dims):
            for cls in classes:
                df_sub = df[df['Class'] == cls]
                color = class_colors[cls]

                # If not enough data for KDE (less than 2 points), skip the curve
                if len(df_sub) < 2: continue

                if i == j:  # DIAGONAL: Smooth KDE curves
                    # Calculate KDE
                    x_range = np.linspace(df[x_col].min(),
                                          df[x_col].max(), 100)
                    try:
                        kde = gaussian_kde(df_sub[x_col])
                        y_kde = kde(x_range)

                        fig.add_trace(
                            go.Scatter(
                                x=x_range, y=y_kde,
                                name=cls, line=dict(color=color, width=2),
                                fill='tozeroy', opacity=0.3,  # Fill under the curve
                                showlegend=(i == 0 and j == 0),
                                legendgroup=cls
                            ),
                            row=i + 1, col=j + 1
                        )
                    except:
                        pass  # In case of zero variance

                else:  # OFF-DIAGONAL: Scatter plots
                    fig.add_trace(
                        go.Scatter(
                            x=df_sub[x_col], y=df_sub[y_col],
                            mode='markers', name=cls, marker_color=color,
                            opacity=0.5, marker_size=4,
                            showlegend=False, legendgroup=cls,
                            hovertext=df_sub['vehicle_id'].apply(lambda x: f"ID: {x}")
                        ),
                        row=i + 1, col=j + 1
                    )

    # 4. Configure fixed axes
    for i, col in enumerate(dims):
        margin = (df[col].max() - df[col].min()) * 0.05
        r = [df[col].min() - margin,
             df[col].max() + margin]

        for k in range(1, n + 1):
            fig.update_xaxes(range=r, row=k, col=i + 1)
            if i != k - 1:  # Don't touch Y axis for diagonal, as it has density scale
                fig.update_yaxes(range=r, row=i + 1, col=k)

    fig.update_layout(
        title_text=f"Interactive Plot: Vehicle Classification by Type ({filter_label})",
        width=1200, height=1100,
        template="plotly_white"
    )
    fig.write_html(plots_dir + 'int_dimension_by_class.html')
    if show_plot:
        fig.show()

def speed_distribution_over_time_plot(data_summary, show_plot=False, km_h=False):
    """
    Plots velocity_y distribution over time.

    Parameters:
    -----------
    data_summary : pd.DataFrame
        DataFrame with vehicle metrics including velocity and time information
        Must contain columns: 't_start', 'velocity_y_px_seconds', 'velocity_y_km_h'
        Optional columns: 'Class', 'solar_phase' for additional visualizations

    km_h : bool, default=False
        If False, plots velocity_y_px_seconds (pixels/second)
        If True, plots velocity_y_km_h (kilometers/hour)

    Returns:
    --------
    None
        Displays matplotlib plots
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    # Select velocity column based on parameter
    speed = 'velocity_y_px_seconds'
    speed_label = 'Velocity Y (pixels/s)'
    if km_h:
        speed = 'velocity_y_km_h'
        speed_label = 'Velocity Y (km/h)'

    # Create copy to avoid modifying original
    df = data_summary.copy()

    # Ensure t_start is datetime
    df['t_start'] = pd.to_datetime(df['t_start'])

    # Sort by time
    df = df.sort_values('t_start')

    # Create figure with subplots
    has_class = 'Class' in df.columns
    has_solar = 'solar_phase' in df.columns

    if has_class and has_solar:
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    elif has_class or has_solar:
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(14, 6))
        axes = [axes]  # Make it iterable

    # Plot 1: Overall velocity distribution over time
    ax1 = axes[0]
    scatter = ax1.scatter(df['t_start'], df[speed], alpha=0.6, s=30, c='#636EFA', edgecolors='white', linewidth=0.5)
    ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
    ax1.set_ylabel(speed_label, fontsize=12, fontweight='bold')
    ax1.set_title(f'Velocity Distribution Over Time - All Vehicles', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Add data gap shading for script errors
    add_data_gap_shading(ax1, data_summary)

    # Format x-axis as time
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    # Add mean line
    mean_speed = df[speed].mean()
    ax1.axhline(mean_speed, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_speed:.2f}')
    ax1.legend(fontsize=10)

    # Plot 2: By Class (if available)
    if has_class:
        ax2 = axes[1]
        cars = df[df['Class'] == 'Car']
        trucks = df[df['Class'] == 'Truck']

        ax2.scatter(cars['t_start'], cars[speed], alpha=0.6, s=30, c='#636EFA',
                   edgecolors='white', linewidth=0.5, label='Car')
        ax2.scatter(trucks['t_start'], trucks[speed], alpha=0.6, s=30, c='#EF553B',
                   edgecolors='white', linewidth=0.5, label='Truck')

        ax2.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax2.set_ylabel(speed_label, fontsize=12, fontweight='bold')
        ax2.set_title(f'Velocity Distribution Over Time - By Vehicle Class', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)

        # Add data gap shading for script errors
        add_data_gap_shading(ax2, data_summary, label_first=False)

        ax2.legend(fontsize=10)

        # Format x-axis
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add mean lines
        mean_cars = cars[speed].mean()
        mean_trucks = trucks[speed].mean()
        ax2.axhline(mean_cars, color='#636EFA', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Car mean: {mean_cars:.2f}')
        ax2.axhline(mean_trucks, color='#EF553B', linestyle='--', linewidth=1.5,
                   alpha=0.7, label=f'Truck mean: {mean_trucks:.2f}')
        ax2.legend(fontsize=10)

    # Plot 3: By Solar Phase (if available)
    if has_solar:
        ax_idx = 2 if has_class else 1
        ax3 = axes[ax_idx]

        for phase in df['solar_phase'].unique():
            phase_data = df[df['solar_phase'] == phase]
            color = '#FFA500' if phase == 'Day' else '#4B0082'
            ax3.scatter(phase_data['t_start'], phase_data[speed], alpha=0.6, s=30,
                       c=color, edgecolors='white', linewidth=0.5, label=phase)

        ax3.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax3.set_ylabel(speed_label, fontsize=12, fontweight='bold')
        ax3.set_title(f'Velocity Distribution Over Time - By Solar Phase', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # Add data gap shading for script errors
        add_data_gap_shading(ax3, data_summary, label_first=False)

        ax3.legend(fontsize=10)

        # Format x-axis
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax3.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')

        # Add mean lines for each phase
        for phase in df['solar_phase'].unique():
            phase_data = df[df['solar_phase'] == phase]
            mean_phase = phase_data[speed].mean()
            color = '#FFA500' if phase == 'Day' else '#4B0082'
            ax3.axhline(mean_phase, color=color, linestyle='--', linewidth=1.5,
                       alpha=0.7, label=f'{phase} mean: {mean_phase:.2f}')
        ax3.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig(plots_dir + 'speed_distribution_over_time.png')
    if show_plot:
        plt.show()

    # Print statistics
    print("=" * 70)
    print(f"VELOCITY DISTRIBUTION STATISTICS ({speed_label})")
    print("=" * 70)
    print(f"\nOverall statistics:")
    print(df[speed].describe().round(2))

    if has_class:
        print(f"\nStatistics by Class:")
        print(df.groupby('Class')[speed].describe().round(2))

    if has_solar:
        print(f"\nStatistics by Solar Phase:")
        print(df.groupby('solar_phase')[speed].describe().round(2))

    print("\n" + "=" * 70)


def vehicle_count_over_time_histogram(data_summary, show_plot=False):
    """
    Plots a 2x2 matrix of histograms showing vehicle counts per day.
    Matrix layout: Day/Night (rows) x Car/Truck (columns)

    Parameters:
    -----------
    data_summary : pd.DataFrame
        DataFrame with vehicle metrics including time and classification information
        Must contain columns: 't_start', 'Class', 'solar_phase'

    Returns:
    --------
    None
        Displays matplotlib histogram matrix
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Validate required columns
    required_cols = ['t_start', 'Class', 'solar_phase']
    missing_cols = [col for col in required_cols if col not in data_summary.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Create copy to avoid modifying original
    df = data_summary.copy()

    # Ensure t_start is datetime
    df['t_start'] = pd.to_datetime(df['t_start'])

    # Extract date for grouping by day
    df['date'] = df['t_start'].dt.date

    # Sort by time
    df = df.sort_values('t_start')

    # Create 2x2 figure matrix
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Define combinations: (row, col) -> (solar_phase, Class)
    combinations = [
        (0, 0, 'Day', 'Car', '#636EFA'),
        (0, 1, 'Day', 'Truck', '#EF553B'),
        (1, 0, 'Night', 'Car', '#636EFA'),
        (1, 1, 'Night', 'Truck', '#EF553B')
    ]

    # Get unique dates for x-axis (sorted descending so highest day is on top)
    unique_dates = sorted(df['date'].unique(), reverse=True)

    # Calculate fixed y-axis limit (max across all combinations + 10% margin)
    all_counts = []
    for _, _, phase, cls, _ in combinations:
        subset = df[(df['solar_phase'] == phase) & (df['Class'] == cls)]
        if len(subset) > 0:
            day_counts = subset.groupby('date').size().reindex(unique_dates, fill_value=0)
            all_counts.append(day_counts.max())
    y_max = max(all_counts) if all_counts else 1
    y_limit = y_max * 1.1

    for idx, (row, col, phase, cls, color) in enumerate(combinations):
        ax = axes[row, col]

        # Add gap indicators first (so they appear behind bars) - horizontal mode
        add_gap_indicators_to_bar_chart(ax, unique_dates, data_summary, label_first=(idx == 0), horizontal=True)

        # Filter data for this combination
        subset = df[(df['solar_phase'] == phase) & (df['Class'] == cls)]

        # Count vehicles per day
        if len(subset) > 0:
            day_counts = subset.groupby('date').size()
            # Reindex to include all dates (fill missing with 0)
            day_counts = day_counts.reindex(unique_dates, fill_value=0)

            ax.barh(range(len(unique_dates)), day_counts.values, alpha=0.7,
                    color=color, edgecolor='black', height=0.6)
        else:
            ax.barh(range(len(unique_dates)), [0] * len(unique_dates), alpha=0.7,
                    color=color, edgecolor='black', height=0.6)

        ax.set_ylabel('Date', fontsize=10, fontweight='bold')
        ax.set_xlabel('Vehicle Count', fontsize=10, fontweight='bold')
        ax.set_title(f'{phase} - {cls}', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(unique_dates)))
        ax.set_yticklabels([d.strftime('%d') for d in unique_dates])
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, y_limit)

        # Format x-axis to show k for thousands
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))

        # Add count label to the right of each bar
        if len(subset) > 0:
            for i, count in enumerate(day_counts.values):
                if count > 0:
                    label = f'{int(count/1000)}k' if count >= 1000 else str(count)
                    ax.text(count + y_limit * 0.01, i, label, ha='left', va='center',
                            fontsize=9, fontweight='bold', color='black')

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='lower right')

    plt.suptitle('Vehicle Count Per Day by Solar Phase and Class', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir + 'vehicle_count_over_time.png')
    if show_plot:
        plt.show()

    # Print statistics
    print("=" * 70)
    print("VEHICLE COUNT STATISTICS")
    print("=" * 70)
    print(f"\nTotal vehicles: {len(df)}")

    # Count by solar phase and class
    print(f"\nCount by Solar Phase and Class:")
    phase_class_counts = df.groupby(['solar_phase', 'Class']).size().unstack(fill_value=0)
    print(phase_class_counts)

    print(f"\nCount per day by Solar Phase and Class:")
    day_phase_class = df.groupby(['date', 'solar_phase', 'Class']).size().unstack(fill_value=0)
    print(day_phase_class)

    print("\n" + "=" * 70)


def average_speed_over_time_plot(data_summary, show_plot=False, speed_limit_kmh=20, by_class=False):
    """
    Plots average speed over the entire data collection timespan with a speed limit reference line.

    Parameters:
    -----------
    data_summary : pd.DataFrame
        DataFrame with vehicle metrics including velocity and time information
        Must contain columns: 't_start', 'velocity_y_km_h'
        Optional column: 'Class' (required if by_class=True)

    speed_limit_kmh : float, default=30
        Speed limit in km/h to display as a reference line

    by_class : bool, default=False
        If True, shows separate lines for Cars and Trucks

    Returns:
    --------
    None
        Displays matplotlib line plot
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create copy to avoid modifying original
    df = data_summary.copy()

    # Ensure t_start is datetime
    df['t_start'] = pd.to_datetime(df['t_start'])

    # Extract date for grouping by day
    df['date'] = df['t_start'].dt.date

    # Sort by time
    df = df.sort_values('t_start')

    # Get unique dates for x-axis
    unique_dates = sorted(df['date'].unique())

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 7))

    if by_class and 'Class' in df.columns:
        # Calculate average speed per day for each class
        cars = df[df['Class'] == 'Car']
        trucks = df[df['Class'] == 'Truck']

        cars_avg = cars.groupby('date')['velocity_y_km_h'].mean().reindex(unique_dates)
        trucks_avg = trucks.groupby('date')['velocity_y_km_h'].mean().reindex(unique_dates)

        # Plot lines for each class
        ax.plot(range(len(unique_dates)), cars_avg.values, marker='o', linewidth=2,
                markersize=8, color='#636EFA', label='Car (avg)')
        ax.plot(range(len(unique_dates)), trucks_avg.values, marker='s', linewidth=2,
                markersize=8, color='#EF553B', label='Truck (avg)')

        # Fill area between lines and speed limit
        ax.fill_between(range(len(unique_dates)), cars_avg.values, speed_limit_kmh,
                       where=(cars_avg.values > speed_limit_kmh),
                       alpha=0.2, color='#636EFA', interpolate=True)
        ax.fill_between(range(len(unique_dates)), trucks_avg.values, speed_limit_kmh,
                       where=(trucks_avg.values > speed_limit_kmh),
                       alpha=0.2, color='#EF553B', interpolate=True)
    else:
        # Calculate overall average speed per day
        avg_speed = df.groupby('date')['velocity_y_km_h'].mean().reindex(unique_dates)

        # Plot line
        ax.plot(range(len(unique_dates)), avg_speed.values, marker='o', linewidth=2.5,
                markersize=10, color='#636EFA', label='Average Speed')

        # Fill area above speed limit
        ax.fill_between(range(len(unique_dates)), avg_speed.values, speed_limit_kmh,
                       where=(avg_speed.values > speed_limit_kmh),
                       alpha=0.3, color='red', label='Above Limit',
                       interpolate=True)

    # Add speed limit line
    ax.axhline(speed_limit_kmh, color='red', linestyle='--', linewidth=2.5,
               label=f'Speed Limit ({speed_limit_kmh} km/h)')

    # Add gap indicators for dates affected by script errors
    add_gap_indicators_to_bar_chart(ax, unique_dates, data_summary)

    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Speed (km/h)', fontsize=12, fontweight='bold')
    ax.set_title('Average Vehicle Speed Over Data Collection Period', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(unique_dates)))
    ax.set_xticklabels([d.strftime('%d') for d in unique_dates], rotation=45, ha='right')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Set y-axis to start from 0
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(plots_dir + 'avg_speed_over_time.png')
    if show_plot:
        plt.show()

    # Print statistics
    print("=" * 70)
    print("AVERAGE SPEED STATISTICS")
    print("=" * 70)
    print(f"\nSpeed limit: {speed_limit_kmh} km/h")
    print(f"\nData collection period: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"\nOverall average speed: {df['velocity_y_km_h'].mean():.2f} km/h")
    print(f"Overall median speed:  {df['velocity_y_km_h'].median():.2f} km/h")
    print(f"Overall max speed:     {df['velocity_y_km_h'].max():.2f} km/h")
    print(f"Overall min speed:     {df['velocity_y_km_h'].min():.2f} km/h")

    # Count vehicles above speed limit
    above_limit = df[df['velocity_y_km_h'] > speed_limit_kmh]
    pct_above = (len(above_limit) / len(df) * 100) if len(df) > 0 else 0
    print(f"\nVehicles above speed limit: {len(above_limit)} ({pct_above:.1f}%)")

    if by_class and 'Class' in df.columns:
        print(f"\nAverage speed by class:")
        class_stats = df.groupby('Class')['velocity_y_km_h'].agg(['mean', 'median', 'max', 'min']).round(2)
        print(class_stats)

        print(f"\nVehicles above speed limit by class:")
        for cls in ['Car', 'Truck']:
            cls_data = df[df['Class'] == cls]
            cls_above = cls_data[cls_data['velocity_y_km_h'] > speed_limit_kmh]
            cls_pct = (len(cls_above) / len(cls_data) * 100) if len(cls_data) > 0 else 0
            print(f"  {cls}: {len(cls_above)} ({cls_pct:.1f}%)")

    print(f"\nAverage speed by date:")
    date_avg = df.groupby('date')['velocity_y_km_h'].mean().round(2)
    print(date_avg)

    print("\n" + "=" * 70)


def average_speed_by_weekday_and_hour(final_summary, show_plot=False, speed_limit_kmh=20):
    """
    Creates a grid of line plots showing average speed by hour of day, one subplot per weekday.

    This allows reading like: "The average speed on Mondays at 14:00 is X km/h"

    Parameters:
    -----------
    final_summary : pd.DataFrame
        DataFrame with vehicle metrics over multiple weeks
        Must contain columns: 't_start', 'velocity_y_km_h'

    speed_limit_kmh : float, default=30
        Speed limit in km/h to display as a reference line

    Returns:
    --------
    None
        Displays matplotlib grid of line plots
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create copy to avoid modifying original
    df = final_summary.copy()

    # Ensure t_start is datetime
    df['t_start'] = pd.to_datetime(df['t_start'])

    # Extract day of week and hour
    df['day_of_week'] = df['t_start'].dt.day_name()
    df['hour'] = df['t_start'].dt.hour

    # Define day order for proper sorting
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

    # Calculate average speed per day of week and hour
    avg_speed = df.groupby(['day_of_week', 'hour'])['velocity_y_km_h'].mean().unstack(level=0)

    # Reorder columns by day of week
    avg_speed = avg_speed[[day for day in day_order if day in avg_speed.columns]]

    # Color palette for days of the week
    colors = {
        'Monday': '#636EFA',
        'Tuesday': '#EF553B',
        'Wednesday': '#00CC96',
        'Thursday': '#AB63FA',
        'Friday': '#FFA15A',
        'Saturday': '#19D3F3',
        'Sunday': '#FF6692'
    }

    # Create grid of subplots (2 rows x 4 cols, last cell empty)
    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes = axes.flatten()

    # Plot each day in its own subplot
    for idx, day in enumerate(day_order):
        ax = axes[idx]

        if day in avg_speed.columns:
            day_data = avg_speed[day].dropna()
            color = colors.get(day, '#636EFA')

            # Plot the line
            ax.plot(day_data.index, day_data.values, marker='o', linewidth=2,
                    markersize=5, color=color, alpha=0.8)

            # Fill area above speed limit
            ax.fill_between(day_data.index, day_data.values, speed_limit_kmh,
                           where=(day_data.values > speed_limit_kmh),
                           alpha=0.3, color='red', interpolate=True)

        # Add speed limit line
        ax.axhline(speed_limit_kmh, color='red', linestyle='--', linewidth=1.5,
                   label=f'Limit: {speed_limit_kmh} km/h')

        # Configure axes
        ax.set_title(day, fontsize=12, fontweight='bold', color=colors.get(day, '#000000'))
        ax.set_xlabel('Hour', fontsize=10)
        ax.set_ylabel('Speed (km/h)', fontsize=10)
        ax.set_xticks([0, 6, 12, 18, 23])
        ax.set_xticklabels(['00', '06', '12', '18', '23'])
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 50)
        ax.set_xlim(-0.5, 23.5)

        # Add legend only to first subplot
        if idx == 0:
            ax.legend(fontsize=8, loc='upper right')

    # Hide the last (8th) subplot since we only have 7 days
    axes[7].axis('off')

    plt.suptitle('Average Speed by Hour for Each Day of the Week', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(plots_dir + 'avg_speed_by_weekday_hour.png')
    if show_plot:
        plt.show()

    # Print statistics table
    print("=" * 70)
    print("AVERAGE SPEED BY DAY OF WEEK AND HOUR (km/h)")
    print("=" * 70)
    print("\nPivot Table (rows=hour, columns=day of week):")
    print(avg_speed.round(2).to_string())

    print("\n" + "-" * 70)
    print("Summary by Day of Week:")
    print("-" * 70)
    for day in day_order:
        if day in df['day_of_week'].values:
            day_data = df[df['day_of_week'] == day]['velocity_y_km_h']
            print(f"{day:12s}: mean={day_data.mean():.2f}, median={day_data.median():.2f}, "
                  f"max={day_data.max():.2f}, count={len(day_data)}")

    print("\n" + "=" * 70)


def speeding_vehicles_histogram(final_summary, show_plot=False, speed_limit_kmh=20):
    """
    Creates three histograms showing vehicle counts by speed category per day:
    1. Speeding vehicles (> speed_limit_kmh)
    2. Slow vehicles (10-20 km/h)
    3. Very slow vehicles (< 10 km/h)

    Parameters:
    -----------
    final_summary : pd.DataFrame
        DataFrame with vehicle metrics over the data collection period
        Must contain columns: 't_start', 'velocity_y_km_h'

    speed_limit_kmh : float, default=20
        Speed limit in km/h to use as threshold for speeding

    Returns:
    --------
    None
        Displays matplotlib histograms
    """
    import pandas as pd
    import matplotlib.pyplot as plt

    # Create copy to avoid modifying original
    df = final_summary.copy()

    # Ensure t_start is datetime
    df['t_start'] = pd.to_datetime(df['t_start'])

    # Extract date for grouping
    df['date'] = df['t_start'].dt.date

    # Filter vehicles by speed categories
    speeding = df[df['velocity_y_km_h'] > speed_limit_kmh]
    slow = df[(df['velocity_y_km_h'] > 10) & (df['velocity_y_km_h'] <= 20)]
    very_slow = df[df['velocity_y_km_h'] <= 10]

    # Get all unique dates (sorted descending so highest day is on top)
    all_dates = sorted(df['date'].unique(), reverse=True)

    # Count vehicles per day for each category
    speeding_per_day = speeding.groupby('date').size().reindex(all_dates, fill_value=0)
    slow_per_day = slow.groupby('date').size().reindex(all_dates, fill_value=0)
    very_slow_per_day = very_slow.groupby('date').size().reindex(all_dates, fill_value=0)

    # Define plot configurations
    plots = [
        (speeding_per_day, f'Speeding Vehicles (>{speed_limit_kmh} km/h)', '#EF553B', 'speeding_vehicles.png'),
        (slow_per_day, 'Slow Vehicles (10-20 km/h)', '#FFA15A', 'slow_vehicles.png'),
        (very_slow_per_day, 'Very Slow Vehicles (<10 km/h)', '#636EFA', 'very_slow_vehicles.png'),
    ]

    # Calculate shared x-axis limit across all categories (extra room for labels)
    x_max = max(speeding_per_day.max(), slow_per_day.max(), very_slow_per_day.max()) * 1.15
    if x_max == 0:
        x_max = 1

    for data, title, color, filename in plots:
        # Create separate figure for each plot
        fig, ax = plt.subplots(figsize=(14, 8))

        # Add gap indicators first (so they appear behind bars) - horizontal mode
        add_gap_indicators_to_bar_chart(ax, all_dates, final_summary, label_first=True, horizontal=True)

        # Create horizontal bar chart with more space between bars
        ax.barh(range(len(all_dates)), data.values, color=color, edgecolor='black', alpha=0.8, height=0.6)

        # Add count labels to the right of bars (format thousands as k)
        for i, count in enumerate(data.values):
            if count > 0:
                label = f'{int(count/1000)}k' if count >= 1000 else str(count)
                ax.text(count + x_max * 0.01, i, label, ha='left', va='center',
                        fontsize=11, fontweight='bold', color='black')

        # Configure axes
        ax.set_ylabel('Date', fontsize=11, fontweight='bold')
        ax.set_xlabel('Number of Vehicles', fontsize=11, fontweight='bold')
        ax.set_title(f'{title} Per Day', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(all_dates)))
        ax.set_yticklabels([d.strftime('%d') for d in all_dates])
        ax.set_ylim(-0.5, len(all_dates) - 0.5)
        ax.grid(True, alpha=0.3, axis='x')
        ax.set_xlim(0, x_max)

        # Format x-axis to show k for thousands
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x/1000)}k' if x >= 1000 else f'{int(x)}'))

        # Add mean line (vertical now)
        mean_val = data.mean()
        ax.axvline(mean_val, color='blue', linestyle='--', linewidth=2,
                   label=f'Daily Average: {mean_val:.1f}')
        ax.legend(fontsize=10, loc='lower right')

        plt.tight_layout()
        plt.savefig(plots_dir + filename)
        if show_plot:
            plt.show()
        plt.close(fig)

    # Print statistics
    print("=" * 70)
    print("VEHICLE SPEED CATEGORY STATISTICS")
    print("=" * 70)
    print(f"\nData collection period: {all_dates[0]} to {all_dates[-1]}")
    print(f"Total days: {len(all_dates)}")
    print(f"Total vehicles: {len(df)}")

    categories = [
        (f'Speeding (>{speed_limit_kmh} km/h)', speeding, speeding_per_day),
        ('Slow (10-20 km/h)', slow, slow_per_day),
        ('Very Slow (<10 km/h)', very_slow, very_slow_per_day),
    ]

    for name, filtered_df, per_day in categories:
        pct = (len(filtered_df) / len(df) * 100) if len(df) > 0 else 0
        print(f"\n{name}:")
        print(f"  Total: {len(filtered_df)} ({pct:.1f}%)")
        print(f"  Daily average: {per_day.mean():.1f}")
        if len(per_day) > 0 and per_day.max() > 0:
            print(f"  Maximum: {per_day.max()} (on {per_day.idxmax()})")
            print(f"  Minimum: {per_day.min()} (on {per_day.idxmin()})")

    print("\n" + "=" * 70)
