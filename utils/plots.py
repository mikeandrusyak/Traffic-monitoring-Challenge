import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

def visualize_vehicle_trajectories(df, session_id=0, max_vehicles=25, min_records=3, 
                                   figsize_per_plot=3, col_wrap=5, title=None, category=None):
    """
Visualises vehicle trajectories with frames and transparency gradient.
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with vehicle data
    session_id : int
        Session ID for analysis (default: 0)
    max_vehicles : int
        Maximum number of vehicle_ids to visualise (default: 25)
    min_records : int
        Minimum number of records for vehicle_id (default: 3)
    figsize_per_plot : int
        Height of each subplot (default: 3)
    col_wrap : int
        Number of columns in the grid (default: 5)
    title : str
        Graph title (if None, default is used)
    category : str
        Category subtitle (if provided, displayed below the title)
    """
    
    # Style settings
    sns.set_theme(style="whitegrid")

    # Limiting the number of IDs for visualisation
    ids_to_plot = df['vehicle_id'].unique()[:max_vehicles]
    plot_data = df[df['vehicle_id'].isin(ids_to_plot)]
    
    # Function for drawing with frames
    def plot_trajectory(data, **kwargs):
        x = data['x']
        y = data['y']
        width = data['width']
        height = data['heigth']
        ax = plt.gca()
        
        # Trajectory line
        ax.plot(x, y, color='gray', alpha=0.3, linewidth=1)
        
        # Number of intermediate points
        n_middle = len(x) - 2
        
        # Draw frames for each position
        for i in range(len(x)):
            if i == 0 or i == len(x) - 1:
                continue
            else:
                # Intermediate frames with transparency gradient (from 0% to 70%)
                if n_middle > 1:
                    alpha_value = 0.7 * (i - 1) / (n_middle - 1)
                else:
                    alpha_value = 0.35
                
                rect = Rectangle(
                    (x.iloc[i], y.iloc[i]),
                    width.iloc[i], 
                    height.iloc[i],
                    linewidth=1,
                    edgecolor='blue',
                    facecolor='none',
                    alpha=alpha_value,
                    zorder=3
                )
                ax.add_patch(rect)
        
        # Draw a green frame (first) at the top
        rect_green = Rectangle(
            (x.iloc[0], y.iloc[0]),
            width.iloc[0], 
            height.iloc[0],
            linewidth=3,
            edgecolor='green',
            facecolor='none',
            alpha=0.9,
            zorder=5
        )
        ax.add_patch(rect_green)
        
        # Draw a red frame (the last one) at the top
        rect_red = Rectangle(
            (x.iloc[-1], y.iloc[-1]),
            width.iloc[-1], 
            height.iloc[-1],
            linewidth=3,
            edgecolor='red',
            facecolor='none',
            alpha=0.9,
            zorder=5
        )
        ax.add_patch(rect_red)
    
    # Creating a grid
    g = sns.FacetGrid(plot_data, col="vehicle_id", col_wrap=col_wrap, 
                      height=figsize_per_plot, sharex=False, sharey=True)
    
    # Using the drawing function
    g.map_dataframe(plot_trajectory)
    
    # Invert Y (0 - top, 300 - bottom for ROI)
    g.set(ylim=(300, -10), xlim=(-10, 260)) 
    
    g.set_axis_labels("Coordinate X", "Coordinate Y")
    
    # Add titles with timestamps
    for ax, vehicle_id in zip(g.axes.flat, ids_to_plot):
        vehicle_data = plot_data[plot_data['vehicle_id'] == vehicle_id]
        min_time = vehicle_data['date_time'].min().strftime('%H:%M:%S')
        max_time = vehicle_data['date_time'].max().strftime('%H:%M:%S')
        ax.set_title(f"Vehicle ID: {vehicle_id}\n{min_time} - {max_time}", fontsize=10)
    
    plt.subplots_adjust(top=0.9)
    
    # Set the title
    if title is None:
        title = f"Analysis of vehicle trajectories (Session {session_id})"
    
    # Add category as subtitle if provided
    if category:
        title = f"{title}\nCategory: {category}"
    
    g.fig.suptitle(title, fontsize=16)
    
    plt.show()
    
    return g


def visualize_vehicle_trajectories_points(df, session_id=0, max_vehicles=25, min_records=3, 
                                         figsize_per_plot=3, col_wrap=5, title=None):
    """
    Visualises vehicle trajectories with colored points (first, last, middle).
    
    Parameters:
    -----------
    df : DataFrame
        Input DataFrame with vehicle data
    session_id : int
        Session ID for analysis (default: 0)
    max_vehicles : int
        Maximum number of vehicle_ids to visualise (default: 25)
    min_records : int
        Minimum number of records for vehicle_id (default: 3)
    figsize_per_plot : int
        Height of each subplot (default: 3)
    col_wrap : int
        Number of columns in the grid (default: 5)
    title : str
        Graph title (if None, default is used)
    """
    
    # Style settings
    sns.set_theme(style="whitegrid")
    
    # Limiting the number of IDs for visualisation
    ids_to_plot = df['vehicle_id'].unique()[:max_vehicles]
    plot_data = df[df['vehicle_id'].isin(ids_to_plot)]
    
    # Function for drawing with different colors for first, last and middle points
    def plot_trajectory(data, **kwargs):
        x = data['x']
        y = data['y']
        ax = plt.gca()
        
        # Line
        ax.plot(x, y, color='gray', alpha=0.3, linewidth=1)
        
        # Middle points
        if len(x) > 2:
            ax.scatter(x.iloc[1:-1], y.iloc[1:-1], color='blue', s=30, alpha=0.7, zorder=3)
        
        # First point (green)
        ax.scatter(x.iloc[0], y.iloc[0], color='green', s=80, marker='o', zorder=4, label='Start')
        
        # Last point (red)
        ax.scatter(x.iloc[-1], y.iloc[-1], color='red', s=80, marker='s', zorder=4, label='End')
    
    # Creating a grid
    g = sns.FacetGrid(plot_data, col="vehicle_id", col_wrap=col_wrap, 
                      height=figsize_per_plot, sharex=False, sharey=True)
    
    # Using the drawing function
    g.map_dataframe(plot_trajectory)
    
    # Invert Y (0 - top, 300 - bottom for ROI)
    g.set(ylim=(300, -10), xlim=(-10, 260)) 
    
    g.set_axis_labels("Coordinate X", "Coordinate Y")
    
    # Add titles with timestamps
    for ax, vehicle_id in zip(g.axes.flat, ids_to_plot):
        vehicle_data = plot_data[plot_data['vehicle_id'] == vehicle_id]
        min_time = vehicle_data['date_time'].min().strftime('%H:%M:%S')
        max_time = vehicle_data['date_time'].max().strftime('%H:%M:%S')
        ax.set_title(f"Vehicle ID: {vehicle_id}\n{min_time} - {max_time}", fontsize=10)
    
    plt.subplots_adjust(top=0.9)
    
    # Set the title
    if title is None:
        title = f"Analysis of vehicle trajectories (Session {session_id})"
    g.fig.suptitle(title, fontsize=16)
    
    plt.show()
    
    return g

def plot_interactive_matrix(final_summary, dims=None, width=1200, height=1100, 
                           category_filter=None, max_points_per_category=None):
    """
    Creates an interactive matrix plot with KDE curves on diagonal and scatter plots on off-diagonal.
    
    Parameters:
    -----------
    final_summary : DataFrame
        Summary DataFrame with vehicle metrics and categories
    dims : list
        List of dimension names to plot (default: ['path_completeness', 'frames_count', 'movement_efficiency', 'w_cv', 'h_cv'])
    width : int
        Figure width in pixels (default: 1200)
    height : int
        Figure height in pixels (default: 1100)
    category_filter : list or None
        List of category names to display (default: None - all categories)
    max_points_per_category : int or None
        Maximum number of points to display per category (default: None - all points)
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The generated figure
    """
    
    # Default dimensions
    if dims is None:
        dims = ['path_completeness', 'frames_count', 'movement_efficiency', 'w_cv', 'h_cv']
    
    # Filter by categories if specified
    if category_filter is not None:
        final_summary = final_summary[final_summary['category'].isin(category_filter)]
    
    # Sample data if max_points_per_category is specified
    if max_points_per_category is not None:
        sampled_dfs = []
        for cat, group in final_summary.groupby('category'):
            sampled = group.sample(min(len(group), max_points_per_category), random_state=42)
            sampled_dfs.append(sampled)
        final_summary = pd.concat(sampled_dfs, ignore_index=True)
    
    categories = final_summary['category'].unique()
    n = len(dims)

    # Create subplot grid
    fig = make_subplots(
        rows=n, cols=n, 
        shared_xaxes=False, shared_yaxes=False,
        horizontal_spacing=0.03, vertical_spacing=0.03,
        column_titles=dims, row_titles=dims
    )

    # Color palette
    colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A']

    # Fill the matrix
    for i, y_col in enumerate(dims):
        for j, x_col in enumerate(dims):
            for k, cat in enumerate(categories):
                df_sub = final_summary[final_summary['category'] == cat]
                
                # If not enough data for KDE (less than 2 points), skip the curve
                if len(df_sub) < 2:
                    continue

                if i == j:  # DIAGONAL: Smooth KDE curves
                    # Calculate KDE
                    x_range = np.linspace(final_summary[x_col].min(), final_summary[x_col].max(), 100)
                    try:
                        kde = gaussian_kde(df_sub[x_col])
                        y_kde = kde(x_range)
                        
                        fig.add_trace(
                            go.Scatter(
                                x=x_range, y=y_kde, 
                                name=cat, line=dict(color=colors[k % len(colors)], width=2),
                                fill='tozeroy', opacity=0.3,  # Fill under the curve
                                showlegend=(i == 0 and j == 0),
                                legendgroup=cat
                            ),
                            row=i+1, col=j+1
                        )
                    except:
                        pass  # In case of zero variance

                else:  # OFF-DIAGONAL: Scatter plots
                    fig.add_trace(
                        go.Scatter(
                            x=df_sub[x_col], y=df_sub[y_col],
                            mode='markers', name=cat, marker_color=colors[k % len(colors)],
                            opacity=0.5, marker_size=4,
                            showlegend=False, legendgroup=cat,
                            hovertext=df_sub['vehicle_id'].apply(lambda x: f"ID: {x}")
                        ),
                        row=i+1, col=j+1
                    )

    # Configure fixed axes
    for i, col in enumerate(dims):
        margin = (final_summary[col].max() - final_summary[col].min()) * 0.05
        r = [final_summary[col].min() - margin, final_summary[col].max() + margin]
        
        for k in range(1, n + 1):
            fig.update_xaxes(range=r, row=k, col=i+1)
            if i != k-1:  # Don't touch Y axis for diagonal, as it has density scale
                fig.update_yaxes(range=r, row=i+1, col=k)

    fig.update_layout(
        title_text="Interactive matrix with distribution curves (KDE) on diagonal",
        width=width, height=height,
        template="plotly_white"
    )

    return fig

def visualize_merge_pairs_grid(merge_results, n_pairs=16, cols=4):
    """
    Visualize merge pairs as vectors in a grid layout
    Direction-aware: includes box height in arrow endpoints
    """
    # Add height columns if not present (for backwards compatibility)
    if 'old_h_start' not in merge_results.columns:
        merge_results = merge_results.copy()
        merge_results['old_h_start'] = 0
        merge_results['old_h_end'] = 0
        merge_results['new_h_start'] = 0
        merge_results['new_h_end'] = 0
    
    n_pairs = min(n_pairs, len(merge_results))
    rows = int(np.ceil(n_pairs / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx in range(n_pairs):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        pair = merge_results.iloc[idx]
        
        # Determine direction for old_id (left lane or right lane)
        old_moving_down = (pair['old_x_start'] + (pair.get('old_w_mean', 0) if 'old_w_mean' in pair else 0)) <= 140
        # Calculate arrow start/end for old_id
        old_arrow_start = pair['old_y_start']
        if old_moving_down:
            old_arrow_end = pair['old_y_end'] + pair.get('old_h_end', 0)
        else:
            old_arrow_start = pair['old_y_start'] + pair.get('old_h_start', 0)
            old_arrow_end = pair['old_y_end']
        
        # Determine direction for new_id
        new_moving_down = (pair['new_x_start'] + (pair.get('new_w_mean', 0) if 'new_w_mean' in pair else 0)) <= 140
        # Calculate arrow start/end for new_id
        new_arrow_start = pair['new_y_start']
        if new_moving_down:
            new_arrow_end = pair['new_y_end'] + pair.get('new_h_end', 0)
        else:
            new_arrow_start = pair['new_y_start'] + pair.get('new_h_start', 0)
            new_arrow_end = pair['new_y_end']
        
        # Plot old_id vector (blue)
        ax.arrow(pair['old_x_start'], old_arrow_start, 
                0, old_arrow_end - old_arrow_start,
                head_width=8, head_length=10, fc='blue', ec='blue', 
                linewidth=2, alpha=0.7, length_includes_head=True)
        
        # Plot new_id vector (red)
        ax.arrow(pair['new_x_start'], new_arrow_start,
                0, new_arrow_end - new_arrow_start,
                head_width=8, head_length=10, fc='red', ec='red',
                linewidth=2, alpha=0.7, length_includes_head=True)
        
        # Gap arrow (green dashed) - connect arrow endpoints
        ax.plot([pair['old_x_end'], pair['new_x_start']], 
               [old_arrow_end, new_arrow_start], 
               'g--', linewidth=1.5, alpha=0.6)
        
        # Add time labels with background for better readability
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
        
        # Old ID times (at arrow endpoints)
        ax.text(pair['old_x_start'] - 20, old_arrow_start, 
               pair['old_t_start'].strftime('%H:%M:%S'), 
               fontsize=9, color='blue', ha='right', va='center', 
               fontweight='bold', bbox=bbox_props)
        ax.text(pair['old_x_end'] - 20, old_arrow_end, 
               pair['old_t_end'].strftime('%H:%M:%S'), 
               fontsize=9, color='blue', ha='right', va='center',
               fontweight='bold', bbox=bbox_props)
        
        # New ID times (at arrow endpoints)
        ax.text(pair['new_x_start'] + 20, new_arrow_start, 
               pair['new_t_start'].strftime('%H:%M:%S'), 
               fontsize=9, color='red', ha='left', va='center',
               fontweight='bold', bbox=bbox_props)
        ax.text(pair['new_x_end'] + 20, new_arrow_end, 
               pair['new_t_end'].strftime('%H:%M:%S'), 
               fontsize=9, color='red', ha='left', va='center',
               fontweight='bold', bbox=bbox_props)
        
        # Title
        ax.set_title(f"{pair['old_id']}→{pair['new_id']} | {pair['gap_sec']}s", 
                    fontsize=9, fontweight='bold')
        
        # Fixed axes (assuming standard ROI)
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 300)
        ax.invert_yaxis()
    
    # Hide empty subplots
    for idx in range(n_pairs, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Merge Pairs - Vector Visualization ({n_pairs} pairs)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def visualize_consolidated_merges_grid(summary_df, n_merges=16, cols=4):
    """
    Visualize consolidated merged IDs as simple arrows (y_start → y_end at x_mean).
    Direction-aware: includes box height in arrow endpoints
    
    Parameters:
    -----------
    summary_df : pd.DataFrame
        Consolidated summary with y_start, y_end, x_mean (after apply_merges_to_summary)
    n_merges : int
        Number of records to visualize
    cols : int
        Number of columns in grid
    """
    records = summary_df.head(n_merges)
    n_merges = len(records)
    
    if n_merges == 0:
        print("No records to visualize")
        return None
    
    rows = int(np.ceil(n_merges / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*3.5, rows*3.5))
    if rows == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    for idx, (_, row) in enumerate(records.iterrows()):
        row_idx = idx // cols
        col_idx = idx % cols
        ax = axes[row_idx, col_idx]
        
        vehicle_ids = row['vehicle_id'] if isinstance(row['vehicle_id'], list) else [row['vehicle_id']]
        unified_id = row['unified_id']
        
        # Determine direction (left lane moving down, right lane moving up)
        moving_down = (row['x_mean'] + row['w_mean']) <= 140
        
        # Calculate arrow start/end including box height
        if moving_down:
            arrow_start = row['y_start']
            arrow_end = row['y_end'] + row['h_end']
            color = 'blue'
        else:
            arrow_start = row['y_start'] + row['h_start']
            arrow_end = row['y_end']
            color = 'orange'
        
        # Draw arrow from calculated start to end
        ax.arrow(row['x_mean'], arrow_start, 
                0, arrow_end - arrow_start,
                head_width=8, head_length=10, fc=color, ec=color, 
                linewidth=2.5, alpha=0.7, length_includes_head=True)
        
        # Add time labels at arrow endpoints
        bbox_props = dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.8)
        
        # Start time (at arrow start)
        ax.text(row['x_mean'] + 20, arrow_start, 
               row['t_start'].strftime('%H:%M:%S'), 
               fontsize=9, color=color, ha='left', va='center', 
               fontweight='bold', bbox=bbox_props)
        
        # End time (at arrow end)
        ax.text(row['x_mean'] + 20, arrow_end, 
               row['t_end'].strftime('%H:%M:%S'), 
               fontsize=9, color=color, ha='left', va='center',
               fontweight='bold', bbox=bbox_props)
        
        # Title with unified_id
        id_chain = ' → '.join(map(str, vehicle_ids))
        if pd.notna(unified_id):
            ax.set_title(f"unified_id={int(unified_id)}\n{id_chain}", 
                        fontsize=9, fontweight='bold')
        else:
            ax.set_title(f"{id_chain}", fontsize=9, fontweight='bold')
        
        # Info text
        info_text = f"{row['frames_count']} frames | {row['path_completeness']:.1%} path"
        ax.text(0.5, -0.05, info_text, transform=ax.transAxes,
               fontsize=8, ha='center', va='top')
        
        # Fixed axes
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 300)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_merges, rows * cols):
        row_idx = idx // cols
        col_idx = idx % cols
        axes[row_idx, col_idx].axis('off')
    
    plt.suptitle(f'Consolidated IDs - Vector Visualization ({n_merges} records)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig

def visualize_merge_chains_grid(merge_results, chains, n_chains=12, cols=3):
    """
    Visualize merge chains (multiple connected IDs) in a grid layout
    Direction-aware: includes box height in arrow endpoints
    """
    if len(chains) == 0:
        print("No chains to visualize")
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(0.5, 0.5, 'No chains found', 
                ha='center', va='center', fontsize=16)
        ax.axis('off')
        return fig
    
    n_chains = min(n_chains, len(chains))
    rows = int(np.ceil(n_chains / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols*4.5, rows*4.5))
    if rows == 1:
        axes = axes.reshape(1, -1)
    elif cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Create lookup for merge data
    merge_lookup = {}
    for _, row in merge_results.iterrows():
        merge_lookup[(row['old_id'], row['new_id'])] = row
    
    colors = plt.cm.tab10(np.linspace(0, 1, 10))
    
    for idx in range(n_chains):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col]
        
        chain = chains[idx]
        
        # Plot each segment in the chain
        for i in range(len(chain) - 1):
            old_id = chain[i]
            new_id = chain[i + 1]
            
            if (old_id, new_id) not in merge_lookup:
                continue
                
            pair = merge_lookup[(old_id, new_id)]
            color = colors[i % len(colors)]
            
            # Determine direction for old_id
            old_moving_down = (pair['old_x_start'] + (pair.get('old_w_mean', 0) if 'old_w_mean' in pair else 0)) <= 140
            old_arrow_start = pair['old_y_start']
            if old_moving_down:
                old_arrow_end = pair['old_y_end'] + pair.get('old_h_end', 0)
            else:
                old_arrow_start = pair['old_y_start'] + pair.get('old_h_start', 0)
                old_arrow_end = pair['old_y_end']
            
            # Determine direction for new_id
            new_moving_down = (pair['new_x_start'] + (pair.get('new_w_mean', 0) if 'new_w_mean' in pair else 0)) <= 140
            new_arrow_start = pair['new_y_start']
            if new_moving_down:
                new_arrow_end = pair['new_y_end'] + pair.get('new_h_end', 0)
            else:
                new_arrow_start = pair['new_y_start'] + pair.get('new_h_start', 0)
                new_arrow_end = pair['new_y_end']
            
            # Plot vector with different color for each segment
            ax.arrow(pair['old_x_start'], old_arrow_start, 
                    0, old_arrow_end - old_arrow_start,
                    head_width=8, head_length=10, fc=color, ec=color, 
                    linewidth=2.5, alpha=0.8, length_includes_head=True)
            
            ax.arrow(pair['new_x_start'], new_arrow_start,
                    0, new_arrow_end - new_arrow_start,
                    head_width=8, head_length=10, fc=color, ec=color,
                    linewidth=2.5, alpha=0.8, length_includes_head=True)
            
            # Gap line - connect arrow endpoints
            ax.plot([pair['old_x_end'], pair['new_x_start']], 
                   [old_arrow_end, new_arrow_start], 
                   '--', color=color, linewidth=1.5, alpha=0.6)
            
            # Add ID labels at key points (at arrow start positions)
            if i == 0:  # First segment - label start
                ax.text(pair['old_x_start'], old_arrow_start - 10, 
                       f"ID {old_id}", fontsize=8, ha='center', 
                       fontweight='bold', color=color)
            
            # Label transition point
            ax.text(pair['new_x_start'], new_arrow_start - 10, 
                   f"ID {new_id}", fontsize=8, ha='center', 
                   fontweight='bold', color=color)
        
        # Title with full chain
        chain_str = ' → '.join(map(str, chain))
        ax.set_title(f"Chain: {chain_str}\n({len(chain)} IDs)", 
                    fontsize=10, fontweight='bold')
        
        # Fixed axes
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 300)
        ax.invert_yaxis()
    
    # Hide empty subplots
    for idx in range(n_chains, rows * cols):
        row = idx // cols
        col = idx % cols
        axes[row, col].axis('off')
    
    plt.suptitle(f'Merge Chains Visualization ({n_chains} chains)', 
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
