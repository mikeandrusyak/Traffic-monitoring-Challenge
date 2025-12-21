import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

def visualize_vehicle_trajectories(df, session_id=0, max_vehicles=25, min_records=3, 
                                   figsize_per_plot=3, col_wrap=5, title=None):
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