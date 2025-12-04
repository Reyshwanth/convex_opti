"""
Interactive 3D Visualization of Minimum Volume Enclosing Ellipsoid
Uses Plotly for rotatable, zoomable 3D plot
"""

import json
import numpy as np
import plotly.graph_objects as go
from scipy.spatial import ConvexHull

def load_data(filename="mvee_3d_data.json"):
    """Load the MVEE data exported from Julia"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    points = np.array(data["points"])
    hull = np.array(data["hull"])
    ellipsoid_surface = data["ellipsoid_surface"]
    center = np.array(data["center"])
    P = np.array(data["P"])
    c = np.array(data["c"])
    semi_axes = np.array(data["semi_axes"])
    
    return points, hull, ellipsoid_surface, center, P, c, semi_axes

def generate_ellipsoid_mesh(P, c, n_theta=50, n_phi=25):
    """
    Generate ellipsoid surface mesh for plotting.
    Ellipsoid: ||Px - c|| <= 1
    Surface: x = P^{-1} * (unit_sphere + c)
    """
    P_inv = np.linalg.inv(P)
    
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    
    # Create meshgrid
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Unit sphere points
    x_sphere = np.sin(phi_grid) * np.cos(theta_grid)
    y_sphere = np.sin(phi_grid) * np.sin(theta_grid)
    z_sphere = np.cos(phi_grid)
    
    # Transform to ellipsoid
    x_ellipsoid = np.zeros_like(x_sphere)
    y_ellipsoid = np.zeros_like(y_sphere)
    z_ellipsoid = np.zeros_like(z_sphere)
    
    for i in range(n_phi):
        for j in range(n_theta):
            unit_vec = np.array([x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]])
            ellipsoid_point = P_inv @ (unit_vec + c)
            x_ellipsoid[i, j] = ellipsoid_point[0]
            y_ellipsoid[i, j] = ellipsoid_point[1]
            z_ellipsoid[i, j] = ellipsoid_point[2]
    
    return x_ellipsoid, y_ellipsoid, z_ellipsoid

def create_interactive_plot(points, hull, center, P, c, semi_axes):
    """Create an interactive 3D plot with Plotly"""
    
    fig = go.Figure()
    
    # 1. Plot all points
    fig.add_trace(go.Scatter3d(
        x=points[:, 0],
        y=points[:, 1],
        z=points[:, 2],
        mode='markers',
        marker=dict(size=4, color='blue', opacity=0.6),
        name=f'Points (n={len(points)})',
        hovertemplate='Point<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # 2. Plot convex hull vertices
    fig.add_trace(go.Scatter3d(
        x=hull[:, 0],
        y=hull[:, 1],
        z=hull[:, 2],
        mode='markers',
        marker=dict(size=6, color='green', symbol='diamond'),
        name=f'Hull vertices (n={len(hull)})',
        hovertemplate='Hull vertex<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # 3. Plot the convex hull surface (triangulated)
    try:
        hull_obj = ConvexHull(hull)
        
        # Create mesh for convex hull
        fig.add_trace(go.Mesh3d(
            x=hull[:, 0],
            y=hull[:, 1],
            z=hull[:, 2],
            i=hull_obj.simplices[:, 0],
            j=hull_obj.simplices[:, 1],
            k=hull_obj.simplices[:, 2],
            color='green',
            opacity=0.3,
            name='Convex Hull',
            showlegend=True,
            hoverinfo='skip'
        ))
        
        # Also add edges of the convex hull for better visibility
        edges_x, edges_y, edges_z = [], [], []
        for simplex in hull_obj.simplices:
            for i in range(3):
                p1 = hull[simplex[i]]
                p2 = hull[simplex[(i + 1) % 3]]
                edges_x.extend([p1[0], p2[0], None])
                edges_y.extend([p1[1], p2[1], None])
                edges_z.extend([p1[2], p2[2], None])
        
        fig.add_trace(go.Scatter3d(
            x=edges_x, y=edges_y, z=edges_z,
            mode='lines',
            line=dict(color='darkgreen', width=2),
            name='Hull edges',
            showlegend=True,
            hoverinfo='skip'
        ))
    except Exception as e:
        print(f"Warning: Could not compute convex hull mesh: {e}")
    
    # 4. Plot the ellipsoid surface
    xe, ye, ze = generate_ellipsoid_mesh(P, c, n_theta=40, n_phi=20)
    
    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        colorscale=[[0, 'rgba(255,0,0,0.4)'], [1, 'rgba(255,100,100,0.4)']],
        showscale=False,
        opacity=0.5,
        name='Min Volume Ellipsoid',
        hovertemplate='Ellipsoid<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # 5. Plot the center
    fig.add_trace(go.Scatter3d(
        x=[center[0]],
        y=[center[1]],
        z=[center[2]],
        mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='Ellipsoid Center',
        hovertemplate='Center<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # 6. Plot principal axes
    # The axes are along the eigenvectors of P'P, with lengths = semi_axes
    U, S, Vt = np.linalg.svd(P)
    P_inv = np.linalg.inv(P)
    
    colors = ['red', 'green', 'blue']
    axis_names = ['Axis 1', 'Axis 2', 'Axis 3']
    
    for i in range(3):
        # Direction of principal axis (column of V from SVD of P_inv)
        direction = P_inv @ U[:, i]
        direction = direction / np.linalg.norm(direction) * semi_axes[i]
        
        fig.add_trace(go.Scatter3d(
            x=[center[0] - direction[0], center[0] + direction[0]],
            y=[center[1] - direction[1], center[1] + direction[1]],
            z=[center[2] - direction[2], center[2] + direction[2]],
            mode='lines',
            line=dict(color=colors[i], width=4),
            name=f'{axis_names[i]} (len={semi_axes[i]:.2f})',
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Minimum Volume Enclosing Ellipsoid (3D)</b><br>' +
                 f'<span style="font-size:12px">Center: ({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) | ' +
                 f'Semi-axes: ({semi_axes[0]:.2f}, {semi_axes[1]:.2f}, {semi_axes[2]:.2f})</span>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            )
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        margin=dict(l=0, r=0, t=80, b=0),
        width=1000,
        height=800
    )
    
    # Add view buttons
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.2}}],
                        label="Default",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5}}],
                        label="Top",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0}}],
                        label="Side X",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 2.5, "z": 0}}],
                        label="Side Y",
                        method="relayout"
                    ),
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.12,
                yanchor="top"
            ),
        ]
    )
    
    return fig

def main():
    print("Loading MVEE 3D data from Julia...")
    
    try:
        points, hull, ellipsoid_surface, center, P, c, semi_axes = load_data()
    except FileNotFoundError:
        print("Error: mvee_3d_data.json not found!")
        print("Please run the Julia script first: julia minimum_volume_ellipsoid.jl")
        return
    
    print(f"Loaded {len(points)} points, {len(hull)} hull vertices")
    print(f"Ellipsoid center: {center}")
    print(f"Semi-axes: {semi_axes}")
    print(f"Shape matrix P:\n{P}")
    
    print("\nCreating interactive 3D plot...")
    fig = create_interactive_plot(points, hull, center, P, c, semi_axes)
    
    # Save as HTML
    html_file = "mvee_3d_interactive.html"
    fig.write_html(html_file, include_plotlyjs=True, full_html=True)
    print(f"\nInteractive plot saved to '{html_file}'")
    print("Open this file in a web browser to interact with the 3D plot.")
    
    # Show the plot
    fig.show()

if __name__ == "__main__":
    main()
