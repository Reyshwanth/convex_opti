"""
Interactive 3D Visualization of the Rey Ellipsoid Method
"""

import json
import numpy as np
import plotly.graph_objects as go

def load_data(filename="rey_ellipsoid_data.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def generate_ellipsoid_mesh(P, c, n_theta=40, n_phi=20):
    P = np.array(P)
    c = np.array(c)
    P_inv = np.linalg.inv(P)
    
    theta = np.linspace(0, 2 * np.pi, n_theta)
    phi = np.linspace(0, np.pi, n_phi)
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    x_sphere = np.sin(phi_grid) * np.cos(theta_grid)
    y_sphere = np.sin(phi_grid) * np.sin(theta_grid)
    z_sphere = np.cos(phi_grid)
    
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

def generate_circle_3d(center, radius, normal, n_points=50):
    """Generate points for a circle in 3D given center, radius, and normal vector"""
    center = np.array(center)
    normal = np.array(normal)
    normal = normal / np.linalg.norm(normal)
    
    # Find two orthonormal vectors in the plane
    if abs(normal[0]) < 0.9:
        u1 = np.cross([1.0, 0.0, 0.0], normal)
    else:
        u1 = np.cross([0.0, 1.0, 0.0], normal)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(normal, u1)
    u2 = u2 / np.linalg.norm(u2)
    
    # Generate circle points
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = center[0] + radius * (np.cos(theta) * u1[0] + np.sin(theta) * u2[0])
    y = center[1] + radius * (np.cos(theta) * u1[1] + np.sin(theta) * u2[1])
    z = center[2] + radius * (np.cos(theta) * u1[2] + np.sin(theta) * u2[2])
    
    return x, y, z

def create_rey_ellipsoid_plot(data):
    """Create an interactive 3D visualization of the Rey Ellipsoid method"""
    
    fig = go.Figure()
    
    # Extract data
    true_position = np.array(data["true_position"])
    beacons = data["beacons"]
    sampled_points = np.array(data["sampled_points"])
    weights = np.array(data["normalized_weights"])
    rey_ellipsoid = data["rey_ellipsoid"]
    
    # Normalize weights to opacity range [0.3, 1.0]
    opacities = 0.3 + 0.7 * weights
    
    # ========================================================================
    # Plot sampled points with weight-based opacity
    # ========================================================================
    
    point_colors = [f'rgba(0, 150, 255, {op:.2f})' for op in opacities]
    
    fig.add_trace(go.Scatter3d(
        x=sampled_points[:, 0],
        y=sampled_points[:, 1],
        z=sampled_points[:, 2],
        mode='markers',
        marker=dict(
            size=4,
            color=point_colors,
            line=dict(width=0.5, color='darkblue')
        ),
        name=f'Sampled Points (n={len(sampled_points)})',
        text=[f'Point {i+1}<br>Weight: {weights[i]:.3f}' for i in range(len(sampled_points))],
        hoverinfo='text'
    ))
    
    # ========================================================================
    # Plot beacons
    # ========================================================================
    
    beacon_positions = np.array([b["position"] for b in beacons])
    beacon_sigmas = np.array([b["sigma"] for b in beacons])
    
    fig.add_trace(go.Scatter3d(
        x=beacon_positions[:, 0],
        y=beacon_positions[:, 1],
        z=beacon_positions[:, 2],
        mode='markers',
        marker=dict(
            size=8,
            color='orange',
            symbol='diamond'
        ),
        name=f'Beacons (n={len(beacons)})',
        text=[f'Beacon {i+1}<br>Range: {b["measured_range"]:.2f}<br>σ: {b["sigma"]:.2f}' 
              for i, b in enumerate(beacons)],
        hoverinfo='text'
    ))
    
    # ========================================================================
    # Plot intersection circles
    # ========================================================================
    
    intersections = data.get("intersections", [])
    for i, inter in enumerate(intersections):
        x, y, z = generate_circle_3d(
            inter["center"], 
            inter["radius"],
            inter["normal"],
            n_points=40
        )
        # Color based on weight (higher weight = more opaque)
        weight = inter["weight"]
        max_weight = max(intr["weight"] for intr in intersections)
        opacity = 0.2 + 0.6 * (weight / max_weight)
        
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z,
            mode='lines',
            line=dict(color=f'rgba(100, 100, 255, {opacity:.2f})', width=2),
            name=f'Intersection circle' if i == 0 else None,
            showlegend=(i == 0),
            hoverinfo='text',
            text=f'Beacons {inter["beacon1"]}-{inter["beacon2"]}<br>Weight: {weight:.2f}'
        ))
    
    # ========================================================================
    # Plot Rey Ellipsoid
    # ========================================================================
    
    P = np.array(rey_ellipsoid["P"])
    c = np.array(rey_ellipsoid["c"])
    center = np.array(rey_ellipsoid["center"])
    
    xe, ye, ze = generate_ellipsoid_mesh(P, c)
    
    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        colorscale=[[0, 'rgba(128, 0, 128, 0.4)'], [1, 'rgba(180, 0, 180, 0.5)']],
        showscale=False,
        opacity=0.5,
        name='Rey Ellipsoid'
    ))
    
    # Plot ellipsoid center (estimated position)
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='markers',
        marker=dict(size=12, color='purple', symbol='diamond'),
        name=f'Estimated Position'
    ))
    
    # ========================================================================
    # Plot true position
    # ========================================================================
    
    fig.add_trace(go.Scatter3d(
        x=[true_position[0]], y=[true_position[1]], z=[true_position[2]],
        mode='markers',
        marker=dict(size=12, color='green', symbol='cross'),
        name='True Position'
    ))
    
    # ========================================================================
    # Add line connecting true and estimated positions
    # ========================================================================
    
    fig.add_trace(go.Scatter3d(
        x=[true_position[0], center[0]],
        y=[true_position[1], center[1]],
        z=[true_position[2], center[2]],
        mode='lines',
        line=dict(color='red', width=3, dash='dash'),
        name=f'Error: {data["rey_ellipsoid"]["position_error"]:.2f}'
    ))
    
    # ========================================================================
    # Layout
    # ========================================================================
    
    fig.update_layout(
        title=dict(
            text=f'<b>Rey Ellipsoid Method</b><br>' +
                 f'<span style="font-size:12px">{data["n_beacons"]} beacons, ' +
                 f'{data["n_intersections"]} intersections, ' +
                 f'{len(sampled_points)} sampled points | ' +
                 f'Position error: {data["rey_ellipsoid"]["position_error"]:.3f}</span>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.0)),
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        height=800,
        width=1100,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def main():
    print("Loading Rey Ellipsoid data...")
    
    try:
        data = load_data()
    except FileNotFoundError:
        print("Error: rey_ellipsoid_data.json not found!")
        print("Please run: julia rey_ellipsoid.jl")
        return
    
    print(f"\nRey Ellipsoid Method Results:")
    print(f"  Beacons: {data['n_beacons']}")
    print(f"  Intersecting pairs: {data['n_intersections']}")
    print(f"  Sampled points: {len(data['sampled_points'])}")
    print(f"  True position: {data['true_position']}")
    print(f"  Estimated position: {np.array(data['rey_ellipsoid']['center']).round(3)}")
    print(f"  Position error: {data['rey_ellipsoid']['position_error']:.4f}")
    
    print("\nCreating visualization...")
    fig = create_rey_ellipsoid_plot(data)
    fig.write_html("rey_ellipsoid.html", include_plotlyjs=True)
    print("Saved: rey_ellipsoid.html")
    
    fig.show()

if __name__ == "__main__":
    main()
