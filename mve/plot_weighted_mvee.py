"""
Interactive 3D Visualization of Robust MVEE
"""

import json
import numpy as np
import plotly.graph_objects as go

def load_data(filename="weighted_mvee_data.json"):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

def generate_ellipsoid_mesh(P, c, n_theta=30, n_phi=15):
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

def create_robust_plot(data):
    """Create a plot showing the robust MVEE result"""
    
    points = np.array(data["points"])
    n_inliers = data["n_inliers"]
    robust = data["robust"]
    
    # Get weights from robust method
    weights = np.array(robust["weights"])
    
    # Normalize weights to opacity range [0.2, 1.0]
    w_min, w_max = weights.min(), weights.max()
    if w_max > w_min:
        opacities = 0.2 + 0.8 * (weights - w_min) / (w_max - w_min)
    else:
        opacities = np.ones_like(weights)
    
    fig = go.Figure()
    
    # Plot inliers with opacity based on weight
    inlier_weights = weights[:n_inliers]
    inlier_opacities = opacities[:n_inliers]
    
    # Create color array with varying opacity for inliers
    inlier_colors = [f'rgba(0, 0, 255, {op:.2f})' for op in inlier_opacities]
    
    fig.add_trace(go.Scatter3d(
        x=points[:n_inliers, 0],
        y=points[:n_inliers, 1],
        z=points[:n_inliers, 2],
        mode='markers',
        marker=dict(
            size=6,
            color=inlier_colors,
            line=dict(width=1, color='darkblue')
        ),
        name=f'Inliers (n={n_inliers})',
        text=[f'Point {i+1}<br>Weight: {inlier_weights[i]:.3f}<br>Opacity: {inlier_opacities[i]:.2f}' 
              for i in range(n_inliers)],
        hoverinfo='text'
    ))
    
    # Plot outliers with opacity based on weight
    outlier_weights = weights[n_inliers:]
    outlier_opacities = opacities[n_inliers:]
    
    outlier_colors = [f'rgba(255, 165, 0, {op:.2f})' for op in outlier_opacities]
    
    fig.add_trace(go.Scatter3d(
        x=points[n_inliers:, 0],
        y=points[n_inliers:, 1],
        z=points[n_inliers:, 2],
        mode='markers',
        marker=dict(
            size=10,
            color=outlier_colors,
            symbol='x',
            line=dict(width=2, color='darkorange')
        ),
        name=f'Outliers (n={len(points)-n_inliers})',
        text=[f'Outlier {i+1}<br>Weight: {outlier_weights[i]:.3f}<br>Opacity: {outlier_opacities[i]:.2f}' 
              for i in range(len(outlier_weights))],
        hoverinfo='text'
    ))
    
    # Plot ellipsoid
    P = np.array(robust["P"])
    c = np.array(robust["c"])
    center = np.array(robust["center"])
    
    xe, ye, ze = generate_ellipsoid_mesh(P, c, n_theta=40, n_phi=20)
    
    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        colorscale=[[0, 'rgba(128, 0, 128, 0.3)'], [1, 'rgba(128, 0, 128, 0.4)']],
        showscale=False,
        opacity=0.4,
        name='Robust Ellipsoid'
    ))
    
    # Plot center
    fig.add_trace(go.Scatter3d(
        x=[center[0]], y=[center[1]], z=[center[2]],
        mode='markers',
        marker=dict(size=10, color='purple', symbol='diamond'),
        name='Ellipsoid Center'
    ))
    
    fig.update_layout(
        title=dict(
            text='<b>Robust MVEE with Iterative Reweighting</b><br>' +
                 '<span style="font-size:12px">Point opacity reflects weight (higher weight = more opaque)</span>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            aspectmode='data',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
        ),
        height=800,
        width=1000,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    
    return fig

def main():
    print("Loading Robust MVEE data from Julia...")
    
    try:
        data = load_data()
    except FileNotFoundError:
        print("Error: weighted_mvee_data.json not found!")
        print("Please run: julia weighted_mvee.jl")
        return
    
    n_points = len(data["points"])
    n_inliers = data["n_inliers"]
    n_outliers = data["n_outliers"]
    
    print(f"Loaded {n_points} points ({n_inliers} inliers, {n_outliers} outliers)")
    
    # Print robust method info
    robust = data["robust"]
    P = np.array(robust["P"])
    det_P = np.linalg.det(P)
    center = np.array(robust["center"])
    weights = np.array(robust["weights"])
    
    print("\nRobust MVEE Results:")
    print(f"  det(P) = {det_P:.6f}")
    print(f"  Volume ∝ 1/det(P) = {1/det_P:.1f}")
    print(f"  Center = {center.round(3)}")
    print(f"  Weight range: [{weights.min():.3f}, {weights.max():.3f}]")
    
    print("\nCreating robust MVEE plot...")
    fig = create_robust_plot(data)
    fig.write_html("robust_mvee.html", include_plotlyjs=True)
    print("Saved: robust_mvee.html")
    
    # Show plot
    fig.show()

if __name__ == "__main__":
    main()
