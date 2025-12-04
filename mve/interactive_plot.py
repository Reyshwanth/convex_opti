"""
Interactive 3D Visualization of Maximum Volume Ellipsoid
Uses Plotly for rotatable, zoomable 3D plot
"""

import json
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def load_data(filename="ellipsoid_data.json"):
    """Load the ellipsoid data exported from Julia"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    beacons = np.array(data["beacons"])
    rho = np.array(data["rho"])
    r_sol = np.array(data["r_sol"])
    P_sol = np.array(data["P_sol"])
    L = data["L"]
    
    return beacons, rho, r_sol, P_sol, L

def generate_ellipsoid_surface(center, P, n_points=50):
    """
    Generate surface points for an ellipsoid defined by:
    point = center + P @ unit_sphere_point
    """
    theta = np.linspace(0, 2 * np.pi, n_points)
    phi = np.linspace(0, np.pi, n_points)
    
    # Create meshgrid for spherical coordinates
    theta_grid, phi_grid = np.meshgrid(theta, phi)
    
    # Unit sphere points
    x_sphere = np.sin(phi_grid) * np.cos(theta_grid)
    y_sphere = np.sin(phi_grid) * np.sin(theta_grid)
    z_sphere = np.cos(phi_grid)
    
    # Transform to ellipsoid
    x_ellipsoid = np.zeros_like(x_sphere)
    y_ellipsoid = np.zeros_like(y_sphere)
    z_ellipsoid = np.zeros_like(z_sphere)
    
    for i in range(n_points):
        for j in range(n_points):
            unit_vec = np.array([x_sphere[i, j], y_sphere[i, j], z_sphere[i, j]])
            ellipsoid_point = center + P @ unit_vec
            x_ellipsoid[i, j] = ellipsoid_point[0]
            y_ellipsoid[i, j] = ellipsoid_point[1]
            z_ellipsoid[i, j] = ellipsoid_point[2]
    
    return x_ellipsoid, y_ellipsoid, z_ellipsoid

def generate_sphere_wireframe(center, radius, n_circles=8, n_points=50):
    """Generate wireframe circles for a sphere"""
    lines_x, lines_y, lines_z = [], [], []
    
    # Latitude circles
    for phi in np.linspace(0.1, np.pi - 0.1, n_circles // 2):
        theta = np.linspace(0, 2 * np.pi, n_points)
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi) * np.ones_like(theta)
        lines_x.extend(list(x) + [None])
        lines_y.extend(list(y) + [None])
        lines_z.extend(list(z) + [None])
    
    # Longitude circles
    for theta in np.linspace(0, np.pi, n_circles // 2):
        phi = np.linspace(0, 2 * np.pi, n_points)
        x = center[0] + radius * np.sin(phi) * np.cos(theta)
        y = center[1] + radius * np.sin(phi) * np.sin(theta)
        z = center[2] + radius * np.cos(phi)
        lines_x.extend(list(x) + [None])
        lines_y.extend(list(y) + [None])
        lines_z.extend(list(z) + [None])
    
    return lines_x, lines_y, lines_z

def compute_sphere_intersection_circle(c1, r1, c2, r2, n_points=50):
    """
    Compute the intersection circle of two spheres.
    
    Sphere 1: center c1, radius r1
    Sphere 2: center c2, radius r2
    
    Returns: (center, radius, normal) of the intersection circle, or None if no intersection
    """
    # Vector from c1 to c2
    d_vec = c2 - c1
    d = np.linalg.norm(d_vec)
    
    if d < 1e-10:
        # Spheres are concentric
        return None
    
    # Check if spheres intersect
    # They intersect if |r1 - r2| <= d <= r1 + r2
    if d > r1 + r2 or d < abs(r1 - r2):
        return None
    
    # Distance from c1 to the plane of intersection
    # Using the formula: h = (d^2 + r1^2 - r2^2) / (2*d)
    h = (d**2 + r1**2 - r2**2) / (2 * d)
    
    # Radius of the intersection circle
    # r_circle^2 = r1^2 - h^2
    r_circle_sq = r1**2 - h**2
    if r_circle_sq < 0:
        return None
    r_circle = np.sqrt(r_circle_sq)
    
    # Center of the intersection circle
    # It lies on the line from c1 to c2, at distance h from c1
    normal = d_vec / d  # Unit vector from c1 to c2
    circle_center = c1 + h * normal
    
    return circle_center, r_circle, normal, h, d

def generate_circle_points(center, radius, normal, n_points=50):
    """
    Generate points on a circle in 3D space.
    
    center: center of the circle
    radius: radius of the circle
    normal: normal vector to the plane containing the circle
    """
    # Create two orthonormal vectors in the plane of the circle
    normal = normal / np.linalg.norm(normal)
    
    # Find a vector not parallel to normal
    if abs(normal[0]) < 0.9:
        v = np.array([1, 0, 0])
    else:
        v = np.array([0, 1, 0])
    
    # Create orthonormal basis in the plane
    u1 = np.cross(normal, v)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(normal, u1)
    u2 = u2 / np.linalg.norm(u2)
    
    # Generate circle points
    theta = np.linspace(0, 2 * np.pi, n_points)
    x = center[0] + radius * (np.cos(theta) * u1[0] + np.sin(theta) * u2[0])
    y = center[1] + radius * (np.cos(theta) * u1[1] + np.sin(theta) * u2[1])
    z = center[2] + radius * (np.cos(theta) * u1[2] + np.sin(theta) * u2[2])
    
    return x, y, z

def generate_lens_surface(c1, r1, c2, r2, n_theta=20, n_phi=10):
    """
    Generate the surface of the lens-shaped intersection volume of two spheres.
    
    The lens is composed of two spherical caps:
    - Cap from sphere 1 (the part inside sphere 2)
    - Cap from sphere 2 (the part inside sphere 1)
    
    Returns arrays of x, y, z coordinates for the lens surface, or None if no intersection.
    """
    result = compute_sphere_intersection_circle(c1, r1, c2, r2)
    if result is None:
        return None
    
    circle_center, circle_radius, normal, h1, d = result
    
    # h1 is distance from c1 to intersection plane
    # h2 is distance from c2 to intersection plane
    h2 = d - h1
    
    # Create orthonormal basis
    if abs(normal[0]) < 0.9:
        v = np.array([1, 0, 0])
    else:
        v = np.array([0, 1, 0])
    
    u1 = np.cross(normal, v)
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(normal, u1)
    u2 = u2 / np.linalg.norm(u2)
    
    theta = np.linspace(0, 2 * np.pi, n_theta)
    
    all_x, all_y, all_z = [], [], []
    
    # Generate spherical cap from sphere 1 (part facing sphere 2)
    # This cap goes from the intersection circle towards c2
    # The cap height on sphere 1 is r1 - h1
    cap_height_1 = r1 - h1
    if cap_height_1 > 0:
        # Parametrize the spherical cap
        # phi goes from 0 (at intersection circle) to angle where cap ends
        phi_max_1 = np.arccos(h1 / r1) if abs(h1 / r1) <= 1 else 0
        phi_1 = np.linspace(0, phi_max_1, n_phi)
        
        for p in phi_1:
            r_ring = r1 * np.sin(np.arccos(h1/r1) - p) if abs(h1/r1) <= 1 else 0
            z_offset = h1 + r1 * (1 - np.cos(np.arccos(h1/r1) - p)) if abs(h1/r1) <= 1 else h1
            
            # Actually, let's use a cleaner parametrization
            # Point on sphere 1 at angle from the axis
            pass
        
    # Simpler approach: generate both caps using spherical coordinates
    # Cap 1: part of sphere 1 that's beyond the intersection plane (towards c2)
    # Cap 2: part of sphere 2 that's beyond the intersection plane (towards c1)
    
    all_x, all_y, all_z = [], [], []
    
    # Cap from sphere 1
    # Angle at intersection circle (from center c1)
    if abs(h1/r1) <= 1:
        theta_cap1 = np.arccos(h1 / r1)
        # Generate cap from theta_cap1 to 0 (the pole facing c2)
        phi_vals = np.linspace(0, theta_cap1, n_phi)
        
        for phi in phi_vals:
            r_ring = r1 * np.sin(phi)
            # Position along axis from c1
            z_along_axis = r1 * np.cos(phi)
            
            for t in theta:
                # Point in local coordinates (axis from c1 to c2)
                local_point = z_along_axis * normal + r_ring * (np.cos(t) * u1 + np.sin(t) * u2)
                point = c1 + local_point
                all_x.append(point[0])
                all_y.append(point[1])
                all_z.append(point[2])
    
    # Cap from sphere 2
    # Angle at intersection circle (from center c2)
    if abs(h2/r2) <= 1:
        theta_cap2 = np.arccos(h2 / r2)
        phi_vals = np.linspace(0, theta_cap2, n_phi)
        
        for phi in phi_vals:
            r_ring = r2 * np.sin(phi)
            z_along_axis = r2 * np.cos(phi)
            
            for t in theta:
                # Point in local coordinates (axis from c2 to c1, so negative normal)
                local_point = z_along_axis * (-normal) + r_ring * (np.cos(t) * u1 + np.sin(t) * u2)
                point = c2 + local_point
                all_x.append(point[0])
                all_y.append(point[1])
                all_z.append(point[2])
    
    if len(all_x) == 0:
        return None
    
    return np.array(all_x), np.array(all_y), np.array(all_z)

def create_interactive_plot(beacons, rho, r_sol, P_sol, L, show_spheres=True):
    """Create an interactive 3D plot with Plotly"""
    
    fig = go.Figure()
    
    # 1. Plot beacons as black spheres
    fig.add_trace(go.Scatter3d(
        x=beacons[:, 0],
        y=beacons[:, 1],
        z=beacons[:, 2],
        mode='markers',
        marker=dict(size=6, color='black'),
        name=f'Beacons (n={L})',
        hovertemplate='Beacon<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # 2. Plot 3D lens-shaped intersection volumes for every pair of range spheres
    if show_spheres:
        intersection_count = 0
        all_lens_x, all_lens_y, all_lens_z = [], [], []
        
        for i in range(L):
            for j in range(i + 1, L):
                result = generate_lens_surface(
                    beacons[i], rho[i], beacons[j], rho[j], n_theta=15, n_phi=8
                )
                if result is not None:
                    lx, ly, lz = result
                    all_lens_x.extend(list(lx))
                    all_lens_y.extend(list(ly))
                    all_lens_z.extend(list(lz))
                    intersection_count += 1
        
        # Add all intersection lens volumes as scatter points with low opacity
        if intersection_count > 0:
            fig.add_trace(go.Scatter3d(
                x=all_lens_x, y=all_lens_y, z=all_lens_z,
                mode='markers',
                marker=dict(
                    size=2,
                    color='purple',
                    opacity=0.08
                ),
                name=f'Intersection volumes ({intersection_count})',
                hoverinfo='skip'
            ))
        
        print(f"Found {intersection_count} intersecting sphere pairs out of {L*(L-1)//2} total pairs")
    
    # 3. Plot the ellipsoid surface
    xe, ye, ze = generate_ellipsoid_surface(r_sol, P_sol, n_points=40)
    
    fig.add_trace(go.Surface(
        x=xe, y=ye, z=ze,
        colorscale=[[0, 'rgba(255,0,0,0.6)'], [1, 'rgba(255,100,100,0.6)']],
        showscale=False,
        name='Maximum Volume Ellipsoid',
        hovertemplate='Ellipsoid<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # 4. Plot the estimated center
    fig.add_trace(go.Scatter3d(
        x=[r_sol[0]],
        y=[r_sol[1]],
        z=[r_sol[2]],
        mode='markers',
        marker=dict(size=10, color='green', symbol='diamond'),
        name='Estimated Position',
        hovertemplate='Estimate<br>x: %{x:.3f}<br>y: %{y:.3f}<br>z: %{z:.3f}<extra></extra>'
    ))
    
    # 5. Draw lines from center to beacons (optional visualization)
    # Uncomment if you want to see the connections
    # for i in range(L):
    #     fig.add_trace(go.Scatter3d(
    #         x=[r_sol[0], beacons[i, 0]],
    #         y=[r_sol[1], beacons[i, 1]],
    #         z=[r_sol[2], beacons[i, 2]],
    #         mode='lines',
    #         line=dict(color='gray', width=1),
    #         showlegend=False,
    #         hoverinfo='skip'
    #     ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='<b>Maximum Volume Ellipsoid - Interactive 3D View</b><br>' +
                 f'<span style="font-size:12px">Estimated position: ({r_sol[0]:.4f}, {r_sol[1]:.4f}, {r_sol[2]:.4f})</span>',
            x=0.5,
            font=dict(size=16)
        ),
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data',  # Equal aspect ratio
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
        width=900,
        height=700
    )
    
    # Add buttons for different views
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="left",
                buttons=list([
                    dict(
                        args=[{"scene.camera.eye": {"x": 1.5, "y": 1.5, "z": 1.2}}],
                        label="Default View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 0, "z": 2.5}}],
                        label="Top View",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 2.5, "y": 0, "z": 0}}],
                        label="Side View (X)",
                        method="relayout"
                    ),
                    dict(
                        args=[{"scene.camera.eye": {"x": 0, "y": 2.5, "z": 0}}],
                        label="Side View (Y)",
                        method="relayout"
                    ),
                ]),
                pad={"r": 10, "t": 10},
                showactive=True,
                x=0.0,
                xanchor="left",
                y=1.15,
                yanchor="top"
            ),
        ]
    )
    
    return fig

def main():
    print("Loading ellipsoid data from Julia...")
    
    try:
        beacons, rho, r_sol, P_sol, L = load_data()
    except FileNotFoundError:
        print("Error: ellipsoid_data.json not found!")
        print("Please run the Julia script first to generate the data.")
        return
    
    print(f"Loaded data for {L} beacons")
    print(f"Estimated position: {r_sol}")
    print(f"Ellipsoid shape matrix P:\n{P_sol}")
    
    # Compute ellipsoid properties
    eigenvalues, eigenvectors = np.linalg.eig(P_sol)
    print(f"\nEllipsoid semi-axes lengths: {np.abs(eigenvalues)}")
    
    print("\nCreating interactive 3D plot...")
    fig = create_interactive_plot(beacons, rho, r_sol, P_sol, L, show_spheres=True)
    
    # Save as HTML for browser viewing
    html_file = "ellipsoid_interactive.html"
    fig.write_html(html_file, include_plotlyjs=True, full_html=True)
    print(f"\nInteractive plot saved to '{html_file}'")
    print("Open this file in a web browser to interact with the 3D plot.")
    
    # Also show the plot directly (opens in browser)
    fig.show()

if __name__ == "__main__":
    main()
