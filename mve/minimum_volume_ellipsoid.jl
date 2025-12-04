# Minimum Volume Ellipsoid enclosing Convex Hull - 3D Version
# This program:
# 1. Generates random points in 3D Euclidean space
# 2. Computes the convex hull of those points
# 3. Fits a minimum volume enclosing ellipsoid (MVEE) using SDP formulation
# 4. Exports data for interactive 3D visualization in Python
#
# Formulation (from MOSEK Modeling Cookbook):
#   Ellipsoid: E = {x | ||Px - c||_2 <= 1}
#   
#   max  log(det(P))
#   s.t. ||P*x_i - c||_2 <= 1,  for all data points x_i
#        P ≽ 0  (positive semidefinite)

using LinearAlgebra
using Random
using Convex
using SCS
using JSON

# ============================================================================
# 3D Convex Hull - Quickhull-inspired approach
# ============================================================================

function convex_hull_3d(points)
    """
    Compute 3D convex hull vertices using extreme point detection.
    Returns the points that lie on the convex hull.
    """
    n = length(points)
    if n <= 4
        return points
    end
    
    # Convert to matrix
    P = hcat([collect(Float64, p) for p in points]...)  # 3 x n
    
    hull_indices = Set{Int}()
    
    # Find extreme points along many directions
    # Principal axes
    for dim in 1:3
        _, min_idx = findmin(P[dim, :])
        _, max_idx = findmax(P[dim, :])
        push!(hull_indices, min_idx)
        push!(hull_indices, max_idx)
    end
    
    # Diagonal and off-axis directions
    directions = [
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1],
        [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1],
        [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
        [0, 1, 1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1],
        [0, -1, 1], [-1, -1, 0], [-1, 0, -1], [0, -1, -1]
    ]
    
    for dir in directions
        d = normalize(Float64.(dir))
        projections = [dot(d, P[:, i]) for i in 1:n]
        _, max_idx = findmax(projections)
        _, min_idx = findmin(projections)
        push!(hull_indices, max_idx)
        push!(hull_indices, min_idx)
    end
    
    # Also sample random directions for better coverage
    Random.seed!(123)
    for _ in 1:50
        d = normalize(randn(3))
        projections = [dot(d, P[:, i]) for i in 1:n]
        _, max_idx = findmax(projections)
        push!(hull_indices, max_idx)
    end
    
    return [points[i] for i in hull_indices]
end

# ============================================================================
# Minimum Volume Enclosing Ellipsoid (MVEE) - SDP Formulation
# ============================================================================

function mvee_sdp(points)
    """
    Compute the Minimum Volume Enclosing Ellipsoid using SDP.
    
    The ellipsoid is represented as: E = {x | ||Px - c||_2 <= 1}
    
    Formulation:
        max  log(det(P))
        s.t. ||P*x_i - c||_2 <= 1,  for all points x_i
             P ≽ 0
    
    Returns:
        P: The ellipsoid shape matrix (positive semidefinite)
        c: The center parameter (note: actual center = P^{-1} * c)
    """
    # Get dimension and number of points
    d = length(points[1])
    m = length(points)
    
    # Convert points to matrix form
    X = hcat([collect(Float64, p) for p in points]...)  # d x m matrix
    
    # Decision variables
    P = Semidefinite(d)  # d x d positive semidefinite matrix
    c = Variable(d)       # d-dimensional vector
    
    # Constraints: ||P*x_i - c||_2 <= 1 for all points
    constraints = Constraint[]
    for i in 1:m
        xi = X[:, i]
        push!(constraints, norm(P * xi - c) <= 1)
    end
    
    # Objective: maximize log(det(P))
    problem = maximize(logdet(P), constraints)
    
    # Solve using SCS
    solve!(problem, SCS.Optimizer; silent=true)
    
    return evaluate(P), evaluate(c)
end

# ============================================================================
# Generate Ellipsoid Surface Points for 3D
# ============================================================================

function ellipsoid_surface_points(P, c; n_theta=50, n_phi=25)
    """
    Generate points on the surface of an ellipsoid defined by:
    ||Px - c||_2 = 1
    
    The surface points satisfy: x = P^{-1} * (unit_sphere + c)
    """
    P_inv = inv(P)
    center = P_inv * c
    
    theta = range(0, 2π, length=n_theta)
    phi = range(0, π, length=n_phi)
    
    xs = Float64[]
    ys = Float64[]
    zs = Float64[]
    
    for t in theta, p in phi
        # Point on unit sphere
        u = [sin(p) * cos(t), sin(p) * sin(t), cos(p)]
        # Transform to ellipsoid surface
        pt = P_inv * (u .+ c)
        push!(xs, pt[1])
        push!(ys, pt[2])
        push!(zs, pt[3])
    end
    
    return xs, ys, zs, center
end

# ============================================================================
# Main Program
# ============================================================================

function main()
    Random.seed!(42)
    
    # Generate random points in 3D
    n_points = 100
    points = [(randn() * 3, randn() * 2 + 1, randn() * 2.5) for _ in 1:n_points]
    
    # Add some outliers to make it more interesting
    push!(points, (8.0, 3.0, 4.0))
    push!(points, (-6.0, -2.0, 3.0))
    push!(points, (2.0, 7.0, -5.0))
    push!(points, (0.0, -4.0, 6.0))
    
    println("Generated $(length(points)) random 3D points")
    
    # Compute convex hull
    hull = convex_hull_3d(points)
    println("Convex hull has $(length(hull)) vertices")
    
    # Compute minimum volume enclosing ellipsoid using SDP
    println("\nSolving SDP for Minimum Volume Enclosing Ellipsoid...")
    P, c = mvee_sdp(hull)
    
    println("\nMinimum Volume Enclosing Ellipsoid:")
    println("  Shape matrix P:")
    display(round.(P, digits=6))
    println("\n  Parameter c: ", round.(c, digits=4))
    
    # Compute the actual center of the ellipsoid
    center = P \ c
    println("  Ellipsoid center: ", round.(center, digits=4))
    
    # Compute ellipsoid parameters (semi-axes)
    svd_P = svd(P)
    semi_axes = 1.0 ./ svd_P.S
    println("  Semi-axes lengths: ", round.(semi_axes, digits=4))
    println("  Determinant of P: ", round(det(P), digits=6))
    
    # Get ellipsoid surface points
    ellipsoid_x, ellipsoid_y, ellipsoid_z, center = ellipsoid_surface_points(P, c)
    
    # ========================================================================
    # Export data for Python interactive visualization
    # ========================================================================
    
    data_export = Dict(
        "points" => [collect(Float64, p) for p in points],
        "hull" => [collect(Float64, p) for p in hull],
        "ellipsoid_surface" => Dict(
            "x" => ellipsoid_x,
            "y" => ellipsoid_y,
            "z" => ellipsoid_z
        ),
        "center" => collect(center),
        "P" => [collect(P[i, :]) for i in 1:3],
        "c" => collect(c),
        "semi_axes" => collect(semi_axes),
        "n_points" => length(points),
        "n_hull" => length(hull)
    )
    
    open("mvee_3d_data.json", "w") do f
        JSON.print(f, data_export)
    end
    println("\nData exported to 'mvee_3d_data.json'")
    println("Run: python3 plot_mvee_3d.py")
    
    return points, hull, P, c
end

# Run the main function
points, hull, P, c = main()
