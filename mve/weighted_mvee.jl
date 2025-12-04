# Weighted Minimum Volume Enclosing Ellipsoid (W-MVEE)
# 
# This program solves the weighted minimum volume enclosing ellipsoid problem
# where each point has an associated weight that controls how much the ellipsoid
# must "cover" that point.
#
# ============================================================================
# FORMULATION
# ============================================================================
#
# Standard MVEE: Find smallest ellipsoid E = {x | ||Px - c||_2 <= 1} containing all points
#
# Weighted MVEE: Each point x_i has weight w_i >= 0
#   - w_i = 1: Point must be inside or on the ellipsoid boundary (standard constraint)
#   - w_i > 1: Point must be strictly inside (shrunk constraint)
#   - w_i < 1: Point can be outside (relaxed constraint, soft margin)
#   - w_i = 0: Point is ignored
#
# Formulation:
#   max  log(det(P))
#   s.t. ||P*x_i - c||_2 <= 1/w_i,  for all points x_i with w_i > 0
#        P ≽ 0  (positive semidefinite)
#
# Equivalently (scaling the constraint):
#   ||P*x_i - c||_2 <= 1/w_i
#   means: w_i * ||P*x_i - c||_2 <= 1
#
# Interpretation:
#   - Higher weight → tighter constraint → point must be well inside
#   - Lower weight → looser constraint → point can be outside
#   - This is useful for:
#     * Outlier handling (low weight for suspected outliers)
#     * Importance weighting (high weight for critical points)
#     * Robust estimation (iteratively reweighted schemes)
#
# ============================================================================

using LinearAlgebra
using Random
using Statistics
using Convex
using SCS
using JSON

# ============================================================================
# Weighted MVEE - SDP Formulation
# ============================================================================

function weighted_mvee_sdp(points, weights)
    """
    Compute the Weighted Minimum Volume Enclosing Ellipsoid using SDP.
    
    The ellipsoid is represented as: E = {x | ||Px - c||_2 <= 1}
    
    Formulation:
        max  log(det(P))
        s.t. w_i * ||P*x_i - c||_2 <= 1,  for all points x_i with w_i > 0
             P ≽ 0
    
    Args:
        points: Vector of points (tuples or vectors)
        weights: Vector of weights (same length as points)
                 w > 1: point must be strictly inside
                 w = 1: point on or inside boundary (standard)
                 0 < w < 1: point can be outside (soft margin)
                 w = 0: point ignored
    
    Returns:
        P: The ellipsoid shape matrix (positive semidefinite)
        c: The center parameter (actual center = P^{-1} * c)
        status: Solver status
    """
    # Get dimension and number of points
    d = length(points[1])
    m = length(points)
    
    @assert length(weights) == m "Number of weights must match number of points"
    @assert all(w -> w >= 0, weights) "All weights must be non-negative"
    
    # Convert points to matrix form
    X = hcat([collect(Float64, p) for p in points]...)  # d x m matrix
    
    # Decision variables
    P = Semidefinite(d)  # d x d positive semidefinite matrix
    c = Variable(d)       # d-dimensional vector
    
    # Constraints: w_i * ||P*x_i - c||_2 <= 1 for all points with w_i > 0
    constraints = Constraint[]
    
    active_count = 0
    for i in 1:m
        if weights[i] > 0
            xi = X[:, i]
            wi = weights[i]
            # Constraint: w_i * ||P*x_i - c||_2 <= 1
            # Equivalent to: ||P*x_i - c||_2 <= 1/w_i
            push!(constraints, wi * norm(P * xi - c) <= 1)
            active_count += 1
        end
    end
    
    println("  Active constraints: $active_count / $m points")
    
    # Objective: maximize log(det(P))
    problem = maximize(logdet(P), constraints)
    
    # Solve using SCS
    solve!(problem, SCS.Optimizer; silent=true)
    
    return evaluate(P), evaluate(c), problem.status
end

# ============================================================================
# Standard (unweighted) MVEE for comparison
# ============================================================================

function standard_mvee_sdp(points)
    """Standard MVEE with all weights = 1"""
    weights = ones(length(points))
    return weighted_mvee_sdp(points, weights)
end

# ============================================================================
# Robust MVEE using iteratively reweighted scheme
# ============================================================================

function robust_mvee(points; max_iter=10, threshold=0.1, verbose=true)
    """
    Robust MVEE that down-weights outliers iteratively.
    
    Algorithm:
    1. Start with uniform weights
    2. Solve weighted MVEE
    3. Compute residuals (distance from ellipsoid boundary)
    4. Down-weight points that are far outside
    5. Repeat until convergence
    
    Args:
        points: Vector of points
        max_iter: Maximum iterations
        threshold: Convergence threshold for weight changes
        verbose: Print iteration info
    
    Returns:
        P, c: Final ellipsoid parameters
        weights: Final weights
        history: Iteration history
    """
    m = length(points)
    d = length(points[1])
    X = hcat([collect(Float64, p) for p in points]...)
    
    # Initialize uniform weights
    weights = ones(m)
    history = []
    
    if verbose
        println("\n=== Robust MVEE (Iteratively Reweighted) ===")
    end
    
    P, c = nothing, nothing
    
    for iter in 1:max_iter
        if verbose
            println("\nIteration $iter:")
        end
        
        # Solve weighted MVEE
        P, c, status = weighted_mvee_sdp(points, weights)
        
        if verbose
            println("  Status: $status")
            println("  det(P) = $(round(det(P), digits=6))")
        end
        
        # Compute residuals: ||P*x_i - c|| for each point
        # Points with residual > 1 are outside the ellipsoid
        residuals = [norm(P * X[:, i] - c) for i in 1:m]
        
        # Store history
        push!(history, (weights=copy(weights), residuals=copy(residuals), detP=det(P)))
        
        if verbose
            n_outside = sum(residuals .> 1.0)
            max_residual = maximum(residuals)
            println("  Points outside ellipsoid: $n_outside / $m")
            println("  Max residual: $(round(max_residual, digits=4))")
        end
        
        # Update weights: down-weight points far outside
        # Using Huber-like weighting: w = min(1, 1/residual) for residual > 1
        new_weights = similar(weights)
        for i in 1:m
            if residuals[i] <= 1.0
                new_weights[i] = 1.0  # Inside: full weight
            else
                # Outside: reduce weight inversely proportional to distance
                new_weights[i] = 1.0 / residuals[i]
            end
        end
        
        # Check convergence
        weight_change = norm(new_weights - weights) / norm(weights)
        if verbose
            println("  Weight change: $(round(weight_change, digits=6))")
        end
        
        if weight_change < threshold
            if verbose
                println("\nConverged after $iter iterations!")
            end
            break
        end
        
        weights = new_weights
    end
    
    return P, c, weights, history
end

# ============================================================================
# Generate Ellipsoid Surface Points for 3D
# ============================================================================

function ellipsoid_surface_points(P, c; n_theta=50, n_phi=25)
    P_inv = inv(P)
    center = P_inv * c
    
    theta = range(0, 2π, length=n_theta)
    phi = range(0, π, length=n_phi)
    
    xs, ys, zs = Float64[], Float64[], Float64[]
    
    for t in theta, p in phi
        u = [sin(p) * cos(t), sin(p) * sin(t), cos(p)]
        pt = P_inv * (u .+ c)
        push!(xs, pt[1])
        push!(ys, pt[2])
        push!(zs, pt[3])
    end
    
    return xs, ys, zs, center
end

# ============================================================================
# Main Program - Robust MVEE Only
# ============================================================================

function demo_robust_mvee()
    Random.seed!(42)
    
    println("=" ^ 70)
    println("ROBUST MINIMUM VOLUME ENCLOSING ELLIPSOID")
    println("=" ^ 70)
    
    # Generate random points in 3D with some outliers
    n_inliers = 50
    n_outliers = 5
    
    # Inliers: points in a cluster
    inliers = [(randn() * 2, randn() * 1.5 + 1, randn() * 1.8) for _ in 1:n_inliers]
    
    # Outliers: points far from the cluster
    outliers = [
        (10.0, 5.0, 3.0),
        (-8.0, -4.0, 6.0),
        (6.0, 8.0, -5.0),
        (-5.0, 7.0, 4.0),
        (3.0, -6.0, 8.0)
    ]
    
    points = vcat(inliers, outliers)
    n_points = length(points)
    
    println("\nGenerated $n_points points ($n_inliers inliers + $n_outliers outliers)")
    
    # ========================================================================
    # Robust MVEE (automatic outlier detection)
    # ========================================================================
    println("\n" * "-" ^ 50)
    println("ROBUST MVEE (iteratively reweighted)")
    println("-" ^ 50)
    
    time_rob = @elapsed begin
        P_rob, c_rob, weights_rob, history = robust_mvee(points; max_iter=10, threshold=0.01)
    end
    center_rob = P_rob \ c_rob
    svd_rob = svd(P_rob)
    axes_rob = 1.0 ./ svd_rob.S
    
    println("\n" * "=" ^ 70)
    println("FINAL RESULTS")
    println("=" ^ 70)
    println("  Center: $(round.(center_rob, digits=3))")
    println("  Semi-axes: $(round.(axes_rob, digits=3))")
    println("  det(P): $(round(det(P_rob), digits=6))")
    println("  Volume ∝ 1/det(P) = $(round(1/det(P_rob), digits=2))")
    println("  Runtime: $(round(time_rob, digits=4)) seconds")
    
    # Show all weights
    println("\nFinal weights for all points:")
    for i in 1:n_points
        point_type = i <= n_inliers ? "inlier" : "outlier"
        println("  Point $i ($point_type): weight = $(round(weights_rob[i], digits=4))")
    end
    
    # ========================================================================
    # Export data for visualization
    # ========================================================================
    data_export = Dict(
        "points" => [collect(Float64, p) for p in points],
        "n_inliers" => n_inliers,
        "n_outliers" => n_outliers,
        "robust" => Dict(
            "P" => [collect(P_rob[i, :]) for i in 1:3],
            "c" => collect(c_rob),
            "center" => collect(center_rob),
            "semi_axes" => collect(axes_rob),
            "weights" => collect(weights_rob)
        )
    )
    
    open("weighted_mvee_data.json", "w") do f
        JSON.print(f, data_export)
    end
    println("\nData exported to 'weighted_mvee_data.json'")
    println("Run: python3 plot_weighted_mvee.py")
    
    return points, P_rob, c_rob, weights_rob
end

# Run the demo
demo_robust_mvee()
