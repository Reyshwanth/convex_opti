# Rey Ellipsoid Method
# 
# This method uses range measurements from beacons with Gaussian uncertainty
# to estimate a position using the weighted minimum volume enclosing ellipsoid.
#
# Algorithm:
# 1. Generate n beacons with range measurements (Gaussian noise)
# 2. For each pair of beacons, check if their spheres intersect
# 3. If they intersect, sample k points uniformly from the intersection circle
# 4. Assign weight = sqrt(1/(σ₁² + σ₂²)) to each point
# 5. Normalize weights to [0, 1]
# 6. Solve weighted MVEE to find the Rey Ellipsoid
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

function weighted_mvee_sdp(points, weights; verbose=true)
    """
    Compute the Weighted Minimum Volume Enclosing Ellipsoid using SDP.
    
    The ellipsoid is represented as: E = {x | ||Px - c||_2 <= 1}
    
    Formulation:
        max  log(det(P))
        s.t. w_i * ||P*x_i - c||_2 <= 1,  for all points x_i with w_i > 0
             P ≽ 0
    """
    d = length(points[1])
    m = length(points)
    
    @assert length(weights) == m "Number of weights must match number of points"
    @assert all(w -> w >= 0, weights) "All weights must be non-negative"
    
    X = hcat([collect(Float64, p) for p in points]...)
    
    P = Semidefinite(d)
    c = Variable(d)
    
    constraints = Constraint[]
    
    active_count = 0
    for i in 1:m
        if weights[i] > 0
            xi = X[:, i]
            wi = weights[i]
            push!(constraints, wi * norm(P * xi - c) <= 1)
            active_count += 1
        end
    end
    
    if verbose
        println("  Active constraints: $active_count / $m points")
    end
    
    problem = maximize(logdet(P), constraints)
    solve!(problem, SCS.Optimizer; silent=true)
    
    return evaluate(P), evaluate(c), problem.status
end

# ============================================================================
# Sphere-Sphere Intersection
# ============================================================================

function spheres_intersect(c1, r1, c2, r2)
    """
    Check if two spheres intersect and return intersection circle parameters.
    
    Returns:
        intersects: Bool - whether they intersect
        circle_center: 3D point - center of intersection circle
        circle_normal: 3D vector - normal to the circle plane
        circle_radius: Float - radius of intersection circle
    """
    d = norm(c2 - c1)  # distance between centers
    
    # Check if spheres are too far apart or one contains the other
    if d > r1 + r2 || d < abs(r1 - r2) || d == 0
        return false, nothing, nothing, nothing
    end
    
    # Distance from c1 to the intersection plane
    # Using: h = (r1² - r2² + d²) / (2d)
    h = (r1^2 - r2^2 + d^2) / (2 * d)
    
    # Radius of intersection circle
    # r_circle² = r1² - h²
    r_circle_sq = r1^2 - h^2
    
    if r_circle_sq < 0
        return false, nothing, nothing, nothing
    end
    
    r_circle = sqrt(r_circle_sq)
    
    # Unit vector from c1 to c2
    n = (c2 - c1) / d
    
    # Center of intersection circle
    circle_center = c1 + h * n
    
    return true, circle_center, n, r_circle
end

# ============================================================================
# Sample Points Uniformly from Circle in 3D
# ============================================================================

function sample_circle_points(center, normal, radius, k)
    """
    Sample k points uniformly from a circle in 3D.
    
    Args:
        center: 3D center of the circle
        normal: Normal vector to the circle plane
        radius: Radius of the circle
        k: Number of points to sample
    
    Returns:
        Vector of k 3D points
    """
    # Create two orthonormal vectors in the plane of the circle
    n = normalize(normal)
    
    # Find a vector not parallel to n
    if abs(n[1]) < 0.9
        v = [1.0, 0.0, 0.0]
    else
        v = [0.0, 1.0, 0.0]
    end
    
    # Create orthonormal basis in the plane
    u1 = normalize(cross(n, v))
    u2 = cross(n, u1)
    
    # Sample k points uniformly around the circle
    points = Vector{Vector{Float64}}()
    for i in 0:(k-1)
        θ = 2π * i / k
        p = center + radius * (cos(θ) * u1 + sin(θ) * u2)
        push!(points, p)
    end
    
    return points
end

# ============================================================================
# Rey Ellipsoid Method
# ============================================================================

function rey_ellipsoid(; n_beacons=10, k_samples=8, seed=42)
    """
    The Rey Ellipsoid Method for position estimation.
    
    Uses range measurements from beacons with Gaussian uncertainty to estimate
    position via weighted MVEE on intersection circle samples.
    
    Args:
        n_beacons: Number of beacons
        k_samples: Number of points to sample from each intersection circle
        seed: Random seed for reproducibility
    
    Returns:
        P, c: Ellipsoid parameters
        points: All sampled points
        weights: Normalized weights for each point
        beacons: Beacon positions and measurements
    """
    Random.seed!(seed)
    
    println("=" ^ 70)
    println("REY ELLIPSOID METHOD")
    println("=" ^ 70)
    
    # ========================================================================
    # Generate true position and beacons
    # ========================================================================
    
    # True position (unknown, we're trying to estimate it)
    true_position = [5.0, 3.0, 2.0]
    
    # Generate beacon positions randomly around the area
    beacons = []
    for i in 1:n_beacons
        # Beacons in a 20x20x20 cube
        pos = [rand() * 20 - 10, rand() * 20 - 10, rand() * 20 - 10]
        
        # True distance to the position
        true_dist = norm(pos - true_position)
        
        # Measurement noise (standard deviation proportional to distance)
        σ = 0.1 + rand() * 0.2  # Variable noise per beacon
        
        # Noisy range measurement (Gaussian)
        measured_dist = true_dist + randn() * σ
        measured_dist = max(measured_dist, 0.1)  # Ensure positive
        
        push!(beacons, (position=pos, measured_range=measured_dist, sigma=σ, true_range=true_dist))
    end
    
    println("\nGenerated $n_beacons beacons")
    println("True position: $true_position")
    println("\nBeacon measurements:")
    for (i, b) in enumerate(beacons)
        println("  Beacon $i: pos=$(round.(b.position, digits=2)), " *
                "range=$(round(b.measured_range, digits=2)), " *
                "σ=$(round(b.sigma, digits=2))")
    end
    
    # ========================================================================
    # Find sphere intersections and sample points
    # ========================================================================
    
    println("\n" * "-" ^ 50)
    println("Finding sphere intersections...")
    println("-" ^ 50)
    
    all_points = Vector{Vector{Float64}}()
    all_weights = Float64[]
    intersection_info = []
    
    n_intersections = 0
    n_pairs = 0
    
    for i in 1:n_beacons
        for j in (i+1):n_beacons
            n_pairs += 1
            
            b1 = beacons[i]
            b2 = beacons[j]
            
            intersects, circle_center, circle_normal, circle_radius = 
                spheres_intersect(b1.position, b1.measured_range, 
                                  b2.position, b2.measured_range)
            
            if intersects && circle_radius > 0.01  # Ignore tiny circles
                n_intersections += 1
                
                # Sample k points from the intersection circle
                circle_points = sample_circle_points(circle_center, circle_normal, 
                                                      circle_radius, k_samples)
                
                # Weight = sqrt(1/(σ₁² + σ₂²))
                weight = sqrt(1.0 / (b1.sigma^2 + b2.sigma^2))
                
                # Add points with their weights
                for p in circle_points
                    push!(all_points, p)
                    push!(all_weights, weight)
                end
                
                push!(intersection_info, (
                    beacon1=i, beacon2=j,
                    center=circle_center, 
                    normal=circle_normal,
                    radius=circle_radius,
                    weight=weight,
                    n_points=k_samples
                ))
            end
        end
    end
    
    println("Total beacon pairs: $n_pairs")
    println("Intersecting pairs: $n_intersections")
    println("Total sampled points: $(length(all_points))")
    
    if length(all_points) < 4
        error("Not enough intersection points! Need at least 4 points for 3D ellipsoid.")
    end
    
    # ========================================================================
    # Normalize weights to [0, 1]
    # ========================================================================
    
    w_min = minimum(all_weights)
    w_max = maximum(all_weights)
    
    if w_max > w_min
        normalized_weights = (all_weights .- w_min) ./ (w_max - w_min)
    else
        normalized_weights = ones(length(all_weights))
    end
    
    # Shift to avoid zero weights (which would ignore points)
    # Map [0,1] to [0.1, 1.0] to keep all points active but with varying importance
    normalized_weights = 0.1 .+ 0.9 .* normalized_weights
    
    println("\nWeight statistics (before normalization):")
    println("  Raw weight range: [$(round(w_min, digits=4)), $(round(w_max, digits=4))]")
    println("  Normalized range: [0.1, 1.0]")
    
    # ========================================================================
    # Solve Weighted MVEE - The Rey Ellipsoid
    # ========================================================================
    
    println("\n" * "-" ^ 50)
    println("Solving Weighted MVEE (Rey Ellipsoid)...")
    println("-" ^ 50)
    
    time_elapsed = @elapsed begin
        P, c, status = weighted_mvee_sdp(all_points, normalized_weights)
    end
    
    center = P \ c
    svd_result = svd(P)
    semi_axes = 1.0 ./ svd_result.S
    
    println("\n" * "=" ^ 70)
    println("REY ELLIPSOID RESULTS")
    println("=" ^ 70)
    println("  Status: $status")
    println("  Estimated center: $(round.(center, digits=3))")
    println("  True position:    $true_position")
    println("  Position error:   $(round(norm(center - true_position), digits=3))")
    println("  Semi-axes: $(round.(semi_axes, digits=3))")
    println("  det(P): $(round(det(P), digits=6))")
    println("  Runtime: $(round(time_elapsed, digits=4)) seconds")
    
    # ========================================================================
    # Export data for visualization
    # ========================================================================
    
    data_export = Dict(
        "true_position" => true_position,
        "beacons" => [Dict(
            "position" => b.position,
            "measured_range" => b.measured_range,
            "sigma" => b.sigma,
            "true_range" => b.true_range
        ) for b in beacons],
        "intersections" => [Dict(
            "beacon1" => info.beacon1,
            "beacon2" => info.beacon2,
            "center" => collect(info.center),
            "normal" => collect(info.normal),
            "radius" => info.radius,
            "weight" => info.weight
        ) for info in intersection_info],
        "sampled_points" => [collect(p) for p in all_points],
        "normalized_weights" => normalized_weights,
        "rey_ellipsoid" => Dict(
            "P" => [collect(P[i, :]) for i in 1:3],
            "c" => collect(c),
            "center" => collect(center),
            "semi_axes" => collect(semi_axes),
            "position_error" => norm(center - true_position)
        ),
        "n_beacons" => n_beacons,
        "k_samples" => k_samples,
        "n_intersections" => n_intersections
    )
    
    open("rey_ellipsoid_data.json", "w") do f
        JSON.print(f, data_export)
    end
    println("\nData exported to 'rey_ellipsoid_data.json'")
    println("Run: python3 plot_rey_ellipsoid.py")
    
    return P, c, all_points, normalized_weights, beacons, true_position
end

# Run the Rey Ellipsoid method
rey_ellipsoid(n_beacons=10, k_samples=4)
