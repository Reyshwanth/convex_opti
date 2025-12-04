"""
Circle Point Optimization

Given a set of circles in 3D, find the optimal point that minimizes the sum of 
distances to points on each circle. Each circle point is parameterized by angle θ.

Variables:
- s ∈ ℝ³: the chosen point
- θᵢ ∈ [0, 2π]: angle for each circle i

Objective: minimize Σᵢ ||s - pᵢ(θᵢ)||₂

where pᵢ(θᵢ) = cᵢ + rᵢ(cos(θᵢ)u₁ᵢ + sin(θᵢ)u₂ᵢ)
- cᵢ is the center of circle i
- rᵢ is the radius of circle i
- u₁ᵢ, u₂ᵢ are orthonormal vectors spanning the plane of circle i
"""

using LinearAlgebra
using Optim
using ForwardDiff
using JSON
using Random

# ============================================================================
# Circle representation
# ============================================================================

struct Circle3D
    center::Vector{Float64}    # Center of the circle
    radius::Float64            # Radius
    u1::Vector{Float64}        # First basis vector in the plane
    u2::Vector{Float64}        # Second basis vector in the plane (orthogonal to u1)
    normal::Vector{Float64}    # Normal to the plane
end

"""
Create a circle from center, radius, and normal vector
"""
function Circle3D(center::Vector{Float64}, radius::Float64, normal::Vector{Float64})
    n = normalize(normal)
    
    # Find two orthonormal vectors in the plane
    if abs(n[1]) < 0.9
        u1 = normalize(cross([1.0, 0.0, 0.0], n))
    else
        u1 = normalize(cross([0.0, 1.0, 0.0], n))
    end
    u2 = normalize(cross(n, u1))
    
    return Circle3D(center, radius, u1, u2, n)
end

"""
Get point on circle at angle θ
"""
function point_on_circle(circle::Circle3D, θ::Float64)
    return circle.center + circle.radius * (cos(θ) * circle.u1 + sin(θ) * circle.u2)
end

# ============================================================================
# Optimization
# ============================================================================

"""
Compute sum of distances given:
- s: the chosen point (3D)
- θs: angles for each circle
- circles: array of Circle3D
"""
function total_distance(s::Vector{T}, θs::Vector{T}, circles::Vector{Circle3D}) where T
    total = zero(T)
    for (i, circle) in enumerate(circles)
        p = circle.center + circle.radius * (cos(θs[i]) * circle.u1 + sin(θs[i]) * circle.u2)
        total += norm(s - p)
    end
    return total
end

"""
Pack variables into single vector for optimization: [s; θs]
"""
function objective(x::Vector{T}, circles::Vector{Circle3D}) where T
    s = x[1:3]
    θs = x[4:end]
    return total_distance(s, θs, circles)
end

"""
Find optimal point s and angles θᵢ that minimize sum of distances
"""
function optimize_circle_points(circles::Vector{Circle3D}; 
                                 max_iterations::Int=1000,
                                 show_trace::Bool=false)
    n = length(circles)
    
    # Initial guess: s = centroid of circle centers, θᵢ = 0
    s0 = sum(c.center for c in circles) / n
    θ0 = zeros(n)
    x0 = vcat(s0, θ0)
    
    # Objective function
    f(x) = objective(x, circles)
    
    # Optimize using L-BFGS with autodiff
    result = optimize(f, x0, LBFGS(), 
                     Optim.Options(iterations=max_iterations, 
                                   show_trace=show_trace,
                                   g_tol=1e-10))
    
    # Extract solution
    x_opt = Optim.minimizer(result)
    s_opt = x_opt[1:3]
    θ_opt = x_opt[4:end]
    
    # Compute circle points
    circle_points = [point_on_circle(circles[i], θ_opt[i]) for i in 1:n]
    
    return s_opt, θ_opt, circle_points, result
end

"""
Alternative: Fix s and optimize only θs (Weiszfeld-style iteration)
"""
function optimize_alternating(circles::Vector{Circle3D}; 
                               max_outer_iter::Int=100,
                               tol::Float64=1e-8)
    n = length(circles)
    
    # Initialize s as centroid
    s = sum(c.center for c in circles) / n
    θs = zeros(n)
    
    prev_obj = Inf
    
    for iter in 1:max_outer_iter
        # Step 1: Fix s, optimize each θᵢ independently
        for i in 1:n
            θs[i] = optimize_theta_for_circle(circles[i], s)
        end
        
        # Step 2: Fix θs, optimize s (geometric median of circle points)
        circle_points = [point_on_circle(circles[i], θs[i]) for i in 1:n]
        s = geometric_median(circle_points)
        
        # Check convergence
        obj = sum(norm(s - p) for p in circle_points)
        if abs(prev_obj - obj) < tol
            println("Converged in $iter iterations")
            break
        end
        prev_obj = obj
    end
    
    circle_points = [point_on_circle(circles[i], θs[i]) for i in 1:n]
    return s, θs, circle_points
end

"""
Given a fixed point s, find θ that minimizes ||s - p(θ)||
This has a closed-form solution!
"""
function optimize_theta_for_circle(circle::Circle3D, s::Vector{Float64})
    # Vector from circle center to s
    v = s - circle.center
    
    # Project onto the plane of the circle
    v_proj = v - dot(v, circle.normal) * circle.normal
    
    # Find angle in the u1-u2 coordinate system
    θ = atan(dot(v_proj, circle.u2), dot(v_proj, circle.u1))
    
    return θ
end

"""
Compute geometric median using Weiszfeld's algorithm
"""
function geometric_median(points::Vector{Vector{Float64}}; max_iter::Int=100, tol::Float64=1e-10)
    n = length(points)
    
    # Initialize with centroid
    y = sum(points) / n
    
    for _ in 1:max_iter
        weights = [1.0 / max(norm(y - p), 1e-10) for p in points]
        y_new = sum(weights[i] * points[i] for i in 1:n) / sum(weights)
        
        if norm(y_new - y) < tol
            break
        end
        y = y_new
    end
    
    return y
end

# ============================================================================
# Demo
# ============================================================================

function demo()
    Random.seed!(42)
    
    println("=" ^ 70)
    println("CIRCLE POINT OPTIMIZATION")
    println("=" ^ 70)
    
    # Create some random circles in 3D
    n_circles = 10
    circles = Circle3D[]
    
    println("\nGenerating $n_circles random circles...")
    
    for i in 1:n_circles
        center = 10 * randn(3)
        radius = 1.0 + 2.0 * rand()
        normal = normalize(randn(3))
        push!(circles, Circle3D(center, radius, normal))
    end
    
    # Print circle info
    println("\nCircle details:")
    for (i, c) in enumerate(circles)
        println("  Circle $i: center=$(round.(c.center, digits=2)), r=$(round(c.radius, digits=2))")
    end
    
    # Method 1: Joint optimization
    println("\n" * "-" ^ 50)
    println("Method 1: Joint Optimization (L-BFGS)")
    println("-" ^ 50)
    
    time1 = @elapsed begin
        s1, θ1, points1, result1 = optimize_circle_points(circles)
    end
    
    obj1 = sum(norm(s1 - p) for p in points1)
    println("  Optimal point s: $(round.(s1, digits=4))")
    println("  Total distance: $(round(obj1, digits=6))")
    println("  Optimizer status: $(Optim.converged(result1) ? "Converged" : "Not converged")")
    println("  Iterations: $(Optim.iterations(result1))")
    println("  Runtime: $(round(time1 * 1000, digits=2)) ms")
    
    # Method 2: Alternating optimization
    println("\n" * "-" ^ 50)
    println("Method 2: Alternating Optimization")
    println("-" ^ 50)
    
    time2 = @elapsed begin
        s2, θ2, points2 = optimize_alternating(circles)
    end
    
    obj2 = sum(norm(s2 - p) for p in points2)
    println("  Optimal point s: $(round.(s2, digits=4))")
    println("  Total distance: $(round(obj2, digits=6))")
    println("  Runtime: $(round(time2 * 1000, digits=2)) ms")
    
    # Comparison
    println("\n" * "=" ^ 70)
    println("COMPARISON")
    println("=" ^ 70)
    println("Method              | Total Distance | Runtime (ms)")
    println("-" ^ 55)
    println("Joint (L-BFGS)      | $(round(obj1, digits=6))       | $(round(time1 * 1000, digits=2))")
    println("Alternating         | $(round(obj2, digits=6))       | $(round(time2 * 1000, digits=2))")
    println("\nDifference in s: $(round(norm(s1 - s2), digits=6))")
    
    # Export for visualization
    data = Dict(
        "circles" => [Dict(
            "center" => c.center,
            "radius" => c.radius,
            "u1" => c.u1,
            "u2" => c.u2,
            "normal" => c.normal
        ) for c in circles],
        "optimal_point" => s1,
        "angles" => θ1,
        "circle_points" => points1,
        "total_distance" => obj1
    )
    
    open("circle_optimization_data.json", "w") do f
        JSON.print(f, data, 2)
    end
    
    println("\nData exported to 'circle_optimization_data.json'")
    
    return s1, θ1, points1, circles
end

# Run demo
demo()
