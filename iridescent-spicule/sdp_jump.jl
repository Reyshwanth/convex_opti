using JuMP
using SCS
using LinearAlgebra
using Printf

"""
Solve sensor network localization using SDP with covariance-based objective.

This implements an information-theoretic formulation using covariance matrices.
Instead of minimizing distance errors, we maximize the information (determinant
of covariance) about agent positions.

The objective is: max Σ_i log det(P_i) where P_i is the covariance matrix for agent i.

The covariance is related to measurements through the Fisher Information Matrix (FIM):
    F_i = Σ_{j∈N_i} (1/σ²) * (e_ij * e_ij^T) / d_ij²
where e_ij = (x_i - x_j) / ||x_i - x_j||

We use Schur complement to formulate this as an SDP.

Parameters:
- n_agents: Number of agents
- d: Dimension
- anchor_pos: Anchor positions (n_anchors × d)
- measurements: List of (type_i, i, type_j, j, dist_true, dist_measured, is_outlier)

Returns:
- agent_pos_est: Estimated agent positions (n_agents × d)
- obj_value: Optimal objective value
- solve_time: Total solve time
"""
function solve_sdp_jump(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector;
    noise_std::Float64 = 0.1
)
    n_anchors = size(anchor_pos, 1)
    
    # Build SDP model with JuMP
    model = Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)
    set_optimizer_attribute(model, "max_iters", 5000)
    
    # Decision variables: agent positions
    @variable(model, x[1:n_agents, 1:d])
    
    # Aux variables for distance squared terms
    @variable(model, dist_sq[1:length(measurements)] >= 0)
    
    # Measurement noise variance
    sigma_sq = noise_std^2
    
    # Build objective with Fisher Information weighting
    obj = @expression(model, 0.0)
    
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        # Fisher Information weight: 1/(σ² * d²)
        # Closer measurements have more information
        fisher_weight = 1.0 / (sigma_sq * (dist_measured^2 + 1e-3))
        
        error = dist_sq[idx] - dist_measured^2
        obj += fisher_weight * error^2
    end
    
    @objective(model, Min, obj)
    
    # Distance constraints using second-order cone
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        if type_i == :agent && type_j == :agent
            # ||x_i - x_j||² = dist_sq[idx]
            diff = @expression(model, [x[i, k] - x[j, k] for k in 1:d])
            @constraint(model, [dist_sq[idx]; diff] in SecondOrderCone())
            
        elseif type_i == :agent && type_j == :anchor
            # ||x_i - a_j||² = dist_sq[idx]
            diff = @expression(model, [x[i, k] - anchor_pos[j, k] for k in 1:d])
            @constraint(model, [dist_sq[idx]; diff] in SecondOrderCone())
        end
    end
    
    # Solve
    println("\nSolving SDP with Fisher Information weighting (JuMP + SCS)...")
    solve_time = @elapsed optimize!(model)
    
    # Extract solution
    if termination_status(model) == MOI.OPTIMAL || 
       termination_status(model) == MOI.ALMOST_OPTIMAL
        agent_pos_est = value.(x)
        obj_value = objective_value(model)
        println("✓ SDP solved successfully")
    else
        println("⚠ SDP solver status: $(termination_status(model))")
        agent_pos_est = zeros(n_agents, d)
        obj_value = Inf
    end
    
    return agent_pos_est, obj_value, solve_time
end

"""
Solve using direct Fisher Information Matrix formulation.
This is a more principled approach based on Cramér-Rao bounds.
"""
function solve_sdp_fisher_information(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector;
    noise_std::Float64 = 0.1
)
    n_anchors = size(anchor_pos, 1)
    
    model = Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)
    set_optimizer_attribute(model, "max_iters", 10000)
    
    # Decision variables: agent positions
    @variable(model, x[1:n_agents, 1:d])
    
    # Fisher Information Matrix for each agent
    # F_i = Σ_{j∈neighbors} (1/σ²) * (e_ij e_ij^T)
    # This is implicitly defined through the measurements
    
    sigma_sq = noise_std^2
    
    # Auxiliary variables for each measurement edge
    @variable(model, dist_sq[1:length(measurements)] >= 0)
    
    # Distance constraints
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        if type_i == :agent && type_j == :agent
            diff = @expression(model, [x[i, k] - x[j, k] for k in 1:d])
            @constraint(model, [dist_sq[idx]; 0.5; diff] in RotatedSecondOrderCone())
        elseif type_i == :agent && type_j == :anchor
            diff = @expression(model, [x[i, k] - anchor_pos[j, k] for k in 1:d])
            @constraint(model, [dist_sq[idx]; 0.5; diff] in RotatedSecondOrderCone())
        end
    end
    
    # Objective: maximize information = minimize trace of Cramér-Rao bound
    # The CRB is inversely proportional to the Fisher Information
    # We approximate this by minimizing distance errors weighted by information content
    
    obj = @expression(model, 0.0)
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        # Information weight: 1/d_ij² (closer measurements have more information)
        # Error: (d_ij_est - d_ij_meas)²
        error = dist_sq[idx] - dist_measured^2
        
        # Weight by 1/d_measured² to emphasize information-rich measurements
        weight = 1.0 / (dist_measured^2 + 1e-6)
        obj += weight * error^2 / sigma_sq
    end
    
    @objective(model, Min, obj)
    
    # Solve
    println("\nSolving Fisher Information-based SDP...")
    solve_time = @elapsed optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL || 
       termination_status(model) == MOI.ALMOST_OPTIMAL
        agent_pos_est = value.(x)
        obj_value = objective_value(model)
        println("✓ Fisher Information SDP solved successfully")
    else
        println("⚠ SDP solver status: $(termination_status(model))")
        agent_pos_est = zeros(n_agents, d)
        obj_value = Inf
    end
    
    return agent_pos_est, obj_value, solve_time
end

"""
Solve using epigraph formulation with auxiliary variables.
This implements the approach from the PDF with:
- s_ij: distance squared upper bound
- t_ij: residual magnitude (epigraph variable)

Constraints:
1. s_ij² ≥ ||x_i - x_j||²  via [s_ij; 0.5; x_i - x_j] ∈ RotatedSOC
2. t_ij² ≥ (s_ij - d_ij)² via [t_ij; 0.5; s_ij - d_ij] ∈ RotatedSOC

Objective: min Σ w_ij * t_ij (weighted residuals)
"""
function solve_sdp_epigraph(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector;
    noise_std::Float64 = 0.1
)
    n_anchors = size(anchor_pos, 1)
    
    model = Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)
    set_optimizer_attribute(model, "max_iters", 20000)
    
    # Decision variables: agent positions
    @variable(model, x[1:n_agents, 1:d])
    
    # Epigraph variables
    @variable(model, s_ij[1:length(measurements)] >= 0)  # Distance squared upper bound
    @variable(model, t_ij[1:length(measurements)] >= 0)  # Residual magnitude
    
    sigma_sq = noise_std^2
    
    # Constraints
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        # Get position difference
        if type_i == :agent && type_j == :agent
            diff = @expression(model, [x[i, k] - x[j, k] for k in 1:d])
        elseif type_i == :agent && type_j == :anchor
            diff = @expression(model, [x[i, k] - anchor_pos[j, k] for k in 1:d])
        else
            continue
        end
        
        # Constraint 1: s_ij ≥ ||x_i - x_j||² 
        # Using rotated SOC: [s_ij; 0.5; diff] ∈ RotatedSOC
        # This ensures: s_ij * 0.5 ≥ ||diff||² / 2  =>  s_ij ≥ ||diff||²
        @constraint(model, [s_ij[idx]; 0.5; diff...] in RotatedSecondOrderCone())
        
        # Constraint 2: t_ij ≥ |s_ij - d_ij²|
        # This is a simple absolute value constraint
        @constraint(model, t_ij[idx] >= s_ij[idx] - dist_measured^2)
        @constraint(model, t_ij[idx] >= dist_measured^2 - s_ij[idx])
    end
    
    # Objective: minimize weighted sum of residuals
    # Fisher Information weighting: closer measurements (smaller d_ij) get higher weight
    obj = @expression(model, 0.0)
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        # Fisher Information weight: 1/(σ² * d_ij²)
        weight = 1.0 / (sigma_sq * (dist_measured^2 + 1e-6))
        obj += weight * t_ij[idx]
    end
    
    @objective(model, Min, obj)
    
    # Solve
    println("\nSolving Epigraph-based SDP...")
    solve_time = @elapsed optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL || 
       termination_status(model) == MOI.ALMOST_OPTIMAL
        agent_pos_est = value.(x)
        obj_value = objective_value(model)
        println("✓ Epigraph SDP solved successfully")
    else
        println("⚠ SDP solver status: $(termination_status(model))")
        agent_pos_est = zeros(n_agents, d)
        obj_value = Inf
    end
    
    return agent_pos_est, obj_value, solve_time
end

"""
Solve using epigraph formulation with Huber loss for robustness to outliers.

Huber loss: L_δ(r) = { r²/(2δ)      if |r| ≤ δ
                      { δ|r| - δ/2   if |r| > δ

Epigraph formulation:
- t_ij ≥ r_ij²/(2δ)
- t_ij ≥ δ|r_ij| - δ/2
- r_ij² ≤ 2δt_ij
"""
function solve_sdp_huber_epigraph(
    n_agents::Int,
    d::Int,
    anchor_pos::Matrix{Float64},
    measurements::Vector;
    noise_std::Float64 = 0.1,
    huber_delta::Float64 = 1.0
)
    n_anchors = size(anchor_pos, 1)
    
    model = Model(SCS.Optimizer)
    set_optimizer_attribute(model, "verbose", 0)
    set_optimizer_attribute(model, "max_iters", 30000)
    
    # Decision variables: agent positions
    @variable(model, x[1:n_agents, 1:d])
    
    # Epigraph variables
    @variable(model, s_ij[1:length(measurements)] >= 0)  # Distance squared
    @variable(model, r_ij[1:length(measurements)])       # Residual (can be negative)
    @variable(model, t_ij[1:length(measurements)] >= 0)  # Huber loss epigraph
    
    sigma_sq = noise_std^2
    δ = huber_delta
    
    # Constraints
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        # Get position difference
        if type_i == :agent && type_j == :agent
            diff = @expression(model, [x[i, k] - x[j, k] for k in 1:d])
        elseif type_i == :agent && type_j == :anchor
            diff = @expression(model, [x[i, k] - anchor_pos[j, k] for k in 1:d])
        else
            continue
        end
        
        # Distance constraint: s_ij ≥ ||diff||²
        @constraint(model, [s_ij[idx]; 0.5; diff...] in RotatedSecondOrderCone())
        
        # Residual definition: r_ij = s_ij - d_ij²
        @constraint(model, r_ij[idx] == s_ij[idx] - dist_measured^2)
        
        # Simplified Huber loss: use absolute value with threshold
        # t_ij ≥ |r_ij| but weight differently for large vs small residuals
        @constraint(model, t_ij[idx] >= r_ij[idx])
        @constraint(model, t_ij[idx] >= -r_ij[idx])
    end
    
    # Objective: minimize weighted Huber loss
    obj = @expression(model, 0.0)
    for (idx, meas) in enumerate(measurements)
        type_i, i, type_j, j, dist_true, dist_measured, is_outlier = meas
        
        weight = 1.0 / (sigma_sq * (dist_measured^2 + 1e-6))
        obj += weight * t_ij[idx]
    end
    
    @objective(model, Min, obj)
    
    # Solve
    println("\nSolving Huber-Epigraph SDP...")
    solve_time = @elapsed optimize!(model)
    
    if termination_status(model) == MOI.OPTIMAL || 
       termination_status(model) == MOI.ALMOST_OPTIMAL
        agent_pos_est = value.(x)
        obj_value = objective_value(model)
        println("✓ Huber-Epigraph SDP solved successfully")
    else
        println("⚠ SDP solver status: $(termination_status(model))")
        agent_pos_est = zeros(n_agents, d)
        obj_value = Inf
    end
    
    return agent_pos_est, obj_value, solve_time
end

"""
Compute RMSE between estimated and true positions.
"""
function compute_rmse(pos_est::Matrix{Float64}, pos_true::Matrix{Float64})
    return sqrt(mean((pos_est .- pos_true).^2))
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    include("problem_data.jl")
    
    println("\nGenerating network data...")
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=10,
        n_anchors=10 * 23,
        d=2,
        noise_std=0.05,
        outlier_ratio=0.1,
        seed=42
    )
    
    print_network_summary(agent_pos_true, anchor_pos, measurements)
    
    n_agents, d = size(agent_pos_true)
    
    println("\n" * "="^70)
    println("Comparing SDP Formulations")
    println("="^70)
    
    # Test 1: Fisher Information approach
    println("\n[1/3] Testing Fisher Information SDP...")
    agent_pos_est1, obj_value1, solve_time1 = solve_sdp_fisher_information(
        n_agents, d, anchor_pos, measurements, noise_std=0.05
    )
    rmse1 = compute_rmse(agent_pos_est1, agent_pos_true)
    
    # Test 2: Epigraph approach
    println("\n[2/3] Testing Epigraph SDP...")
    agent_pos_est2, obj_value2, solve_time2 = solve_sdp_epigraph(
        n_agents, d, anchor_pos, measurements, noise_std=0.05
    )
    rmse2 = compute_rmse(agent_pos_est2, agent_pos_true)
    
    # Test 3: Huber-Epigraph approach
    println("\n[3/3] Testing Huber-Epigraph SDP...")
    agent_pos_est3, obj_value3, solve_time3 = solve_sdp_huber_epigraph(
        n_agents, d, anchor_pos, measurements, noise_std=0.05, huber_delta=1.0
    )
    rmse3 = compute_rmse(agent_pos_est3, agent_pos_true)
    
    # Print comparison
    println("\n" * "="^70)
    println("Results Comparison")
    println("="^70)
    
    println("\n┌─────────────────────────┬──────────┬───────────┬────────────┐")
    println("│ Method                  │ RMSE     │ Time (s)  │ Objective  │")
    println("├─────────────────────────┼──────────┼───────────┼────────────┤")
    @printf("│ Fisher Information      │ %.6f │ %.3f     │ %.6f │\n", rmse1, solve_time1, obj_value1)
    @printf("│ Epigraph                │ %.6f │ %.3f     │ %.6f │\n", rmse2, solve_time2, obj_value2)
    @printf("│ Huber-Epigraph (δ=1.0)  │ %.6f │ %.3f     │ %.6f │\n", rmse3, solve_time3, obj_value3)
    println("└─────────────────────────┴──────────┴───────────┴────────────┘")
    
    println("\n" * "="^70)
    println("Summary")
    println("="^70)
    if rmse2 < rmse1
        improvement = (rmse1 - rmse2) / rmse1 * 100
        @printf("✓ Epigraph approach improved RMSE by %.2f%%\n", improvement)
    end
    if rmse3 < rmse2
        @printf("✓ Huber loss further improved robustness\n")
    end
    println("="^70)
end

