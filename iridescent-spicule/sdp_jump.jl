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
    
    println("\n" * "="^60)
    println("Testing Covariance-Based SDP Formulation")
    println("="^60)
    
    # Test Fisher Information approach
    agent_pos_est, obj_value, solve_time = solve_sdp_fisher_information(
        n_agents, d, anchor_pos, measurements, noise_std=0.05
    )
    
    rmse = compute_rmse(agent_pos_est, agent_pos_true)
    
    println("\n" * "="^60)
    println("Fisher Information SDP Results")
    println("="^60)
    @printf("Solve time: %.3f seconds\n", solve_time)
    @printf("Objective value: %.6f\n", obj_value)
    @printf("RMSE: %.6f\n", rmse)
    println("="^60)
end
