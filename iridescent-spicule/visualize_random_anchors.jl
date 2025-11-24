#!/usr/bin/env julia

"""
Visualize agent trajectories with random anchor placement.
Shows both true paths and estimated positions from localization methods.
"""

using Plots
using Printf
using Random
using LinearAlgebra

include("problem_data.jl")
include("sdp_jump.jl")
include("miqp_outlier.jl")

"""
Generate and visualize static localization with random anchors.
"""
function plot_random_anchor_localization(;
    n_agents::Int=10,
    n_anchors::Int=50,
    noise_std::Float64=0.1,
    outlier_ratio::Float64=0.2,
    seed::Int=42
)
    Random.seed!(seed)
    
    println("\n" * "="^60)
    println("Agent Localization with Random Anchor Placement")
    println("="^60)
    
    # Generate network with random anchors
    println("\nGenerating network with random anchors...")
    agent_pos_true, anchor_pos, measurements = generate_network_data(
        n_agents=n_agents,
        n_anchors=n_anchors,
        d=2,
        noise_std=noise_std,
        outlier_ratio=outlier_ratio,
        seed=seed
    )
    
    print_network_summary(agent_pos_true, anchor_pos, measurements)
    
    # Solve with SDP
    println("\nSolving with SDP Fisher Information method...")
    agent_pos_sdp, _, time_sdp = solve_sdp_jump(
        n_agents, 2, anchor_pos, measurements, noise_std=noise_std
    )
    rmse_sdp = sqrt(mean((agent_pos_sdp .- agent_pos_true).^2))
    @printf("SDP RMSE: %.4f, Time: %.2fs\n", rmse_sdp, time_sdp)
    
    # Solve with MIQP
    println("\nSolving with MIQP...")
    agent_pos_miqp, outliers_detected, _, time_miqp = solve_miqp_outlier_rejection(
        n_agents, 2, anchor_pos, measurements,
        lambda_outlier=2.0,
        warm_start=agent_pos_sdp,
        use_relaxation=true
    )
    rmse_miqp = sqrt(mean((agent_pos_miqp .- agent_pos_true).^2))
    @printf("MIQP RMSE: %.4f, Time: %.2fs\n", rmse_miqp, time_miqp)
    
    # Create visualization
    println("\nCreating visualization...")
    
    # Main comparison plot
    p = plot(layout=(1, 3), size=(1800, 600), 
             xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal)
    
    # Plot 1: Network topology
    plot!(p[1], title="Network Topology\n(Random Anchors)", legend=:outertopright)
    
    # Anchors
    scatter!(p[1], anchor_pos[:, 1], anchor_pos[:, 2], 
            marker=:dtriangle, markersize=6, color=:gray, alpha=0.6, label="Anchors ($n_anchors)")
    
    # True agent positions
    scatter!(p[1], agent_pos_true[:, 1], agent_pos_true[:, 2],
            marker=:circle, markersize=10, color=:blue, label="True Positions")
    
    # Agent labels
    for i in 1:n_agents
        annotate!(p[1], agent_pos_true[i, 1], agent_pos_true[i, 2] + 0.3, 
                 text("$i", 8, :blue))
    end
    
    # Plot 2: SDP Results
    plot!(p[2], title="SDP Fisher Information\nRMSE = $(round(rmse_sdp, digits=3))", 
          legend=:outertopright)
    
    scatter!(p[2], anchor_pos[:, 1], anchor_pos[:, 2],
            marker=:dtriangle, markersize=4, color=:gray, alpha=0.3, label="")
    
    scatter!(p[2], agent_pos_true[:, 1], agent_pos_true[:, 2],
            marker=:circle, markersize=8, color=:blue, alpha=0.7, label="True")
    
    scatter!(p[2], agent_pos_sdp[:, 1], agent_pos_sdp[:, 2],
            marker=:xcross, markersize=10, color=:red, linewidth=3, label="Estimated")
    
    # Error lines
    for i in 1:n_agents
        plot!(p[2], [agent_pos_true[i, 1], agent_pos_sdp[i, 1]],
                    [agent_pos_true[i, 2], agent_pos_sdp[i, 2]],
                    color=:red, alpha=0.3, linestyle=:dash, label="")
    end
    
    # Plot 3: MIQP Results  
    plot!(p[3], title="MIQP Outlier Rejection\nRMSE = $(round(rmse_miqp, digits=3))",
          legend=:outertopright)
    
    scatter!(p[3], anchor_pos[:, 1], anchor_pos[:, 2],
            marker=:dtriangle, markersize=4, color=:gray, alpha=0.3, label="")
    
    scatter!(p[3], agent_pos_true[:, 1], agent_pos_true[:, 2],
            marker=:circle, markersize=8, color=:blue, alpha=0.7, label="True")
    
    scatter!(p[3], agent_pos_miqp[:, 1], agent_pos_miqp[:, 2],
            marker=:xcross, markersize=10, color=:green, linewidth=3, label="Estimated")
    
    # Error lines
    for i in 1:n_agents
        plot!(p[3], [agent_pos_true[i, 1], agent_pos_miqp[i, 1]],
                    [agent_pos_true[i, 2], agent_pos_miqp[i, 2]],
                    color=:green, alpha=0.3, linestyle=:dash, label="")
    end
    
    savefig(p, "random_anchors_localization.png")
    println("✓ Saved random_anchors_localization.png")
    
    # Create a zoomed-in plot showing trajectories/movements
    p2 = plot(title="Agent Trajectories with Random Anchors\n(Static Localization)", 
              xlabel="X (m)", ylabel="Y (m)", aspect_ratio=:equal,
              legend=:outertopright, size=(800, 800))
    
    # Anchors with transparency
    scatter!(p2, anchor_pos[:, 1], anchor_pos[:, 2],
            marker=:dtriangle, markersize=5, color=:gray, alpha=0.4, label="Random Anchors")
    
    # For each agent, show the "trajectory" from estimated to true position
    colors = palette(:tab10)
    for i in 1:n_agents
        # True position
        scatter!(p2, [agent_pos_true[i, 1]], [agent_pos_true[i, 2]],
                marker=:star5, markersize=12, color=colors[mod(i-1, 10)+1], 
                label=i==1 ? "True Positions" : "")
        
        # SDP estimate  
        scatter!(p2, [agent_pos_sdp[i, 1]], [agent_pos_sdp[i, 2]],
                marker=:circle, markersize=8, color=colors[mod(i-1, 10)+1], 
                alpha=0.5, label=i==1 ? "SDP Estimates" : "")
        
        # MIQP estimate
        scatter!(p2, [agent_pos_miqp[i, 1]], [agent_pos_miqp[i, 2]],
                marker=:xcross, markersize=8, color=colors[mod(i-1, 10)+1],
                linewidth=2, label=i==1 ? "MIQP Estimates" : "")
        
        # Trajectories showing estimation error
        plot!(p2, [agent_pos_true[i, 1], agent_pos_sdp[i, 1]],
                  [agent_pos_true[i, 2], agent_pos_sdp[i, 2]],
                  color=colors[mod(i-1, 10)+1], alpha=0.3, linestyle=:dash, label="")
        
        # Agent number
        annotate!(p2, agent_pos_true[i, 1] + 0.2, agent_pos_true[i, 2] + 0.2,
                 text("A$i", 7, :black))
    end
    
    savefig(p2, "agent_trajectories_random_anchors.png")
    println("✓ Saved agent_trajectories_random_anchors.png")
    
    return p, p2, agent_pos_true, anchor_pos, agent_pos_sdp, agent_pos_miqp
end

# Main execution
if abspath(PROGRAM_FILE) == @__FILE__
    println("\n" * "="^60)
    println("Visualizing Localization with Random Anchor Placement")
    println("="^60)
    
    # Generate visualization
    plot_random_anchor_localization(
        n_agents=12,
        n_anchors=60,
        noise_std=0.1,
        outlier_ratio=0.2,
        seed=123
    )
    
    println("\n" * "="^60)
    println("✅ Visualization Complete!")
    println("="^60)
    println("\nGenerated files:")
    println("  • random_anchors_localization.png - Comparison of methods")
    println("  • agent_trajectories_random_anchors.png - Detailed agent view")
    println("\nKey Features:")
    println("  ✓ Random anchor placement (not circular)")
    println("  ✓ Agent trajectories shown as error vectors")
    println("  ✓ Comparison of SDP vs MIQP estimates")
    println("  ✓ Fisher Information-based weighting")
end
