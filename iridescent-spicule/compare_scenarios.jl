#!/usr/bin/env julia

"""
Compare different dynamic simulation scenarios.
Runs multiple configurations and generates comparison visualizations.
"""

include("dynamic_simulation.jl")

using Printf
using Plots
using Statistics

"""
Run comparison across different scenarios.
"""
function compare_scenarios()
    println("\n" * "="^70)
    println("DYNAMIC SIMULATION SCENARIO COMPARISON")
    println("="^70)
    
    scenarios = [
        (name="Circular Motion", trajectory=:circular, rc=5.0),
        (name="Linear Motion", trajectory=:linear, rc=5.0),
        (name="Varied Motion", trajectory=:varied, rc=5.0),
        (name="Circular (Extended Range)", trajectory=:circular, rc=8.0),
    ]
    
    results_list = []
    
    for (idx, scenario) in enumerate(scenarios)
        println("\n[Scenario $idx/$(length(scenarios))]: $(scenario.name)")
        println("-" * "^"^60)
        
        results = run_dynamic_simulation(
            n_agents=5,
            n_anchors=5 * 23,
            duration=10.0,
            dt=0.1,
            rc=scenario.rc,
            trajectory_type=scenario.trajectory,
            use_static_init=true,
            seed=42
        )
        
        push!(results_list, (scenario=scenario, results=results))
    end
    
    # Create comparison visualizations
    visualize_scenario_comparison(results_list)
    
    return results_list
end

"""
Visualize comparison of different scenarios.
"""
function visualize_scenario_comparison(results_list)
    n_scenarios = length(results_list)
    
    # Extract metrics - compute from actual results
    scenario_names = [r.scenario.name for r in results_list]
    
    # Compute metrics from position errors
    final_pos_rmse = []
    avg_pos_error = []
    max_pos_error = []
    avg_vel_error = []
    avg_neighbors = []
    
    for r in results_list
        # Position RMSE: sqrt of mean squared error at final timestep
        final_errors = r.results.position_errors[:, end]
        push!(final_pos_rmse, sqrt(mean(final_errors.^2)))
        push!(avg_pos_error, mean(final_errors))
        push!(max_pos_error, maximum(final_errors))
        
        # Velocity error at final timestep
        push!(avg_vel_error, mean(r.results.velocity_errors[:, end]))
        
        # Average neighbors across all time
        push!(avg_neighbors, mean(r.results.neighbor_counts))
    end
    
    # Create comprehensive comparison plot
    plt = plot(size=(1400, 1000), layout=(3, 2))
    
    # 1. Final Position RMSE comparison (bar chart)
    bar!(plt[1], scenario_names, final_pos_rmse,
         xlabel="", ylabel="RMSE (m)",
         title="Final Position RMSE",
         legend=false, color=:steelblue,
         xrotation=45)
    
    # 2. Position Error Statistics
    x_pos = 1:n_scenarios
    bar!(plt[2], x_pos .- 0.2, avg_pos_error, 
         bar_width=0.35, label="Average", color=:coral)
    bar!(plt[2], x_pos .+ 0.2, max_pos_error,
         bar_width=0.35, label="Maximum", color=:crimson)
    plot!(plt[2], xlabel="", ylabel="Error (m)",
          title="Position Error Statistics",
          xticks=(x_pos, scenario_names), xrotation=45)
    
    # 3. Velocity Error comparison
    bar!(plt[3], scenario_names, avg_vel_error,
         xlabel="", ylabel="Velocity Error (m/s)",
         title="Final Velocity Error",
         legend=false, color=:seagreen,
         xrotation=45)
    
    # 4. Network Connectivity
    bar!(plt[4], scenario_names, avg_neighbors,
         xlabel="", ylabel="Avg Neighbors",
         title="Average Network Connectivity",
         legend=false, color=:purple,
         xrotation=45)
    
    # 5. Position Error Evolution for all scenarios
    colors = [:blue, :red, :green, :orange, :purple]
    plot!(plt[5], xlabel="Time Step", ylabel="Avg Position Error (m)",
          title="Position Error Evolution", legend=:topleft)
    
    for (idx, r) in enumerate(results_list)
        avg_error_over_time = vec(mean(r.results.position_errors, dims=1))
        plot!(plt[5], avg_error_over_time,
              label=r.scenario.name,
              color=colors[idx], linewidth=2)
    end
    
    # 6. Trajectories for first scenario as example
    plot!(plt[6], xlabel="X (m)", ylabel="Y (m)",
          title="Example: $(scenario_names[1]) Trajectories",
          aspect_ratio=:equal, legend=:topright)
    
    first_result = results_list[1].results
    # Plot anchors
    scatter!(plt[6], first_result.anchors[:, 1], first_result.anchors[:, 2],
             color=:black, marker=:square, markersize=3,
             label="Anchors", alpha=0.3)
    
    # Plot trajectories
    for i in 1:length(first_result.true_trajectories)
        true_traj = hcat(first_result.true_trajectories[i]...)'
        est_traj = hcat(first_result.est_trajectories[i]...)'
        
        plot!(plt[6], true_traj[:, 1], true_traj[:, 2],
              color=colors[i], linewidth=2, linestyle=:solid,
              label=i==1 ? "True" : "")
        plot!(plt[6], est_traj[:, 1], est_traj[:, 2],
              color=colors[i], linewidth=2, linestyle=:dash,
              label=i==1 ? "Estimated" : "")
    end
    
    savefig(plt, "scenario_comparison.png")
    println("\n✓ Comparison visualization saved to scenario_comparison.png")
    
    # Print summary table
    println("\n" * "="^70)
    println("SCENARIO COMPARISON SUMMARY")
    println("="^70)
    println("\n┌────────────────────────────┬──────────┬──────────┬──────────┬──────────┐")
    println("│ Scenario                   │ Pos RMSE │ Vel Err  │ Neighbors│ Max Err  │")
    println("├────────────────────────────┼──────────┼──────────┼──────────┼──────────┤")
    
    for i in 1:n_scenarios
        @printf("│ %-26s │ %8.2f │ %8.2f │ %8.2f │ %8.2f │\n",
                scenario_names[i], final_pos_rmse[i], avg_vel_error[i],
                avg_neighbors[i], max_pos_error[i])
    end
    println("└────────────────────────────┴──────────┴──────────┴──────────┴──────────┘")
    
    # Find best scenarios
    best_pos = argmin(final_pos_rmse)
    best_vel = argmin(avg_vel_error)
    best_conn = argmax(avg_neighbors)
    
    println("\n" * "="^70)
    println("KEY FINDINGS")
    println("="^70)
    println("✓ Best Position Accuracy: $(scenario_names[best_pos])")
    println("✓ Best Velocity Tracking: $(scenario_names[best_vel])")
    println("✓ Best Connectivity: $(scenario_names[best_conn])")
    println("="^70)
    
    return plt
end

# Run comparison
if abspath(PROGRAM_FILE) == @__FILE__
    results = compare_scenarios()
end
