module Simulate

__precompile__()

# this module to store the functions relating to simulating a given grid or model.
# ==============================================================================

using Plots
using LaTeXStrings

using ..SurvivalSignatureUtils
using ..Structures: System, Simulation, Methods, Points, Model
using ..Structures: SimulationType, MonteCarloSimulation, IntervalPredictorSimulation
using ..Structures: MonteCarloModel, PredictorModel
using ..Structures: Metrics, Times

using ..Evaluation
using ..StartingPoints
using ..Centers
using ..ShapeParameter
using ..BasisFunction
using ..AdaptiveRefinement
using ..IntervalPredictorModel
using ..Visualization

# convert this to a Module: RadialBasisFunctions (?)
# needed for monotonicity_constraints and lsqr
include("../src/rbf/radialbasisfunctions.jl")

# ==============================================================================
export simulate
# ==============================================================================

function simulate(
    methods::Vector{Methods},
    sys::System,
    sims::Vector{Simulation};
    verbose::Bool=false,
    save::Bool=false,
    timing::Bool=false,
    plot_sample_space::Bool=false,
    shape_parameter::Union{Nothing,Float64}=nothing,
)::Matrix{Model}

    # Initialize the signatures array
    signatures = Matrix{Model}(undef, (length(methods), length(sims)))

    for (i, method) in enumerate(methods)   # columns
        for (j, sim) in enumerate(sims) #   # rows
            signatures[i, j] = simulate(
                method.simulation_method,
                sys,
                sim,
                method;
                verbose=verbose,
                save=save,
                timing=timing,
                plot_sample_space=plot_sample_space,
                shape_parameter=shape_parameter,
            )
        end
    end

    return signatures
end

function simulate(
    methods::Vector{Methods},
    sys::System,
    sim::Simulation;
    verbose::Bool=false,
    save::Bool=false,
    plot_sample_space::Bool=false,
    timing::Bool=false,
    shape_parameter::Union{Nothing,Float64}=nothing,
)::Vector{Model}

    # Length check
    len = length(methods)

    #Initialize the signatures array
    signatures = Vector{Model}(undef, len)

    # Iterate over the elements of all vectors simultaneously
    for (i, method) in enumerate(methods)
        signatures[i] = simulate(
            method.simulation_method,
            sys,
            sim,
            method;
            verbose=verbose,
            save=save,
            timing=timing,
            plot_sample_space=plot_sample_space,
            shape_parameter=shape_parameter,
        )
    end

    return signatures
end

function simulate(
    method::Methods,
    sys::System,
    sims::Vector{Simulation};
    verbose::Bool=false,
    save::Bool=false,
    timing::Bool=false,
    plot_sample_space::Bool=false,
    shape_parameter::Union{Nothing,Float64}=nothing,
)::Vector{Model}

    # Initialize the signatures array
    signatures = Vector{Model}(undef, length(sims))

    # Iterate over the elements of all vectors simultaneously
    for (i, sim) in enumerate(sims)
        signatures[i] = simulate(
            method.simulation_method,
            sys,
            sim,
            method;
            verbose=verbose,
            save=save,
            timing=timing,
            plot_sample_space=plot_sample_space,
            shape_parameter=shape_parameter,
        )
    end

    return signatures
end

function simulate(
    method::Methods,
    sys::System,
    sim::Simulation;
    verbose::Bool=false,
    save::Bool=false,
    timing::Bool=false,
    plot_sample_space::Bool=false,
    shape_parameter::Union{Nothing,Float64}=nothing,
)::Model
    return simulate(
        method.simulation_method,
        sys,
        sim,
        method;
        verbose=verbose,
        save=save,
        timing=timing,
        plot_sample_space=plot_sample_space,
        shape_parameter=shape_parameter,
    )
end

function simulate(
    method::MonteCarloSimulation,
    sys::System,
    sim::Simulation,
    methods::Methods;
    verbose::Bool=false,
    save::Bool=false,
    timing::Bool=false,
    plot_sample_space::Bool=false,
    shape_parameter::Union{Nothing,Float64}=nothing,
)::Model
    if timing
        start_time = time_ns()
    end

    if verbose
        printDetails(sys, sim, methods)
    end
    state_vectors, percolated_state_vectors, sim.threshold = Evaluation.generateStateVectors(
        sys
    )

    Phi = Points(
        percolated_state_vectors,
        collect(1:size(percolated_state_vectors, 2)),
        fill(Inf, size(percolated_state_vectors, 2)),
        fill(Inf, size(percolated_state_vectors, 2)),
    )

    # ==========================================================================
    if verbose
        println("Calculating Survival Signature Entires...")
    end
    Phi.solution, Phi.confidence = Evaluation.computeSurvivalSignatureEntry(
        sys, sim, Phi.coordinates; verbose=verbose
    )
    if verbose
        println("Survival Signature Calculated\n")
    end
    # ==========================================================================

    # clean-up unnessesary method definitions
    methods = Methods(method, nothing, nothing, nothing, nothing, nothing, nothing)

    Phi = expandPhi!(Phi, state_vectors)        # resize Phi to full (non-percolated) version
    Phi = reshapePhi!(Phi)                      # reshape Phi to retangular arrays
    # for the purpose of comparison
    # might make more sense to start with this size
    # but many changes would be necessary   

    signature = Model(Phi, MonteCarloModel(), sys, sim, methods, nothing, nothing)

    # ==========================================================================
    # Timing
    if timing
        elapsed_time = (time_ns() - start_time) / 1e9       # in seconds
        metrics = Metrics(Times(elapsed_time, 0.0, 0.0, 0.0, 0.0, 0.0), 0)
        signature.metrics = metrics
    end

    return signature
end
function simulate(
    method::IntervalPredictorSimulation,
    sys::System,
    sim::Simulation,
    methods::Methods;
    verbose::Bool=false,
    save::Bool=false,
    timing::Bool=false,
    plot_sample_space::Bool=false,
    shape_parameter::Union{Nothing,Float64}=nothing,
)::Model
    if verbose
        printDetails(sys, sim, methods)
    end

    if timing
        start_time = time_ns()
    end

    # =============================== PERCOLATION ==============================
    state_vectors, percolated_state_vectors, sim.threshold = Evaluation.generateStateVectors(
        sys
    )

    Phi = Points(
        percolated_state_vectors,
        collect(1:size(percolated_state_vectors, 2)),
        fill(Inf, size(percolated_state_vectors, 2)),
        fill(Inf, size(percolated_state_vectors, 2)),
    )

    # ============================= STARTING POINTS ============================
    if verbose
        println("Generating Starting Points...")
    end
    starting_points = StartingPoints.generateStartingPoints(
        methods.starting_points_method, Phi.coordinates, sys.types
    )

    starting_points.solution, starting_points.confidence = Evaluation.computeSurvivalSignatureEntry(
        sys, sim, starting_points.coordinates; verbose=verbose
    )

    if verbose
        println("Starting Points Generated.\n")
    end

    # ================================ CENTERS =================================
    centers_time = 0.0
    if timing
        centers_start_time = time_ns()
    end
    if verbose
        println("Generating Centers...")
    end
    centers = Centers.generateCenters(
        methods.centers_method, sys, sim, state_vectors, sim.threshold; verbose=verbose
    )
    if verbose
        println("\tNumber of Centers: $(size(centers, 2))")
        println("Centers Generated.\n")
    end
    if timing
        centers_time = (time_ns() - centers_start_time) / 1e9
    end

    # =================PLOT CENTERS AND STARTING POINTS=========================

    if plot_sample_space
        plt = plotSampleSpace(
            state_vectors,
            Phi;
            centers=centers,
            initial_points=nothing,
            combine_initial_additional=false,
        )
        if save
            clean_method = replace(string(methods.centers_method), r"[\[\]\(\), ]" => "_")

            # Construct the filename based on the percolation flag and cleaned centers_method
            if sys.percolation
                title = "centers_plot_$(clean_method)_percolation.pdf"
            else
                title = "centers_plot_$(clean_method).pdf"
            end

            # Define the path to save the figure
            file_path = joinpath("figures", "centers", title)

            # Save the plot
            savefig(plt, file_path)
        end
        display(plt)
    end
    # ==========================================================================

    # ============================== CONSTRAINTS ===============================

    constraints = monotonicity_constraints(centers)

    # ============================ SHAPE PARAMETER =============================
    shape_param_time = 0.0
    if timing
        shape_param_start_time = time_ns()
    end
    if verbose
        println("Compute Shape Parameters...")
    end
    if isnothing(shape_parameter)
        shape_parameter = ShapeParameter.computeShapeParameter(
            methods.shape_parameter_method,
            Phi.coordinates,
            starting_points,
            centers;
            verbose=verbose,
        )
    else
        shape_parameter = shape_parameter
    end
    if verbose
        println("\tShape Parameter: $(shape_parameter)")
        println("Shape Parameter Computed.\n")
    end
    if timing
        shape_param_time = (time_ns() - shape_param_start_time) / 1e9
    end

    # ============================ BASIS FUNCTION ==============================
    if verbose
        println("Initializing Basis Function...")
    end
    starting_basis = BasisFunction.basis(
        methods.basis_function_method, shape_parameter, starting_points.coordinates, centers
    )
    if verbose
        println("Basis Function Initialized.\n")
    end

    # ============================ INITIAL WEIGHTS =============================
    if verbose
        println("Initializing Weights...")
    end
    initial_weights = lsqr(starting_basis, starting_points.solution, constraints)
    if verbose
        println("Weights Initialized.\n")
    end

    # ========================== ADAPTIVE REFINEMENT ===========================
    adaptive_refinement_time = 0.0
    if timing
        adaptive_refinement_start_time = time_ns()
    end
    if verbose
        println("Beginning Adaptive Refinement...")
    end

    evaluated_points, weights, upper_bound, lower_bound, iterations, shape_parameter = AdaptiveRefinement.adaptiveRefinement(
        methods.adaptive_refinement_method,
        Phi,
        starting_points,
        sys,
        sim,
        methods,
        initial_weights,
        centers,
        constraints,
        shape_parameter;
        verbose=verbose,
    )

    if verbose
        println("Adaptive Refinement Completed\n")
    end

    if timing
        adaptive_refinement_time = (time_ns() - adaptive_refinement_start_time) / 1e9
    end

    if plot_sample_space
        additional_points = Matrix{Float64}(
            undef,
            size(starting_points.coordinates, 1),
            size(evaluated_points.coordinates, 2) - size(starting_points.coordinates, 2),
        )
        i = 1
        for point in eachcol(evaluated_points.coordinates)
            if !(point in eachcol(starting_points.coordinates))
                additional_points[:, i] = point
                i += 1
            end
        end

        plt = plotSampleSpace(
            state_vectors,
            Phi;
            centers=centers,
            initial_points=starting_points.coordinates,
            additional_points=additional_points,
            combine_initial_additional=true,
        )
        display(plt)
    end

    # ========================== INTERVAL PREDICTOR ============================
    ipm = IntervalPredictorModel.intervalPredictor(
        evaluated_points,
        upper_bound,
        lower_bound,
        centers,
        shape_parameter,
        weights,
        methods,
    )

    Phi = mergePoints!(Phi, evaluated_points)   # fill evaluated values into Phi
    Phi = expandPhi!(Phi, state_vectors)        # resize Phi to full (non-percolated) version
    Phi = reshapePhi!(Phi)                      # reshape Phi to retangular arrays
    #                                           # for the purpose of comparison
    #                                           # might make more sense to start with this size
    #                                           # but many changes would be necessary   

    signature = Model(Phi, ipm, sys, sim, methods, nothing, nothing) # struct

    # ====================== EVALUATE REMAINING POINTS =========================
    if verbose
        println("Evaluating Remaining Points...")
    end

    if timing
        evaluation_start_time = time_ns()
    end

    signature = Evaluation.evaluate(signature)

    if timing
        evaluation_time = (time_ns() - evaluation_start_time) / 1e9
    end

    if verbose
        println("Remaining Points Evaluated.\n")
    end

    # ==========================================================================

    # Timing
    if timing
        total_time = (time_ns() - start_time) / 1e9       # in seconds
        times = Times(
            total_time,
            centers_time,
            shape_param_time,
            adaptive_refinement_time,
            evaluation_time,
        )
        signature.metrics = Metrics(times, iterations)
    end

    return signature
end

# ==============================================================================

# ==============================================================================
# relocate this function
function mergePoints!(Phi::Points, evaluated::Points)::Points
    for (i, idx) in enumerate(evaluated.idx)
        Phi.solution[idx] = evaluated.solution[i]
        Phi.confidence[idx] = evaluated.confidence[i]
    end
    return Phi
end

function expandPhi!(Phi::Points, full_state_vector::Array)::Points
    solutions = zeros(size(full_state_vector, 2))
    confidence = zeros(size(full_state_vector, 2))

    for (i, state_vector) in enumerate(eachcol(full_state_vector))
        # find if that state_vector (vector) is in Phi.coordinates
        idx = findfirst(x -> x == state_vector, eachcol(Phi.coordinates))

        if idx !== nothing
            solutions[i] = Phi.solution[idx]
            confidence[i] = Phi.confidence[idx]
        else
            # set solutions[i] and confidence[i] to zero
            solutions[i] = 0.0
            confidence[i] = 0.0
        end
    end

    return Points(
        full_state_vector, collect(1:size(full_state_vector, 2)), solutions, confidence
    )
end

function reshapePhi!(Phi::Points)::Points
    # Create zero arrays with appropriate dimensions
    dimensions = Int.(Phi.coordinates[:, end])
    coordinates = Array{Vector{Int},length(dimensions)}(undef, dimensions...)
    index = zeros(Int, dimensions...)
    solution = zeros(Float64, dimensions...)
    confidence = zeros(Float64, dimensions...)

    for idx in CartesianIndices(coordinates)
        # Convert CartesianIndex to a Tuple and then to a Vector
        idx_vector = collect(Tuple(idx))

        # Store the vector at the current index
        coordinates[idx] = idx_vector

        # Find the corresponding index in Phi.coordinates
        id = findfirst(x -> x == idx_vector, eachcol(Phi.coordinates))

        # Ensure id is not nothing before assignment
        if id !== nothing
            index[idx] = Phi.idx[id]
            solution[idx] = Phi.solution[id]
            confidence[idx] = Phi.confidence[id]
        end
    end

    Phi = Points(coordinates, index, solution, confidence)

    return Phi
end
# ==============================================================================

# relocate this function.

default(;
    framestyle=:box,
    label=nothing,
    grid=true,
    legend=:bottomleft,
    legend_font_halign=:left,
    size=(300, 300),
    titlefontsize=8,
    guidefontsize=8,
    legendfontsize=7,
    tickfontsize=8,
    left_margin=-2 * Plots.mm,
    bottom_margin=-2 * Plots.mm,
    fontfamily="Computer Modern",
    dpi=600,
)

function plotSampleSpace(
    state_vectors::Array{Float64},
    Phi::Points;
    centers::Union{Nothing,Matrix{Float64}}=nothing,
    initial_points::Union{Nothing,Array{Float64}}=nothing,
    additional_points::Union{Nothing,Array{Float64}}=nothing,
    combine_initial_additional::Bool=false,
)
    marker_size = 1.75 * sqrt(2601) / sqrt(size(state_vectors, 2))

    # Extract coordinates
    x_state = state_vectors[1, :]
    y_state = state_vectors[2, :]
    x_phi = Phi.coordinates[1, :]
    y_phi = Phi.coordinates[2, :]

    # Determine which points are percolated and which are not
    percolated_x = []
    percolated_y = []
    non_percolated_x = []
    non_percolated_y = []

    for i in eachindex(x_state)
        if (x_state[i], y_state[i]) in zip(x_phi, y_phi)
            push!(non_percolated_x, x_state[i])
            push!(non_percolated_y, y_state[i])
        else
            push!(percolated_x, x_state[i])
            push!(percolated_y, y_state[i])
        end
    end

    legend_labels = String[]  # To keep track of the labels

    Plots.plot(
        non_percolated_x,
        non_percolated_y;
        seriestype=:scatter,
        markersize=marker_size,
        color="#cfe2f3",
        markerstrokecolor="#cfe2f3",
        label=nothing,
        xlabel=L"l_1",
        ylabel=L"l_2",
    )

    # Plot percolated points
    if !isempty(percolated_x)
        Plots.plot!(
            percolated_x,
            percolated_y;
            seriestype=:scatter,
            markersize=marker_size,
            color="#edc951",
            markerstrokecolor="#edc951",
            label="Percolated",
        )
        push!(legend_labels, "Percolated")
    end
    # Plot centers if provided
    if !isnothing(centers)
        x_centers = centers[1, :]
        y_centers = centers[2, :]
        plt = Plots.plot!(
            x_centers,
            y_centers;
            seriestype=:scatter,
            markersize=marker_size,
            color=:tomato,
            markerstrokecolor=:tomato,
            label="Center",
        )
        push!(legend_labels, "Center")
    end

    # Plot initial and additional points based on options
    if !combine_initial_additional
        label_initial = "Initial"
        color_initial = "#7DCE82"

        label_additional = "Additional"
        color_additional = :dodgerblue
    else
        label_initial = "Evaluated"
        color_initial = :dodgerblue

        label_additional = ""
        color_additional = :dodgerblue
    end

    # Plot initial points
    if !isnothing(initial_points)
        Plots.plot!(
            initial_points[1, :],
            initial_points[2, :];
            seriestype=:scatter,
            markersize=marker_size,
            color=color_initial,
            markerstrokecolor=color_initial,
            label=label_initial,
        )
        push!(legend_labels, label_initial)
    end

    # Plot additional points
    if !isnothing(additional_points)
        Plots.plot!(
            additional_points[1, :],
            additional_points[2, :];
            seriestype=:scatter,
            markersize=marker_size,
            color=color_additional,
            markerstrokecolor=color_additional,
            label=label_additional,
        )
        push!(legend_labels, label_additional)
    end

    # Determine legend visibility based on the number of unique labels
    if length(legend_labels) < 2
        Plots.plot!(; legend=false)
    else
        Plots.plot!(; legend=:bottomleft)
    end
    # Display the plot
    plt = Plots.plot!()

    return plt
end
end