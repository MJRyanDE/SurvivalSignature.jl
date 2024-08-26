module Simulate

__precompile__()

# this module to store the functions relating to simulating a given grid or model.
# ==============================================================================

using Plots

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
        sys, sim, Phi.coordinates
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
        metrics = Metrics(Times(elapsed_time, 0.0, 0.0, 0.0, 0.0), 0)
        signature.metrics = metrics

        if verbose
            println("\tTime Elapsed: $(elapsed_time)s")
            println(" ")
            println("Finished Successfully.\n")
        end
    end

    return signature
end
function simulate(
    method::IntervalPredictorSimulation,
    sys::System,
    sim::Simulation,
    methods::Methods;
    verbose::Bool=false,
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
        methods.centers_method, state_vectors, sim.threshold; verbose=verbose
    )
    if verbose
        println("Centers Generated.\n")
    end
    if timing
        centers_time = (time_ns() - centers_start_time) / 1e9
    end

    # =================PLOT CENTERS AND STARTING POINTS=========================

    if plot_sample_space
        plotSampleSpace(
            Phi;
            centers=centers,
            methods=methods,
            initial_points=starting_points.coordinates,
            show_points=true,
            combine_initial_additional=false,
        )
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
            methods.shape_parameter_method, Phi.coordinates, starting_points, centers
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

    evaluated_points, weights, upper_bound, lower_bound, iterations = AdaptiveRefinement.adaptiveRefinement(
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

        plotSampleSpace(
            Phi;
            centers=centers,
            methods=methods,
            initial_points=starting_points.coordinates,
            additional_points=additional_points,
            show_points=true,
            combine_initial_additional=false,
        )
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
    signature = Evaluation.evaluate(signature)
    if verbose
        println("Remaining Points Evaluated.\n")
    end
    # ==========================================================================

    # Timing
    if timing
        total_time = (time_ns() - start_time) / 1e9       # in seconds
        times = Times(total_time, centers_time, shape_param_time, adaptive_refinement_time)
        signature.metrics = Metrics(times, iterations)

        if verbose
            println("Total Time Elapsed: $(total_time)s")
            println("Centers Generation Time: $(centers_time)s")
            println("Shape Parameter Time: $(shape_param_time)s")
            println("Adaptive Refinement Time: $(adaptive_refinement_time)s")
            println(" ")
            println("Finished Successfully.\n")
        end
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
function plotSampleSpace(
    Phi::Points;
    centers::Union{Nothing,Matrix{Float64}}=nothing,
    methods::Union{Nothing,Methods}=nothing,
    initial_points::Union{Nothing,Array{Float64}}=nothing,
    additional_points::Union{Nothing,Array{Float64}}=nothing,
    show_points::Bool=true,
    combine_initial_additional::Bool=false,
)

    # currently only works for 2D - two types.

    x_phi = Phi.coordinates[1, :]  # x-coordinates for Phi
    y_phi = Phi.coordinates[2, :]  # y-coordinates for Phi

    if !isnothing(centers)
        x_centers = centers[1, :]  # x-coordinates for centers
        y_centers = centers[2, :]  # y-coordinates for centers
    end

    if !isnothing(initial_points)
        x_initial = initial_points[1, :]  # x-coordinates for initial points
        y_initial = initial_points[2, :]  # y-coordinates for initial points
    end

    if !isnothing(additional_points)
        x_evaluated = additional_points[1, :]  # x-coordinates for additional points
        y_evaluated = additional_points[2, :]  # y-coordinates for additional points
    end

    # Plot Phi coordinates without a label (no legend entry)
    if show_points
        Plots.plot(
            x_phi,
            y_phi;
            seriestype=:scatter,
            marker=:+,
            markersize=0.5,
            color="black",
            label=false,
        )
    end

    # Plot centers with the label "Center"
    if !isnothing(centers)
        Plots.plot!(
            x_centers,
            y_centers;
            seriestype=:scatter,
            markersize=3,
            color="red",
            label="Center",
        )
    end

    if combine_initial_additional
        label_initial = "Evaluated"
        color_initial = "blue"

        label_additional = "Evaluated"
        color_additional = "blue"
    else
        label_initial = "Initial"
        color_initial = "blue"

        label_additional = "Additional"
        color_additional = "yellow"
    end

    # Plot initial points with the label "Initial"
    if !isnothing(initial_points)
        Plots.plot!(
            x_initial,
            y_initial;
            seriestype=:scatter,
            markersize=3,
            color=color_initial,
            label=label_initial,
        )
    end

    # Plot additional points with the label "Evaluated"
    if !isnothing(additional_points)
        Plots.plot!(
            x_evaluated,
            y_evaluated;
            seriestype=:scatter,
            markersize=3,
            color=color_additional,
            label=label_additional,
        )
    end

    # Add title and display
    if !isnothing(methods)
        if !isnothing(methods.additional_points)
            Plots.plot!(;
                title="Sample Space - $(nameof(typeof(method.centers_method))) - $(nameof(typeof(method.adaptive_refinement_method)))",
            )
        else
            Plots.plot!(; title="Sample Space - $(nameof(typeof(method.centers_method)))")
        end
    else
        Plots.plot!(; title="Sample Space")
    end

    return display(Plots.plot!())
end
end