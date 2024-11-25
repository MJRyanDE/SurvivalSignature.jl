module AdaptiveRefinement

__precompile__()

# ==============================================================================

using Base.Threads
using Statistics
using Distributions
using LaTeXStrings
using Plots
using Surrogates
using CairoMakie

using Optim
using BlackBoxOptim
using JuMP
using Ipopt
using Evolutionary

using ForwardDiff
using NearestNeighbors
using InvertedIndices
using ProgressMeter
using LinearAlgebra

# ==============================================================================

using ..SurvivalSignatureUtils
using ..Structures: BasisFunctionMethod, Gaussian
using ..Structures: Points, System, Simulation, Methods
using ..Structures: None, TEAD, MEPE, MIPT, OIPT, SFCVT, MASA, EI, EIGF
using ..Structures: EnumerationX, BlackBoxX, OptimX, SimulatedAnnealingX, EvolutionX
using ..Structures: NORM, NRMSE
using ..Evaluation
using ..BasisFunction
using ..ShapeParameter
using ..Error

using ..Visualization

# access to monotonicity_constraints and lsqr 
include("../src/rbf/radialbasisfunctions.jl")

# ==============================================================================

export adaptiveRefinement

# move this later
function survialSignatureDistanceMatrix(state_vectors::Matrix{Float64})
    n = size(state_vectors, 2)  # Number of state vectors (columns)
    DM = zeros(Float64, n, n)   # Initialize the distance matrix

    # Iterate over each pair of columns using eachcol()
    columns = collect(eachcol(state_vectors))

    for i in 1:n
        for j in (i + 1):n
            # Compute the Euclidean distance between column i and column j
            DM[i, j] = norm(columns[i] - columns[j])
            DM[j, i] = DM[i, j]  # Symmetric matrix
        end
    end

    return DM
end

# ==============================================================================

function adaptiveRefinement(
    method::None,
    total_points::Points,
    evaluated_points::Points,
    sys::System,
    sim::Simulation,
    methods::Methods,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64;
    verbose::Bool=false,
)
    iterations = 0

    upper_bound =
        min.(
            evaluated_points.solution .+
            (evaluated_points.solution .* evaluated_points.confidence),
            1.0,
        )
    lower_bound =
        max.(
            evaluated_points.solution .-
            (evaluated_points.solution .* evaluated_points.confidence),
            0.0,
        )

    replace!(upper_bound, NaN => 1 / (sim.samples + 1))
    replace!(lower_bound, NaN => 0.0)

    return evaluated_points, weights, upper_bound, lower_bound, iterations, shape_parameter
end
# ==============================================================================
#  Taylor Expansion based Adaptive Design
# ==============================================================================
function adaptiveRefinement(
    method::TEAD,
    total_points::Points,
    evaluated_points::Points,
    sys::System,
    sim::Simulation,
    methods::Methods,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64;
    verbose::Bool=false,
)
    total_basis = BasisFunction.basis(
        methods.basis_function_method, shape_parameter, total_points.coordinates, centers
    )

    x = deepcopy(evaluated_points) # evaluated points
    l_max = maximumLength(total_points)
    candidates, cand_idx = remainingCandidates(total_points, x)

    prog = if verbose
        ProgressMeter.ProgressThresh(sim.weight_change_tolerance, "\tAdaptive Refinement")
    else
        nothing
    end

    stop = 0
    f_max = maximum(evaluated_points.solution)
    iterations = 0
    while stop < 2 && length(candidates) > 0 && iterations < method.max_iterations
        iterations += 1

        function s(a::Points)
            return (BasisFunction.basis(methods.basis_function_method, shape_parameter, a.coordinates, centers) * weights)[1]
        end

        # exploration score
        idx, D = explorationScore(x, candidates)

        # exploitation score

        R = exploitationScore(total_basis, x, s, weights, candidates, cand_idx, idx)

        # weight function
        w = weightFunction(idx, l_max)

        # hybrid score function
        J = hybridScore(D, R, w)

        optimal_candidates, optimal_idx = optimalPoint(total_points, J, candidates)

        true_values, coefficient_variation = Evaluation.computeSurvivalSignatureEntry(
            sys, sim, optimal_candidates; verbose=verbose
        )

        # update the computed state_vectors
        x.coordinates = hcat(x.coordinates, optimal_candidates)

        x.solution = vcat(x.solution, true_values)
        x.confidence = vcat(x.confidence, coefficient_variation)
        x.idx = vcat(x.idx, optimal_idx)

        candidates, cand_idx = remainingCandidates(total_points, x)

        old_weights = weights

        if method.adaptive_refinement_shape_parameter
            #weight_change_method = NRMSE()
            shape_parameter =
                shape_parameter = ShapeParameter.computeShapeParameter(
                    methods.shape_parameter_method,
                    total_points.coordinates,
                    x,
                    centers;
                    verbose=verbose,
                )
            #else
            #weight_change_method = methods.weight_change_method
        end

        # recompute the basis Function
        basis = BasisFunction.basis(
            methods.basis_function_method, shape_parameter, x.coordinates, centers
        )

        # update weights
        weights = lsqr(basis, x.solution, constraints)

        weight_change_method = methods.weight_change_method
        if isa(weight_change_method, NORM)
            error_value = calculateError(weight_change_method, weights, old_weights)

        elseif isa(weight_change_method, NRMSE)
            predicted_values::Vector{Float64} = [
                (BasisFunction.basis(methods.basis_function_method, shape_parameter, y, centers) * weights)[1]
                for y::Vector in eachcol(optimal_candidates)
            ]

            error_value, f_max = calculateError(
                weight_change_method, [true_values], predicted_values, f_max
            )

        else
            error("Unsupported error calculation method")
        end

        if error_value < sim.weight_change_tolerance
            stop += 1
        else
            stop = 0
        end

        if prog !== nothing
            stop != 1 && ProgressMeter.update!(prog, error_value)
        end
    end

    upper_bound = min.(x.solution .+ (x.solution .* x.confidence), 1.0)
    lower_bound = max.(x.solution .- (x.solution .* x.confidence), 0.0)

    replace!(upper_bound, NaN => 1 / (sim.samples + 1))
    replace!(lower_bound, NaN => 0.0)

    if verbose
        println("\tRequired Iterations: ", iterations)
    end

    return x, weights, upper_bound, lower_bound, iterations, shape_parameter
end

function explorationScore(x::Points, candidates::Matrix)
    tree = NearestNeighbors.KDTree(x.coordinates)
    idx, dist = NearestNeighbors.nn(tree, candidates)
    return idx, dist
end

function exploitationScore(
    total_basis::Matrix,
    X::Points,
    func::Function,
    weights::Array,
    candidates::Array,
    cand_idx::Array,
    nearest_neighbor_idx::Array,
)
    # Combine coordinates and solutions for local function s(a)
    combined = [(X.coordinates[:, i], X.solution[i]) for i in 1:size(X.coordinates, 2)]

    # Initialize gradient as a zeros array
    ∇s = [zeros(size(X.coordinates, 1)) for _ in 1:size(X.coordinates, 2)]
    ∇s = map(ForwardDiff.DiffResults.GradientResult, ∇s)  # Convert to gradient object

    # Compute gradients for each set of coordinates in `combined` with respect to `func`.
    # For each pair of coordinates (`coords`) and corresponding solution (`sol`),
    # a new `Points` instance is created with `coords` as `y` and `sol` as `solution`.
    # The gradient is calculated and stored in the preallocated gradient result `r`.
    ∇s = map(
        ((coords, sol), r) ->
            ForwardDiff.gradient!(r, y -> func(Points(y, nothing, sol, nothing)), coords),
        combined,
        ∇s,
    )

    gradient_values = ForwardDiff.DiffResults.value.(∇s)
    ∇s = ForwardDiff.DiffResults.gradient.(∇s)

    # Exploration score function
    idx = nearest_neighbor_idx
    t = @views gradient_values[idx] .+ map(
        (∇, a) -> dot(∇, a), ∇s[idx], eachcol(candidates .- X.coordinates[:, idx])
    )

    return @views abs.(total_basis[cand_idx, :] * weights .- t)
end

function weightFunction(nearest_neighbor_distance::Array, l_max::Number)
    return (1 .- nearest_neighbor_distance ./ l_max)
end

function maximumLength(Ω::Points)
    lb = minimum(Ω.coordinates; dims=2)
    ub = maximum(Ω.coordinates; dims=2)

    return sqrt(sum((ub .- lb) .^ 2))
end

function hybridScore(
    exploration_score::Array, exploitation_score::Array, weight_function::Array
)

    # normalization
    exploration_score = exploration_score ./ maximum(exploration_score)
    exploitation_score = exploitation_score ./ maximum(exploitation_score)

    return exploration_score + weight_function .* exploitation_score
end

function optimalPoint(Ω::Points, scores::Array, candidates::Array)
    _, idx = findmax(scores)
    optimal_candidates = candidates[:, idx]  # optimal candidate (state_vector)

    # convert from the candidates index to the equivalent Ω index
    optimal_candidates_idx = findfirst(
        x -> all(x .== optimal_candidates), eachcol(Ω.coordinates)
    )

    return optimal_candidates, optimal_candidates_idx
end

function remainingCandidates(Ω::Points, x::Points)
    # remaining values

    cand_idx = InvertedIndices.Not(x.idx)
    candidates = Ω.coordinates[:, cand_idx]

    # convert from InvertedIndices to Array
    all_indices = 1:size(Ω.coordinates, 2)
    cand_idx = filter(i -> !(i in x.idx), all_indices)

    return candidates, cand_idx
end

function remainingCandidates(candidates::Array, remove_points::Array, cand_idx::Array)
    candidates = candidates[.!in.(1:size(candidates, 1), Ref(remove_points)), :]
    cand_idx = cand_idx[.!in.(cand_idx, Ref(remove_points))]
    return candidates, cand_idx
end

# ==============================================================================
#  Maximum Expected Prediction Error
# ==============================================================================
function adaptiveRefinement(
    method::MEPE,
    total_points::Points,
    evaluated_points::Points,
    sys::System,
    sim::Simulation,
    methods::Methods,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64;
    verbose::Bool=false,
)
    x = deepcopy(evaluated_points)

    x, iterations = mepe(
        method.max_iterations,
        sys,
        sim,
        x,
        total_points,
        weights,
        centers,
        constraints,
        shape_parameter,
        methods;
        verbose=verbose,
    )

    upper_bound = min.(x.solution .+ (x.solution .* x.confidence), 1.0)
    lower_bound = max.(x.solution .- (x.solution .* x.confidence), 0.0)

    replace!(upper_bound, NaN => 1 / (sim.samples + 1))
    replace!(lower_bound, NaN => 0.0)

    if verbose
        println("\tRequired Iterations: ", iterations)
    end

    return x, weights, upper_bound, lower_bound, iterations, shape_parameter
end

function mepe(
    max_iterations::Int,
    sys::System,
    sim::Simulation,
    evaluated_points::Points,
    total_points::Points,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64,
    methods::Methods;
    verbose::Bool=false,
)
    basis = BasisFunction.basis(
        methods.basis_function_method,
        shape_parameter,
        evaluated_points.coordinates,
        centers,
    )

    distance_matrix = survialSignatureDistanceMatrix(evaluated_points.coordinates)

    R = rbfCorrelationMatrix(distance_matrix, shape_parameter)
    invR = inv(R)

    f_max = maximum(evaluated_points.solution)
    cross_validation_error_array = fastCrossValidationErrorArray(
        basis, invR, evaluated_points.solution
    )
    candidates, cand_idx = remainingCandidates(total_points, evaluated_points)

    stop, iterations, alpha = 0, 0, 0.5

    prog = if verbose
        ProgressMeter.ProgressThresh(sim.weight_change_tolerance, "\tAdaptive Refinement")
    else
        nothing
    end

    while stop < 1 && iterations < max_iterations
        iterations += 1
        new_point = chooseNewPointMEPE(
            candidates,
            evaluated_points.coordinates,
            invR,
            shape_parameter,
            alpha,
            cross_validation_error_array,
        )

        if isnothing(new_point)
            iterations -= 1
            continue
        end

        prediction, confidence = Evaluation.computeSurvivalSignatureEntry(
            sys, sim, new_point; verbose=verbose
        )

        matching_index = findfirst(x -> all(x .== new_point), eachcol(candidates))#

        evaluated_points.coordinates = hcat(evaluated_points.coordinates, new_point)
        evaluated_points.solution = vcat(evaluated_points.solution, prediction)
        evaluated_points.confidence = vcat(evaluated_points.confidence, confidence)
        evaluated_points.idx = vcat(evaluated_points.idx, cand_idx[matching_index])

        # update cross_validation_array and balance_factor
        candidates, cand_idx = remainingCandidates(total_points, evaluated_points)

        basis = BasisFunction.basis(
            methods.basis_function_method,
            shape_parameter,
            evaluated_points.coordinates,
            centers,
        )

        distance_matrix = survialSignatureDistanceMatrix(evaluated_points.coordinates)
        R = rbfCorrelationMatrix(distance_matrix, shape_parameter)
        invR = inv(R)
        cross_validation_error_array = fastCrossValidationErrorArray(
            basis, invR, evaluated_points.solution
        )

        prediction, _ = Evaluation.evaluateSurrogate(
            evaluated_points.coordinates[:, end - 1],
            weights,
            shape_parameter,
            centers,
            methods,
        )
        true_error = abs(evaluated_points.solution[end - 1] - prediction)

        alpha = balanceFactor(true_error, cross_validation_error_array[end - 1])

        # update model and check stopping criteria:
        old_weights = weights
        basis = BasisFunction.basis(
            methods.basis_function_method,
            shape_parameter,
            evaluated_points.coordinates,
            centers,
        )
        weights = lsqr(basis, evaluated_points.solution, constraints)

        if isa(methods.weight_change_method, NORM)
            error_value = calculateError(methods.weight_change_method, weights, old_weights)

        elseif isa(methods.weight_change_method, NRMSE)
            predicted_values::Vector{Float64} = [
                (BasisFunction.basis(methods.basis_function_method, shape_parameter, y, centers) * weights)[1]
                for y::Vector in eachcol(new_point)
            ]
            error_value, f_max = calculateError(
                methods.weight_change_method, prediction, predicted_values, f_max
            )

        else
            error("Unsupported error calculation method")
        end

        if error_value < sim.weight_change_tolerance
            stop += 1
        else
            stop = 0
        end

        if prog !== nothing
            ProgressMeter.update!(prog, error_value)
        end
    end
    return evaluated_points, iterations
end

function balanceFactor(true_error::Float64, cross_validation_error::Float64)
    alpha = 0.99 * min(1, 0.5 * (true_error^2 / cross_validation_error^2))
    return alpha
end

function chooseNewPointMEPE(
    candidates::Matrix{Float64},
    existing_points::Matrix{Float64},
    invR::Matrix,
    shape_parameter::Float64,
    alpha::Float64,
    cross_validation_error_array::Vector{Float64},
)::Vector{Float64}

    # uses Enumeration to find the optimal point - potentially slow.
    optimal_point = zeros(size(candidates, 1))
    optimal_score = -Inf
    lock_obj = ReentrantLock()

    candidate_columns = collect(eachcol(candidates))

    @threads for candidate::Vector in candidate_columns
        epe = computeEPE(
            candidate,
            existing_points,
            invR,
            shape_parameter,
            alpha,
            cross_validation_error_array,
        )
        lock(lock_obj) do
            if epe > optimal_score
                optimal_score = epe
                optimal_point = candidate
            end
        end
    end

    return optimal_point
end

function computeGlobalExploration(
    candidate::Vector,
    existing_points::Matrix,
    invR::Matrix,
    shape_parameter::Float64,
    sigma2::Float64,
)::Float64

    # prediction variance 
    r_x = reshape(
        [
            exp(-shape_parameter * norm(candidate .- x)) for
            x::Vector in eachcol(existing_points)
        ],
        :,
        1,
    )

    global_exploration_term = (sigma2 * (1 .- r_x' * invR * r_x))[1]   # convert from Matrix to Float64

    return global_exploration_term #s^2(x)
end

function computeEPE(
    candidate::Vector,
    existing_points::Matrix,
    invR::Matrix,
    shape_parameter::Float64,
    alpha::Float64,
    cross_validation_error_array::Vector,
)
    distances = [norm(candidate .- existing_points[i]) for i in 1:size(existing_points, 2)]
    nearest_neighbor_idx = argmin(distances)
    e2_cv_x = cross_validation_error_array[nearest_neighbor_idx]

    s2_x = computeGlobalExploration(
        candidate, existing_points, invR, shape_parameter, e2_cv_x
    )

    epe = alpha * e2_cv_x + (1 - alpha) * s2_x

    return epe
end

function rbfCorrelationMatrix(
    distance_matrix::Matrix{Float64},
    shape_parameter::Float64;
    method::BasisFunctionMethod=Gaussian(),
)
    return BasisFunction.basis(method, shape_parameter, distance_matrix)
end

function fastCrossValidationErrorArray(F::Matrix, invR::Matrix, solutions::Vector{Float64})
    # Step 1: Calculate the inverse of the normal matrix with conditional regularization
    gram_matrix = F' * F
    condition_num = cond(gram_matrix)

    # Set a threshold for the condition number to determine if regularization is needed
    threshold = 1e10  # You can adjust this threshold as needed

    if condition_num > threshold
        λ = 1e-6  # Small regularization parameter
        identity_matrix = I(size(F, 2))  # Identity matrix with appropriate size
        inv_normal_matrix = (gram_matrix + λ * identity_matrix) \ identity_matrix
    else
        inv_normal_matrix = inv(gram_matrix)
    end

    # Step 2: Calculate beta_hat
    beta_hat = inv_normal_matrix * F' * solutions

    # Step 3: Calculate d = y_D - F * beta_hat
    d = solutions - F * beta_hat

    # Step 4: Calculate H = F * inv(F' * F) * F' in a stable manner
    H = F * inv_normal_matrix * F'

    # Step 5: Initialize an array for CV errors
    cross_validation_error_array = zeros(length(solutions))

    # Step 6: Calculate cross-validation error for each point
    for i in 1:length(solutions)
        # Extract relevant components
        R_inv_i = invR[i, :]  # i-th row of R_inv
        H_ii = H[i, i]        # diagonal element of H
        H_col_i = H[:, i]     # i-th column of H
        d_i = d[i]            # i-th element of d

        # Calculate the CV error using the formula
        numerator = dot(R_inv_i, d + H_col_i * d_i / (1 - H_ii))
        denominator = invR[i, i]

        cross_validation_error_array[i] = (numerator / denominator)^2
    end

    return cross_validation_error_array
end

# ==============================================================================
#  Expected Improvement 
# ==============================================================================
function adaptiveRefinement(
    method::EI,
    total_points::Points,
    evaluated_points::Points,
    sys::System,
    sim::Simulation,
    methods::Methods,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64;
    verbose::Bool=false,
)
    x = deepcopy(evaluated_points)

    candidates, cand_idx = remainingCandidates(total_points, x)

    prog = if verbose
        ProgressMeter.ProgressThresh(sim.weight_change_tolerance, "\tAdaptive Refinement")
    else
        nothing
    end

    f_max = 0.0
    iterations = 0
    stop = 0
    while stop < 5 && length(candidates) > 0 && iterations < method.max_iterations
        iterations += 1
        y_min = minimum(x.solution)

        optimal_point = ei(candidates, y_min, weights, shape_parameter, centers, methods)

        true_value, coefficient_variation = Evaluation.computeSurvivalSignatureEntry(
            sys, sim, optimal_point; verbose=verbose
        )

        optimal_idx = findfirst(
            x -> all(x .== optimal_point), eachcol(total_points.coordinates)
        )

        if isnothing(optimal_idx)
            # the optimization is capable of finding the same point multiple times
            # this is a workaround to ensure that the same point is not added multiple times
            continue
            # in this case it is important to note that the iterations will not equal 
            # the number of additional points.
        end

        # update the computed state_vectors
        x.coordinates = hcat(x.coordinates, optimal_point)

        x.solution = vcat(x.solution, true_value)
        x.confidence = vcat(x.confidence, coefficient_variation)
        x.idx = vcat(x.idx, optimal_idx)

        candidates, cand_idx = remainingCandidates(total_points, x)

        old_weights = weights

        # recompute the basis Function
        basis = BasisFunction.basis(
            methods.basis_function_method, shape_parameter, x.coordinates, centers
        )

        # update weights
        weights = lsqr(basis, x.solution, constraints)

        if isa(methods.weight_change_method, NORM)
            error_value = calculateError(methods.weight_change_method, weights, old_weights)

        elseif isa(methods.weight_change_method, NRMSE)
            predicted_values::Vector{Float64} = [
                (BasisFunction.basis(methods.basis_function_method, shape_parameter, y, centers) * weights)[1]
                for y::Vector in eachcol(optimal_candidates)
            ]
            error_value, f_max = calculateError(
                methods.weight_change_method, true_values, predicted_values, f_max
            )

        else
            error("Unsupported error calculation method")
        end

        if error_value < sim.weight_change_tolerance
            stop += 1
        else
            stop = 0
        end

        if prog !== nothing
            # this stops the progress meter from ending early, however the 
            # number of iterations will not be equal to the number of iterations
            # because its internal counter is only updated when the conditions are met.
            # thus when stop = 1, 2, 3, and 4, the progress meter will not update
            (stop == 0 || stop == 5) && ProgressMeter.update!(prog, error_value)
            # the issue is that when the threshold is reached the progress meter
            # will end. however, we need the threshold to me reached multiple times.
        end
    end

    upper_bound = min.(x.solution .+ (x.solution .* x.confidence), 1.0)
    lower_bound = max.(x.solution .- (x.solution .* x.confidence), 0.0)

    replace!(upper_bound, NaN => 1 / (sim.samples + 1))
    replace!(lower_bound, NaN => 0.0)

    if verbose
        println("\tRequired Iterations: ", iterations)
    end

    return x, weights, upper_bound, lower_bound, iterations, shape_parameter
end

function ei(candidates, y_min, weights, shape_parameter, centers, methods)
    best_score = -Inf
    best_candidate = nothing
    lock_obj = ReentrantLock()

    candidate_columns = collect(eachcol(candidates))

    @threads for candidate::Vector in candidate_columns
        ei = expectedImprovement(
            candidate, y_min, weights, shape_parameter, centers, methods
        )

        lock(lock_obj) do
            if ei > best_score
                best_score = ei
                best_candidate = candidate
            end
        end
    end

    return best_candidate
end

function expectedImprovement(candidate, y_min, weights, shape_parameter, centers, methods)
    prediction, rmse = Evaluation.evaluateSurrogate(
        candidate, weights, shape_parameter, centers, methods
    )

    z = (y_min - prediction) / rmse

    cdf_value = Distributions.cdf(Normal(0, 1), z)
    pdf_value = Distributions.pdf(Normal(0, 1), z)

    expected_improvement = (y_min - prediction) * cdf_value + (rmse * pdf_value)

    return expected_improvement
end

# ==============================================================================
#  Expected Improvement for Global Fit
# ==============================================================================
function adaptiveRefinement(
    method::EIGF,
    total_points::Points,
    evaluated_points::Points,
    sys::System,
    sim::Simulation,
    methods::Methods,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64;
    verbose::Bool=false,
)
    x = deepcopy(evaluated_points)

    candidates, cand_idx = remainingCandidates(total_points, x)

    prog = if verbose
        ProgressMeter.ProgressThresh(sim.weight_change_tolerance, "\tAdaptive Refinement")
    else
        nothing
    end

    f_max = 0.0
    iterations = 0
    stop = 0
    while stop < 1 && length(candidates) > 0 && iterations < method.max_iterations
        iterations += 1

        optimal_point = eigf(candidates, x, weights, shape_parameter, centers, methods)

        true_value, coefficient_variation = Evaluation.computeSurvivalSignatureEntry(
            sys, sim, optimal_point; verbose=verbose
        )

        optimal_idx = findfirst(
            x -> all(x .== optimal_point), eachcol(total_points.coordinates)
        )

        if isnothing(optimal_idx)
            # the optimization is capable of finding the same point multiple times
            # this is a workaround to ensure that the same point is not added multiple times
            continue
            # in this case it is important to note that the iterations will not equal 
            # the number of additional points.
        end

        # update the computed state_vectors
        x.coordinates = hcat(x.coordinates, optimal_point)

        x.solution = vcat(x.solution, true_value)
        x.confidence = vcat(x.confidence, coefficient_variation)
        x.idx = vcat(x.idx, optimal_idx)

        candidates, cand_idx = remainingCandidates(total_points, x)

        old_weights = weights

        # recompute the basis Function
        basis = BasisFunction.basis(
            methods.basis_function_method, shape_parameter, x.coordinates, centers
        )

        # update weights
        weights = lsqr(basis, x.solution, constraints)

        if isa(methods.weight_change_method, NORM)
            error_value = calculateError(methods.weight_change_method, weights, old_weights)

        elseif isa(methods.weight_change_method, NRMSE)
            predicted_values::Vector{Float64} = [
                (BasisFunction.basis(methods.basis_function_method, shape_parameter, y, centers) * weights)[1]
                for y::Vector in eachcol(optimal_candidates)
            ]
            error_value, f_max = calculateError(
                methods.weight_change_method, true_values, predicted_values, f_max
            )

        else
            error("Unsupported error calculation method")
        end

        if error_value < sim.weight_change_tolerance
            stop += 1
        else
            stop = 0
        end

        if prog !== nothing
            # this stops the progress meter from ending early, however the 
            # number of iterations will not be equal to the number of iterations
            # because its internal counter is only updated when the conditions are met.
            # thus when stop = 1, 2, 3, and 4, the progress meter will not update
            ProgressMeter.update!(prog, error_value)
            # the issue is that when the threshold is reached the progress meter
            # will end. however, we need the threshold to me reached multiple times.
        end
    end

    upper_bound = min.(x.solution .+ (x.solution .* x.confidence), 1.0)
    lower_bound = max.(x.solution .- (x.solution .* x.confidence), 0.0)

    replace!(upper_bound, NaN => 1 / (sim.samples + 1))
    replace!(lower_bound, NaN => 0.0)

    if verbose
        println("\tRequired Iterations: ", iterations)
    end

    return x, weights, upper_bound, lower_bound, iterations, shape_parameter
end

function eigf(candidates, y_min, weights, shape_parameter, centers, methods)
    best_score = -Inf
    best_candidate = nothing
    #lock_obj = ReentrantLock()

    candidate_columns = collect(eachcol(candidates))

    for candidate::Vector in candidate_columns
        eigf = expectedImprovementGlobalFit(
            candidate, y_min, weights, shape_parameter, centers, methods
        )

        #lock(lock_obj) do
        if eigf > best_score
            best_score = eigf
            best_candidate = candidate
        end
    end

    return best_candidate
end

function expectedImprovementGlobalFit(
    candidate, evaluated_points, weights, shape_parameter, centers, methods
)
    # evaluate surrogate returns the rmse, the eigf wants the variance.
    # for the time being the rmse will be used instead.
    prediction, rmse = Evaluation.evaluateSurrogate(
        candidate, weights, shape_parameter, centers, methods
    )

    nearest_neighbor = argmin([
        norm(candidate - x) for x::Vector in eachcol(evaluated_points.coordinates)
    ])

    nearest_neighbor_value = evaluated_points.solution[nearest_neighbor]

    expected_improvement_global_fit = (prediction - nearest_neighbor_value)^2 + rmse

    return expected_improvement_global_fit
end

# ==============================================================================
#  Optimizer Intersite Projective Threshold
# ==============================================================================
function adaptiveRefinement(
    method::OIPT,
    total_points::Points,
    evaluated_points::Points,
    sys::System,
    sim::Simulation,
    methods::Methods,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64;
    verbose::Bool=false,
)
    x = deepcopy(evaluated_points)

    # chose optimal candidates based on the MIPT methodology
    # finds the X best candidates, based on user input. 
    optimal_candidates, optimal_indices, num_additional_points = oipt(
        method, total_points, x; verbose=verbose
    )

    # compute the 'true values' and coefficient of variation
    true_values, coefficient_variation = Evaluation.computeSurvivalSignatureEntry(
        sys, sim, optimal_candidates; verbose=verbose
    )

    # update the computed state_vectors
    x.coordinates = hcat(x.coordinates, optimal_candidates)

    x.solution = vcat(x.solution, true_values)
    x.confidence = vcat(x.confidence, coefficient_variation)
    x.idx = vcat(x.idx, optimal_indices)

    # recompute the basis Function
    basis = BasisFunction.basis(
        methods.basis_function_method, shape_parameter, x.coordinates, centers
    )

    # update weights
    weights = lsqr(basis, x.solution, constraints)

    upper_bound = min.(x.solution .+ (x.solution .* x.confidence), 1.0)
    lower_bound = max.(x.solution .- (x.solution .* x.confidence), 0.0)

    replace!(upper_bound, NaN => 1 / (sim.samples + 1))
    replace!(lower_bound, NaN => 0.0)

    if verbose
        println("\tRequired Iterations: ", num_additional_points)
    end

    return x, weights, upper_bound, lower_bound, num_additional_points, shape_parameter
end

function oipt(method::OIPT, total_points::Points, evaluated_points::Points; verbose=verbose)
    n = size(evaluated_points.coordinates, 2)

    candidates, candidates_idx = remainingCandidates(total_points, evaluated_points)

    num_initial_points = min(method.num_initial_points, size(candidates, 2)) # incase the num_initial_points is 
    #                                                                        # greater than the number of candidates

    max_bounds = vec(maximum(candidates; dims=2))

    # limit candidates 
    selected_candidates = candidates[:, rand(1:size(candidates, 2), num_initial_points)]

    intersite_scores = [
        intersiteProjection(candidate, evaluated_points.coordinates; alpha=method.alpha) for
        candidate::Vector in eachcol(selected_candidates)
    ]

    num_additional_points = min(method.num_additional_points, num_initial_points)  # incase the num_additional_points 
    #                                                                              # is greater than #                                                                              # the number of candidates

    # Sort scores in descending order and get the best scores
    sorted_indices = sortperm(intersite_scores; rev=true)  # Get indices of sorted scores
    best_indices = sorted_indices[1:num_additional_points]  # Get indices of the top `num_best_points`

    best_points = selected_candidates[:, best_indices]  # Get the top candidates
    best_scores = intersite_scores[best_indices]  # Get the top scores

    optimal_points = Matrix{Float64}(undef, size(candidates, 1), num_additional_points)
    optimal_scores = Vector{Float64}(undef, num_additional_points)
    optimal_indices = Vector{Int}(undef, num_additional_points)

    # might not work with @threads
    prog = if verbose
        Progress(size(best_points, 2); desc="\tAdaptive Refinement")
    else
        nothing
    end

    best_points_scores = collect(enumerate(zip(eachcol(best_points), best_scores)))

    @threads for (i, (best_point::Vector, best_score)) in best_points_scores
        enumerate(zip(eachcol(best_points), best_scores))
        d_max = method.beta * best_score / 2
        # necessary because the state_vectors must be int values because you cant have 
        # a fraction of a working component. 
        d_max = max(round(Int, d_max), 1)

        lower_bounds = max.(repeat([0.0], length(best_point)), best_point .- d_max)
        upper_bounds = min.(max_bounds, best_point .+ d_max)

        # create a range for each dimension
        ranges = [lower_bounds[i]:upper_bounds[i] for i in eachindex(lower_bounds)]

        # optimize pnew towards -inf Norm of P new on [pnew - dmax, pnew + dmax]
        best_score = -Inf
        optimal_score = -Inf
        optimal_point = zeros(length(best_point))

        for point in Iterators.product(ranges...)
            point = collect(point) # convert from tuple to vector

            if point in eachcol(evaluated_points.coordinates)
                continue
            end

            score = minimum([
                LinearAlgebra.norm(point - evaluated_points.coordinates[:, i], -Inf) for
                i in 1:n
            ])

            if score > best_score
                optimal_score = score
                optimal_point = point
            end
        end

        if verbose
            next!(prog)
        end

        optimal_points[:, i] = optimal_point
        optimal_scores[i] = optimal_score

        matching_index = findfirst(x -> all(x .== optimal_point), eachcol(candidates)) # index with respect to the #                                                                              # total points 

        if isnothing(matching_index)
            println("max_bounds: ", max_bounds)
            println("Optimal point: ", optimal_point)
            error("Optimal point not found in candidates")
        end

        optimal_indices[i] = candidates_idx[matching_index]
    end

    if verbose
        finish!(prog)
    end

    return optimal_points, optimal_indices, num_additional_points
end

function intersiteProjection(
    candidate::Vector{Float64}, evaluated_points::Matrix{Float64}; alpha::Float64=0.5
)::Float64
    n = size(evaluated_points, 2) # evaluated_points is a (2, X) where each column is a point

    # compute a threshold value based on a given tolerance parameter (alpha)
    dmin = (2 * alpha) / n

    projection = minimum([
        LinearAlgebra.norm(candidate[i] - evaluated_point, -Inf) for
        evaluated_point in eachcol(evaluated_points)
    ])

    if projection >= dmin
        return minimum([
            LinearAlgebra.norm(candidate[i] - evaluated_point, 2) for
            evaluated_point in eachcol(evaluated_points)
        ])
    else
        return 0.0
    end
end

# ==============================================================================
#  Monte Carlo Intersite Projective Threshold
# ==============================================================================
function adaptiveRefinement(
    method::MIPT,
    total_points::Points,
    evaluated_points::Points,
    sys::System,
    sim::Simulation,
    methods::Methods,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64;
    verbose::Bool=false,
)
    x = deepcopy(evaluated_points)

    optimal_points, iterations = mipt(method, total_points, x; verbose=verbose)

    optimal_candidates_idx = []

    # Loop through each column of optimal_points and find the matching indices
    for col in eachcol(optimal_points)
        idx = findall(x -> all(x .== col), eachcol(total_points.coordinates))
        append!(optimal_candidates_idx, idx)
    end

    true_values, coefficient_variation = Evaluation.computeSurvivalSignatureEntry(
        sys, sim, optimal_points; verbose=verbose
    )

    x.coordinates = hcat(x.coordinates, optimal_points)
    x.solution = vcat(x.solution, true_values)
    x.confidence = vcat(x.confidence, coefficient_variation)
    x.idx = vcat(x.idx, optimal_candidates_idx)

    # recompute the basis Function
    basis = BasisFunction.basis(
        methods.basis_function_method, shape_parameter, x.coordinates, centers
    )
    weights = lsqr(basis, x.solution, constraints)

    upper_bound = min.(x.solution .+ (x.solution .* x.confidence), 1.0)
    lower_bound = max.(x.solution .- (x.solution .* x.confidence), 0.0)

    replace!(upper_bound, NaN => 1 / (sim.samples + 1))
    replace!(lower_bound, NaN => 0.0)

    if verbose
        println("\tRequired Iterations: ", iterations)
    end

    return x, weights, upper_bound, lower_bound, iterations, shape_parameter
end

function mipt(method::MIPT, total_points::Points, evaluated_points::Points; verbose=verbose)
    evaluated_point_coordinates = evaluated_points.coordinates
    candidates, _ = remainingCandidates(total_points, evaluated_points)

    prog = if verbose
        ProgressMeter.ProgressThresh(method.threshold, "\tAdaptive Refinement")
    else
        nothing
    end

    optimal_points = Array{Float64}(undef, size(candidates, 1), 0)
    iterations = 0
    while iterations < min(method.maximum_points, size(candidates, 2))
        iterations += 1
        x_vals, y_vals, distances = calculateIntersiteProjection(
            evaluated_point_coordinates, candidates
        )
        best_score = maximum(distances)

        if prog !== nothing
            ProgressMeter.update!(prog, best_score)
        end

        if best_score < method.threshold
            iterations -= 1
            break
        end

        # plt = plotIntersiteProjection(
        #     evaluated_point_coordinates, x_vals, y_vals, distances
        # )

        best_index = argmax(distances)
        best_point = candidates[:, best_index]
        optimal_points = hcat(optimal_points, best_point)
        evaluated_point_coordinates = hcat(evaluated_point_coordinates, best_point)

        candidates, _ = remainingCandidates(candidates, best_point, [best_index])
    end

    return optimal_points, iterations
end

function intersite_projection(X, x, dmin)::Float64
    n = size(X, 2)

    projection = minimum([LinearAlgebra.norm(x - X[:, i], -Inf) for i in 1:n])

    if projection >= dmin
        return minimum([LinearAlgebra.norm(x - X[:, i], 2) for i in 1:n])
    else
        return eps()
    end
end

function calculateIntersiteProjection(evaluated_points, candidates)

    # Create the grid (discretize the [0,1] domain)
    x_vals = candidates[1, :]
    y_vals = candidates[2, :]

    dmin = (2 * 0.05) / size(evaluated_points, 2)  # Define dmin (can be adjusted as needed)

    # Compute distances using the appropriate method
    distances = [
        intersite_projection(evaluated_points, candidate, dmin) for
        candidate in eachcol(candidates)
    ]

    return x_vals, y_vals, distances
end

function plotIntersiteProjection(X, x_vals, y_vals, distances)
    default(;
        framestyle=:box,
        label=nothing,
        grid=true,
        widen=false,
        size=(300, 300),
        titlefontsize=8,
        guidefontsize=16,
        legendfontsize=7,
        tickfontsize=12,
        left_margin=-2 * Plots.mm,
        bottom_margin=-2 * Plots.mm,
        fontfamily="Computer Modern",
        dpi=600,
    )

    color = :oslo  # Color scheme for the heatmap
    levels = 15  # Number of levels for the heatmap

    scatter(
        x_vals,
        y_vals;
        zcolor=distances,
        color=:oslo,
        markerstrokewidth=0,
        markersize=3,
        label="",
    )

    # Overlay the starting points as white dots
    return scatter!(X[2, :], X[1, :]; color=:yellow, markersize=3, label="")
end

# ==============================================================================
#  Space-Filling Cross-Validation Takeoff
# ==============================================================================
function adaptiveRefinement(
    method::SFCVT,
    total_points::Points,
    evaluated_points::Points,
    sys::System,
    sim::Simulation,
    methods::Methods,
    weights::Array,
    centers::Array,
    constraints::Array,
    shape_parameter::Float64;
    verbose::Bool=false,
)
    x = deepcopy(evaluated_points)

    iterations = 0
    f_max = maximum(evaluated_points.solution)

    prog = if verbose
        ProgressMeter.ProgressThresh(sim.weight_change_tolerance, "\tAdaptive Refinement")
    else
        nothing
    end

    stop = 0
    while stop < 2 && iterations < method.max_iterations
        iterations += 1

        candidates, _ = remainingCandidates(total_points, x)
        errors = relativeLOOCVErrorRBF(methods, x, centers, shape_parameter)
        space_filling_metric = spaceFillingMetric(candidates, x.coordinates)

        # Errors are considered the solutions in this new model.
        error_basis = BasisFunction.basis(
            methods.basis_function_method, shape_parameter, x.coordinates, centers
        )
        error_weights = lsqr(error_basis, errors, constraints)

        best_candidate = sfvct(
            method.method,
            candidates,
            x.coordinates,
            centers,
            shape_parameter,
            error_weights,
            methods,
            space_filling_metric,
        )

        optimal_indices = findfirst(
            x -> all(x .== best_candidate), eachcol(total_points.coordinates)
        )

        if isnothing(optimal_indices)
            println("\tWasted Iteration.")
            iterations -= 1
            continue
        end

        candidate_solution, coefficient_variation = Evaluation.computeSurvivalSignatureEntry(
            sys, sim, best_candidate; verbose=verbose
        )

        x.coordinates = hcat(x.coordinates, best_candidate)
        x.solution = vcat(x.solution, candidate_solution)
        x.confidence = vcat(x.confidence, coefficient_variation)
        x.idx = vcat(x.idx, optimal_indices)

        old_weights = weights

        # Recompute the basis Function
        basis = BasisFunction.basis(
            methods.basis_function_method, shape_parameter, x.coordinates, centers
        )

        # Update weights
        weights = lsqr(basis, x.solution, constraints)

        if isa(methods.weight_change_method, NORM)
            error_value = calculateError(methods.weight_change_method, weights, old_weights)

        elseif isa(methods.weight_change_method, NRMSE)
            predicted_values::Vector{Float64} = [
                (BasisFunction.basis(methods.basis_function_method, shape_parameter, y, centers) * weights)[1]
                for y::Vector in eachcol(best_candidate)
            ]
            error_value, f_max = calculateError(
                methods.weight_change_method, candidate_solution, predicted_values, f_max
            )

        else
            error("Unsupported error calculation method")
        end

        if error_value < sim.weight_change_tolerance
            stop += 1
        else
            stop = 0
        end

        if prog !== nothing
            stop != 1 && ProgressMeter.update!(prog, error_value)
        end
    end

    # plt = Visualization.plotSurrogateComparison(rbf_times, kriging_times)

    # CairoMakie.save("rbf_vs_kriging.pdf", plt)
    # CairoMakie.display(plt)

    upper_bound = min.(x.solution .+ (x.solution .* x.confidence), 1.0)
    lower_bound = max.(x.solution .- (x.solution .* x.confidence), 0.0)

    replace!(upper_bound, NaN => 1 / (sim.samples + 1))
    replace!(lower_bound, NaN => 0.0)

    if verbose
        println("\tRequired Iterations: ", iterations)
    end

    return x, weights, upper_bound, lower_bound, iterations, shape_parameter
end

function sfvct(
    method::EnumerationX,
    candidates::Matrix{Float64},
    evaluated_points::Matrix{Float64},
    centers::Matrix{Float64},
    shape_parameter::Float64,
    error_weights::Array,
    methods::Methods,
    space_filling_metric::Float64,
)
    best_value = -Inf
    best_candidate = nothing
    lock_obj = ReentrantLock()

    # Necessary for threading
    candidate_columns = collect(eachcol(candidates))

    # Measure the total time for the threaded operation
    @threads for candidate::Vector in candidate_columns
        # Measure time for each candidate
        if sfcvtConstraints(candidate, evaluated_points, space_filling_metric) >= 0
            value, _ = Evaluation.evaluateSurrogate(
                candidate, error_weights, shape_parameter, centers, methods
            )

            lock(lock_obj) do
                if value > best_value
                    best_value = value
                    best_candidate = candidate
                end
            end
        end
    end

    return best_candidate
end

function sfvct(
    method::OptimX,
    candidates::Matrix{Float64},
    evaluated_points::Matrix{Float64},
    centers::Matrix{Float64},
    shape_parameter::Float64,
    error_weights::Array,
    methods::Methods,
    space_filling_metric::Float64,
)
    # Define the objective function for optimization
    function objective(candidate::Vector)
        candidate = round.(Int, candidate)  # Round to nearest integer

        if sfcvtConstraints(candidate, evaluated_points, space_filling_metric) >= 0
            s, _ = Evaluation.evaluateSurrogate(
                candidate, error_weights, shape_parameter, centers, methods
            )
            return -s
        else
            return Inf  # Penalize infeasible candidates
        end
    end

    # Choose the initial candidate (you can modify this selection)
    midpoint = Statistics.mean(candidates; dims=2)[:, 1]  # Take the mean along each row (dimension)
    midpoint = Float64.(round.(Int, midpoint))  # Round the midpoint to ensure it's an integer

    # Set bounds (define lower and upper bounds based on your problem)
    lower_bounds = minimum(candidates; dims=2)[:, 1]  # Set minimum value per dimension
    upper_bounds = maximum(candidates; dims=2)[:, 1]  # Set maximum value per dimension

    # Optimize using Fminbox with bounds
    result = optimize(
        objective, lower_bounds, upper_bounds, midpoint, Fminbox(NelderMead())
    )

    # Extract the best candidate
    best_candidate = Optim.minimizer(result)

    best_candidate = round.(Int, best_candidate)  # Round to nearest integer

    return best_candidate
end

function sfvct(
    method::BlackBoxX,
    candidates::Matrix{Float64},
    evaluated_points::Matrix{Float64},
    centers::Matrix{Float64},
    shape_parameter::Float64,
    error_weights::Array,
    methods::Methods,
    space_filling_metric::Float64,
)
    # Define the objective function for BlackBoxOptim
    function objective(candidate::Vector)
        candidate = round.(Int, candidate)  # Ensure integer candidates

        if sfcvtConstraints(candidate, evaluated_points, space_filling_metric) >= 0
            s, _ = Evaluation.evaluateSurrogate(
                candidate, error_weights, shape_parameter, centers, methods
            )
            return -s
        else
            return Inf  # Penalize infeasible candidates
        end
    end

    # Define the search space
    # Define the search space for each dimension as a tuple of ranges
    # Define the search space as a vector of tuples (min, max) for each dimension
    search_space = [
        (minimum(candidates[d, :]), maximum(candidates[d, :])) for
        d in 1:size(candidates, 1)
    ]

    # Optimize using BlackBoxOptim
    # Method chosen using 'compare_optimizers' function
    result = bboptimize(
        objective;
        SearchRange=search_space,
        Method=:separable_nes,
        MaxSteps=5000,
        TraceMode=:silent,
    )

    # Extract the best candidate
    best_candidate = round.(Int, BlackBoxOptim.best_candidate(result))

    return best_candidate
end

function sfvct(
    method::SimulatedAnnealingX,
    candidates::Matrix{Float64},
    evaluated_points::Matrix{Float64},
    centers::Matrix{Float64},
    shape_parameter::Float64,
    error_weights::Array,
    methods::Methods,
    space_filling_metric::Float64,
)
    # Define the objective function
    function objective(candidate::Vector)
        candidate = round.(Int, candidate)  # Ensure integer candidates

        # Check if candidate satisfies constraints
        if sfcvtConstraints(candidate, evaluated_points, space_filling_metric) >= 0
            # Evaluate using the custom surrogate evaluation function
            s, _ = Evaluation.evaluateSurrogate(
                candidate, error_weights, shape_parameter, centers, methods
            )
            return -s
        else
            return Inf  # Penalize infeasible candidates
        end
    end

    # Set the initial candidate (could also be random or predefined)
    midpoint = Statistics.mean(candidates; dims=2)[:, 1]  # Take the mean along each row (dimension)
    midpoint = Float64.(round.(Int, midpoint))  # Round the midpoint to ensure it's an integer

    result = Optim.optimize(
        objective,
        [1.0, 1.0],
        candidates[:, end],
        midpoint,
        Optim.SAMIN(),
        Optim.Options(; iterations=10000),
    )

    # Extract the best candidate
    best_candidate = Optim.minimizer(result)

    best_candidate = round.(Int, best_candidate)  # Ensure integers

    return best_candidate
end

function sfvct(
    method::EvolutionX,
    candidates::Matrix{Float64},
    evaluated_points::Matrix{Float64},
    centers::Matrix{Float64},
    shape_parameter::Float64,
    error_weights::Array,
    methods::Methods,
    space_filling_metric::Float64,
)
    # Define the objective function for Evolutionary.jl
    function objective(candidate::Vector)
        candidate = round.(Int, candidate)  # Ensure integer candidates

        if sfcvtConstraints(candidate, evaluated_points, space_filling_metric) >= 0
            s, _ = Evaluation.evaluateSurrogate(
                candidate, error_weights, shape_parameter, centers, methods
            )
            return -s
        else
            return Inf  # Penalize infeasible candidates
        end
    end

    # Set bounds for the optimization variables
    lower_bounds = fill(1.0, size(candidates, 1))  # Adjust bounds as necessary
    upper_bounds = fill(maximum(candidates), size(candidates, 1))

    population_size = 100
    dimensions = length(lower_bounds)

    # Set the initial candidate (could also be random or predefined)
    population = [
        rand(dimensions) .* (upper_bounds .- lower_bounds) .+ lower_bounds for
        _ in 1:population_size
    ]
    # Optimize using Genetic Algorithm (GA)
    result = Evolutionary.optimize(
        objective, population, GA(; populationSize=population_size)
    )

    # Extract the best candidate (already an integer because of rounding)
    best_candidate = result.minimizer
    best_candidate = round.(Int, best_candidate)  # Ensure integers

    return best_candidate
end

function relativeLOOCVErrorRBF(
    methods::Methods, starting_points::Points, centers::Matrix, shape_parameter::Float64
)
    full_solutions = starting_points.solution
    errors = zeros(Float64, size(starting_points.coordinates, 2))

    for (idx, candidate::Vector) in enumerate(eachcol(starting_points.coordinates))
        # Remove the candidate from the starting_points
        remaining_points = starting_points.coordinates[
            :, setdiff(1:size(starting_points.coordinates, 2), idx)
        ]
        remaining_solutions = full_solutions[.!in.(1:length(full_solutions), Ref(idx))]

        # Recompute the basis function
        basis = BasisFunction.basis(
            methods.basis_function_method, shape_parameter, remaining_points, centers
        )

        # Update weights
        loocv_weights = pinv(basis) * remaining_solutions

        # Compute the predicted value
        predicted_value, _ = Evaluation.evaluateSurrogate(
            candidate, loocv_weights, shape_parameter, centers, methods
        )
        # Calculate the error
        errors[idx] = LinearAlgebra.norm(full_solutions[idx] - predicted_value)
    end

    return errors
end

function relativeLOOCVErrorKriging(
    methods::Methods, starting_points::Points, centers::Matrix, shape_parameter::Float64
)
    full_solutions = starting_points.solution
    errors = zeros(Float64, size(starting_points.coordinates, 2))

    for (idx, candidate::Vector) in enumerate(eachcol(starting_points.coordinates))
        # Remove the candidate from the starting_points
        remaining_points = starting_points.coordinates[
            :, setdiff(1:size(starting_points.coordinates, 2), idx)
        ]

        remaining_solutions = full_solutions[.!in.(1:length(full_solutions), Ref(idx))]

        # use the Kriging Method
        coordinate_tuples = [Tuple(col) for col in eachcol(remaining_points)]
        kriging_model = Kriging(
            coordinate_tuples,
            remaining_solutions,
            [minimum(starting_points.coordinates; dims=2)...],
            [maximum(starting_points.coordinates; dims=2)...],
        )

        # predict the value using the Kriging model.
        predicted_value = kriging_model(tuple(candidate...))

        # Calculate the error
        errors[idx] = LinearAlgebra.norm(full_solutions[idx] - predicted_value)
    end

    return errors
end

function spaceFillingCriterion(
    new_point::Vector{Float64}, existing_points::Matrix{Float64}
)::Float64
    # Measure time for distance calculation
    distances = [
        LinearAlgebra.norm(new_point .- point) for point in eachcol(existing_points)
    ]

    return minimum(distances)
end

function spaceFillingMetric(
    candidates::Matrix{Float64}, existing_points::Matrix{Float64}
)::Float64
    # Measure time for maximum distance calculation
    d_max_min = maximum([
        spaceFillingCriterion(candidate, existing_points) for
        candidate::Vector in eachcol(candidates)
    ])

    return 0.5 * d_max_min
end

function sfcvtConstraints(
    candidate::Union{Vector{Float64},Vector{Int64},Vector{JuMP.VariableRef}},
    existing_points::Matrix{Float64},
    space_filling_metric::Float64,
)
    # Measure time for minimum distance calculation
    min_distance = minimum([
        LinearAlgebra.norm(candidate .- existing_points[:, i]) for
        i in 1:size(existing_points, 2)
    ])

    return min_distance - space_filling_metric
end

# ==============================================================================

end
