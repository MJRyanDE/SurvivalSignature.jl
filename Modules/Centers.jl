module Centers

__precompile__()

# ==============================================================================

using Base.Threads
using ProgressMeter

using IterTools
using LinearAlgebra

using Profile

# ==============================================================================

using ..Structures: CentersMethod, GridCenters, Greedy, GeometricGreedy, Leja
using ..Structures: Gaussian
using ..Structures: Hardy
using ..BasisFunction
using ..ShapeParameter

using ..SurvivalSignatureUtils

# ==============================================================================

export generateCenters
# ============================== METHODS =======================================

function generateCenters(
    method::GridCenters, state_vectors::Matrix, threshold::Float64; verbose::Bool=true
)
    # grid centers
    lb = minimum(state_vectors; dims=2)
    ub = maximum(state_vectors; dims=2)

    ranges = [range(l, u, c) for (l, u, c) in zip(lb, ub, method.centers_interval)]

    return hcat(
        [
            [c...] for
            c in IterTools.Iterators.product(ranges...) if sum(c .- 1) > threshold
        ]...,
    )
end

# ==============================================================================
function generateCenters(
    method::Greedy, state_vectors::Matrix{Float64}, threshold::Float64; verbose::Bool=true
)

    # this function does not work correctly. 

    percolated_state_vectors = hcat(
        [Vector{Float64}(x) for x in eachcol(state_vectors) if sum(x .- 1) > threshold]...
    )
    centers = Matrix{Float64}(undef, size(percolated_state_vectors, 1), method.nCenters)
    lagrange_coefficients = lagrangeCoefficients(percolated_state_vectors)

    # Select the first center arbitrarily
    selected_points = [percolated_state_vectors[:, 1]]  # Assuming we start with the first point
    selected_indices = [1]  # Track indices in the original percolated_state_vectors
    centers[:, 1] = percolated_state_vectors[:, 1]

    temp_shape_parameter = ShapeParameter.computeShapeParameter(
        Hardy(), percolated_state_vectors
    )

    progress = if verbose
        Progress(
            min(method.nCenters - 1, size(percolated_state_vectors, 2) - 1);
            desc="\tSelecting Centers...",
        )
    else
        nothing
    end

    # Loop to find the remaining centers
    for i in 2:(min(method.nCenters, size(percolated_state_vectors, 2)))
        max_power, best_point, best_index = -Inf, nothing, nothing

        for (j, point::Vector) in enumerate(eachcol(percolated_state_vectors))
            if j in selected_indices
                continue  # Skip already selected points
            end

            power = powerFunction(
                hcat(selected_points...),
                point,
                lagrange_coefficients[selected_indices],
                temp_shape_parameter,
            )
            if power > max_power
                max_power, best_point, best_index = power, point, j
            end
        end

        # Add the best point and its index to selected points and update the centers
        push!(selected_points, best_point)
        push!(selected_indices, best_index)
        centers[:, i] = best_point

        # Conditionally update the progress bar
        if verbose
            next!(progress)
        end
    end

    if verbose
        finish!(progress)
    end

    return centers
end

function powerFunction(
    points::Matrix{Float64},
    new_point::Vector{Float64},
    lagrange_coefficients::Vector{Float64},
    temp_shape_parameter::Float64,
)
    num_points = size(points, 2)

    # for the gaussian kernel, this is equal to 1. however, this is not the case 
    # for all kernels, so this prepares for future changes with different kernels
    phi_xx = BasisFunction.basis(Gaussian(), temp_shape_parameter, new_point, new_point)

    power = 0.0
    @inbounds @simd for j in 1:num_points
        phi_x = BasisFunction.basis(
            Gaussian(), temp_shape_parameter, points[:, j], new_point
        )
        power += lagrange_coefficients[j] * sum(phi_x)
    end

    power = sum(phi_xx) - power

    return power
end

function powerFunction(points::Matrix{Float64}, new_point::Vector{Float64})
    lagrange_coefficients = lagrangeCoefficients(points)
    return powerFunction(points, new_point, lagrange_coefficients)
end

function powerFunction(
    points::Matrix{Float64},
    new_point::Vector{Float64},
    lagrange_coefficients::Vector{Float64},
)
    temp_shape_parameter = ShapeParameter.computeShapeParameter(Hardy(), points)

    return powerFunction(points, new_point, lagrange_coefficients, temp_shape_parameter)
end

function lagrangeCoefficients(points::Matrix{Float64}; verbose::Bool=true)
    nPoints = size(points, 2)

    # Compute the shape parameter
    temp_shape_parameter = ShapeParameter.computeShapeParameter(Hardy(), points)
    denominator = 2 * temp_shape_parameter^2

    # Preallocate the matrix A
    A = zeros(nPoints, nPoints)

    # Calculate pairwise squared distances and fill matrix A

    progress = if verbose
        Progress(nPoints; desc="\tComputing Lagrange Coefficients...")
    else
        nothing
    end

    @threads for i in 1:nPoints
        for j in 1:nPoints
            diff = points[:, i] - points[:, j]
            A[i, j] = exp(-sum(diff .^ 2) / denominator)
        end

        if verbose
            next!(progress)
        end
    end

    if verbose
        finish!(progress)
    end

    # Solve the linear system
    if verbose
        start_time = time_ns()
        println("\t\tSolving Linear System...")
    end
    lagrange = lu(A) \ ones(nPoints)
    if verbose
        println("\t\t\tTime: $(round((time_ns() - start_time) / 1e9, digits = 3))s")
        println("\t\tLinear System Solved.")
    end

    return lagrange
end

# ==============================================================================

function generateCenters(
    method::GeometricGreedy,
    state_vectors::Matrix{Float64},
    threshold::Float64;
    verbose::Bool=false,
)
    # Percolate state vectors based on threshold
    percolated_state_vectors = hcat(
        [x for x in eachcol(state_vectors) if sum(x .- 1) > threshold]...
    )

    centers = Matrix{Float64}(undef, size(percolated_state_vectors, 1), method.nCenters)

    # Select the first center (boundary point)
    selected_points = [percolated_state_vectors[:, 1]]  # Start with the first point
    centers[:, 1] = percolated_state_vectors[:, 1]

    # Loop to find the remaining centers based on maximum distance

    progress = if verbose
        Progress(method.nCenters - 1; desc="\tSelecting Centers...")
    else
        nothing
    end

    for i in 2:(method.nCenters)
        max_distance, best_point = -Inf, nothing

        for point::Vector in eachcol(percolated_state_vectors)
            if point in selected_points
                continue  # Skip already selected points
            end

            # Compute the minimum distance to the current set of selected points
            min_distance = minimum([norm(point .- p) for p in selected_points])

            # Select the point with the maximum of these minimum distances
            if min_distance > max_distance
                max_distance, best_point = min_distance, point
            end
        end

        # Add the best point to selected points and update the centers
        push!(selected_points, best_point)
        centers[:, i] = best_point

        # Conditionally update the progress bar
        if verbose
            next!(progress)
        end
    end

    if verbose
        finish!(progress)
    end

    return centers
end

# ==============================================================================

function generateCenters(
    method::Leja, state_vectors::Matrix{Float64}, threshold::Float64; verbose::Bool=false
)
    # Percolate state vectors based on threshold
    percolated_state_vectors = hcat(
        [x for x in eachcol(state_vectors) if sum(x .- 1) > threshold]...
    )

    centers = Matrix{Float64}(undef, size(percolated_state_vectors, 1), method.nCenters)

    # Select the first center arbitrarily
    selected_points = [percolated_state_vectors[:, 1]]  # Start with the first point
    centers[:, 1] = percolated_state_vectors[:, 1]

    # Loop to find the remaining centers based on Leja sequence criterion

    progress = if verbose
        Progress(method.nCenters - 1; desc="\tSelecting Centers...")
    else
        nothing
    end

    for i in 2:(method.nCenters)
        max_min_distance, best_point = -Inf, nothing

        for point::Vector in eachcol(percolated_state_vectors)
            if point in selected_points
                continue  # Skip already selected points
            end

            # Compute the minimum distance to all selected points
            min_distance = minimum([norm(point .- p) for p in selected_points])

            # Select the point with the maximum of these minimum distances
            if min_distance > max_min_distance
                max_min_distance, best_point = min_distance, point
            end
        end

        # Add the best point to selected points and update the centers
        push!(selected_points, best_point)
        centers[:, i] = best_point

        # Conditionally update the progress bar
        if verbose
            next!(progress)
        end
    end

    if verbose
        finish!(progress)
    end

    return centers
end

# ==============================================================================
end