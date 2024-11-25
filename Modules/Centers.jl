module Centers

__precompile__()

# ==============================================================================

using Base.Threads
using ProgressMeter

using IterTools
using LinearAlgebra

using Profile

using Plots
using CairoMakie

# ==============================================================================

using ..Structures: System, Simulation
using ..Structures: CentersMethod, Grid, SparseGrid, Greedy, GeometricGreedy, Leja
using ..Structures: Gaussian, LaguerreGaussian
using ..Structures: Hardy
using ..BasisFunction
using ..ShapeParameter
using ..Evaluation

using ..SurvivalSignatureUtils

# ==============================================================================

export generateCenters
# ============================== METHODS =======================================

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

function generateCenters(
    method::Grid,
    sys::System,
    sim::Simulation,
    state_vectors::Matrix,
    threshold::Float64;
    verbose::Bool=true,
)
    # grid centers
    lb = minimum(state_vectors; dims=2)
    ub = maximum(state_vectors; dims=2)

    ranges = [range(l, u, c) for (l, u, c) in zip(lb, ub, method.centers_interval)]

    centers = hcat([[c...] for c in IterTools.Iterators.product(ranges...)]...)

    if method.percolation
        centers = hcat(
            [Vector{Float64}(x) for x in eachcol(centers) if sum(x .- 1) > threshold]...
        )
    end

    return centers
end

# ==============================================================================

function generateCenters(
    method::SparseGrid,
    sys::System,
    sim::Simulation,
    state_vectors::Matrix,
    threshold::Float64;
    verbose::Bool=true,
)

    # grid centers
    lb = minimum(state_vectors; dims=2)
    ub = maximum(state_vectors; dims=2)

    ranges = [range(l, u, c) for (l, u, c) in zip(lb, ub, method.centers_interval)]

    centers = hcat([[c...] for c in IterTools.Iterators.product(ranges...)]...)

    shifted_points = centers[1, 2:2:((method.centers_interval[1]))]

    middle = abs(centers[1, 1] - centers[1, 2]) / 2

    for (i, center) in enumerate(eachcol(centers))
        if any(abs.(center[1] .- shifted_points) .< 1e-6)
            centers[2, i] = center[2] + middle
        end
    end

    if method.percolation
        centers = hcat(
            [
                Vector{Float64}(x) for
                x in eachcol(centers) if sum(x .- 1) > threshold && all(x .<= ub)
            ]...,
        )
    end

    return centers
end

# ==============================================================================

function plotPowerValuesHeatmap(
    x_coords::Vector{Float64}, y_coords::Vector{Float64}, power_values::Vector{Float64}
)
    function _normalize(values::Vector{Float64})
        # Filter out NaNs for computing min and max
        valid_values = filter(!isnan, values)

        # Check if there are valid values to normalize
        if isempty(valid_values)
            return values  # Return original if all values are NaN
        end

        min_val = minimum(valid_values)
        max_val = maximum(valid_values)

        # Normalize values while preserving NaNs
        normalized_values = [
            (isnan(v) ? NaN : (v - min_val) / (max_val - min_val + eps())) for v in values
        ]
        return normalized_values
    end

    # Ensure x_coords, y_coords, and power_values are of compatible length
    if length(x_coords) != length(y_coords) || length(x_coords) != length(power_values)
        error("x_coords, y_coords, and power_values must all have the same length")
    end

    # Normalize the power values, preserving NaNs
    normalized_power_values = _normalize(power_values)

    # Find unique x and y values
    x_unique = unique(x_coords)
    y_unique = unique(y_coords)

    # Sort the unique values (important if they are not already sorted)
    x_unique = sort(x_unique)
    y_unique = sort(y_unique)

    # Create a matrix of power values (assuming that power_values are organized row-wise)
    power_grid = reshape(normalized_power_values, length(y_unique), length(x_unique))

    # Create a heatmap for the power values
    plt = Plots.heatmap(
        x_unique,       # Unique x coordinates for heatmap x-axis
        y_unique,       # Unique y coordinates for heatmap y-axis
        power_grid';    # Transpose to match expected x, y dimensions for plotting
        color=:plasma, # Use the :inferno color scheme
        nan_color=:black,
        colorbar=false,
    )

    return display(plt)
end

# ==============================================================================

function generateCenters(
    method::Greedy,
    sys::System,
    sim::Simulation,
    state_vectors::Matrix{Float64},
    threshold::Float64;
    verbose::Bool=true,
)
    min_val = minimum(state_vectors; dims=2)  # Minimum value in each row
    max_val = maximum(state_vectors; dims=2)  # Maximum value in each row

    # Normalize the data between -1 and 1
    normalized_state_vectors = 2 .* (state_vectors .- min_val) ./ (max_val .- min_val) .- 1

    centers = Matrix{Float64}(undef, size(state_vectors, 1), method.nCenters)

    # Select the first center arbitrarily
    start_point = round.(4 / 4 .* state_vectors[:, end], digits=0)

    selected_indices = [findfirst(col -> all(start_point .== col), eachcol(state_vectors))]  # Track indices in the original state_vectors
    selected_points = [normalized_state_vectors[:, selected_indices]]  # Start with the first point

    centers[:, 1] = start_point

    temp_shape_parameter = ShapeParameter.computeShapeParameter(Hardy(), state_vectors)

    progress = if verbose
        Progress(
            min(method.nCenters - 1, size(state_vectors, 2) - 1);
            desc="\tSelecting Centers...",
        )
    else
        nothing
    end

    f = Float64[]
    f_x, _ = Evaluation.computeSurvivalSignatureEntry(sys, sim, start_point)
    push!(f, f_x)

    all_power_values = Vector{Vector{Float64}}()
    max_power_overall = -Inf
    # Loop to find the remaining centers
    for i in 2:(min(method.nCenters, size(normalized_state_vectors, 2)))
        max_power, best_point, best_index = -Inf, nothing, nothing

        # Dynamically update the Lagrange coefficients for the current selected points
        lagrange_coefficients = lagrangeCoefficients(
            f, hcat(selected_points...), temp_shape_parameter
        )
        power_values = Float64[]
        for (j, point::Vector) in enumerate(eachcol(normalized_state_vectors))
            if j in selected_indices
                push!(power_values, NaN)
                continue  # Skip already selected points
            end
            # Compute power function for each potential new point
            power = abs(
                powerFunction(
                    hcat(selected_points...),
                    point,
                    lagrange_coefficients,
                    temp_shape_parameter,
                ),
            )
            push!(power_values, power)
            if power > max_power
                max_power, best_point, best_index = power, point, j
            end
            max_power_overall = max_power
        end

        push!(all_power_values, copy(power_values))

        x_coords = state_vectors[1, :]
        y_coords = state_vectors[2, :]

        # Plot the heatmap for the current iteration, power_values contains NaNs for skipped points
        # plotPowerValuesHeatmap(x_coords, y_coords, power_values)

        f_x, _ = computeSurvivalSignatureEntry(sys, sim, state_vectors[:, best_index])
        f = copy(f)
        push!(f, f_x)

        # Add the best point and its index to selected points and update the centers
        best_point = reshape(best_point, :, 1)

        push!(selected_points, best_point)
        push!(selected_indices, best_index)
        centers[:, i] = state_vectors[:, best_index]

        # Update the progress bar if verbose is enabled
        if verbose
            next!(progress)
        end

        #println("Power Norm: ", maximum(abs, power_values))
        if maximum(abs, power_values) < 2e-5 && i > 10
            println(
                "Stopping early as the power function is below the threshold: $maximum(abs, power_values)",
            )
            break
        end
    end

    if verbose
        finish!(progress)
    end

    centers = centers[:, 1:length(selected_points)]
    if method.percolation
        centers = hcat(
            [Vector{Float64}(x) for x in eachcol(centers) if sum(x .- 1) >= threshold]...
        )
    end

    return centers
end

function powerFunction(
    points::Matrix{Float64},
    new_point::Vector{Float64},
    lagrange_coefficients::Vector{Float64},
    temp_shape_parameter::Float64;
)
    num_points = size(points, 2)

    # for the gaussian kernel, this is equal to 1. however, this is not the case 
    # for all kernels, so this prepares for future changes with different kernels
    phi_xx = sum(
        BasisFunction.basis(Gaussian(), temp_shape_parameter, new_point, new_point)
    )

    power = 0.0
    @inbounds @simd for j in 1:num_points
        phi_x = sum(
            BasisFunction.basis(Gaussian(), temp_shape_parameter, new_point, points[:, j])
        )

        power += lagrange_coefficients[j] * phi_x
    end

    power = phi_xx - power

    return power
end

function lagrangeCoefficients(
    f::Vector{Float64}, points::Matrix{Float64}, shape_parameter::Number
)
    nPoints = size(points, 2)

    # Compute the shape parameter
    denominator = 2 * shape_parameter^2

    # Preallocate the matrix A
    A = zeros(nPoints, nPoints)

    # Calculate pairwise squared distances and fill matrix A
    @threads for i in 1:nPoints
        for j in 1:nPoints
            diff = points[:, i] - points[:, j]
            A[i, j] = exp(-sum(diff .^ 2) / denominator)
        end
    end

    cond_A = cond(A)
    if cond_A > 1e10
        λ = 1e-4 * maximum(diag(A))
        A = A + λ * I  # Add a small value to the diagonal
    end

    f = f .+ eps()

    #f = ones(nPoints)  # for now, we are just using the ones vector

    lagrange = lu(A) \ f

    return lagrange
end

# ==============================================================================

function generateCenters(
    method::GeometricGreedy,
    sys::System,
    sim::Simulation,
    state_vectors::Matrix{Float64},
    threshold::Float64;
    verbose::Bool=false,
)
    centers = Matrix{Float64}(undef, size(state_vectors, 1), method.nCenters)

    # Select the first center (boundary point)
    selected_points = [state_vectors[:, 1]]  # Start with the first point
    centers[:, 1] = state_vectors[:, 1]

    # Loop to find the remaining centers based on maximum distance

    progress = if verbose
        Progress(method.nCenters - 1; desc="\tSelecting Centers...")
    else
        nothing
    end

    for i in 2:(method.nCenters)
        max_distance, best_point = -Inf, nothing

        for point::Vector in eachcol(state_vectors)
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

    # its be decided to percolate after the centers are selected. this wastes some 
    # centers, but it makes it more consistent with the other methods.
    if method.percolation
        centers = hcat(
            [Vector{Float64}(x) for x in eachcol(centers) if sum(x .- 1) > threshold]...
        )
    end
    return centers
end

# ==============================================================================

function generateCenters(
    method::Leja,
    sys::System,
    sim::Simulation,
    state_vectors::Matrix{Float64},
    threshold::Float64;
    verbose::Bool=true,
)
    verbose = false

    min_val = minimum(state_vectors; dims=2)  # Minimum value in each row
    max_val = maximum(state_vectors; dims=2)  # Maximum value in each row

    # Normalize the data between -1 and 1
    normalized_state_vectors = 2 .* (state_vectors .- min_val) ./ (max_val .- min_val) .- 1

    centers = Matrix{Float64}(undef, size(state_vectors, 1), method.nCenters)

    temp_shape_parameter =
        method.scale * ShapeParameter.computeShapeParameter(Hardy(), state_vectors)

    idx = argmax([
        weightFunction(j, temp_shape_parameter) * LinearAlgebra.norm(j) for
        j::Vector in eachcol(normalized_state_vectors)
    ])

    selected_points = [normalized_state_vectors[:, idx]]  # Start with the first point
    centers[:, 1] = state_vectors[:, idx]

    progress = if verbose
        Progress(
            min(method.nCenters - 1, size(state_vectors, 2) - 1);
            desc="\tSelecting Centers...",
        )
    else
        nothing
    end

    # Loop to find the remaining centers
    for i in 2:(min(method.nCenters, size(state_vectors, 2)))
        max_score, best_point, best_index = -Inf, nothing, nothing

        for (j, point::Vector) in enumerate(eachcol(normalized_state_vectors))
            if point in selected_points
                continue  # Skip already selected points
            end

            score = 1.0
            for p::Vector in selected_points
                if p == point
                    continue
                end
                score *= LinearAlgebra.norm(point .- p)
            end

            weight = weightFunction(point, temp_shape_parameter)
            score *= weight

            # Select the point with the maximum of these minimum distances
            if score > max_score
                max_score, best_point, best_index = score, point, j
            end
        end
        # Add the best point and its index to selected points and update the centers
        best_point = reshape(best_point, :)

        push!(selected_points, best_point)
        centers[:, i] = state_vectors[:, best_index]

        # Conditionally update the progress bar
        if verbose
            next!(progress)
        end
    end

    if verbose
        finish!(progress)
    end

    # its be decided to percolate after the centers are selected. this wastes some 
    # centers, but it makes it more consistent with the other methods.
    if method.percolation
        centers = hcat(
            [Vector{Float64}(x) for x in eachcol(centers) if sum(x .- 1) > threshold]...
        )
    end
    return centers
end

function weightFunction(x::Vector{Float64}, alpha::Float64)::Float64
    return exp(-alpha^2 * LinearAlgebra.norm(x)^2)
end

# ==============================================================================
end