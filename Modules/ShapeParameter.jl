module ShapeParameter

__precompile__()

# ==============================================================================
using Plots
using Statistics
using IterativeSolvers

using NearestNeighbors
using LinearAlgebra # needed for norm
using Distances
using IterTools
using Plots
using Surrogates
using Convex # needed for Variable and evaluate
using SCS # needed for Optimizer

using Optim

# ==============================================================================

using ..Structures: System
using ..Structures: Points, Methods
using ..Structures: Hardy, Rippa, DirectAMLS, IterativeAMLS

using ..Structures: Gaussian, LaguerreGaussian

using ..SurvivalSignatureUtils

using ..BasisFunction

# ==============================================================================

export computeShapeParameter

# ============================= METHODS ========================================

function computeShapeParameter(
    method::Hardy,
    points::Array,
    starting_points::Points,
    centers::Array;
    verbose::Bool=false,
)::Float64

    # knn(k=2) returns the 2 closest points, since the 1. is itself 
    _, d = NearestNeighbors.knn(NearestNeighbors.KDTree(points), points, 2)

    d = sum(sum(d))

    return 1 / (0.815 * (d / size(points, 2)))
end

function computeShapeParameter(method::Hardy, points::Array)::Float64

    # knn(k=2) returns the 2 closest points, since the 1. is itself 
    _, d = NearestNeighbors.knn(NearestNeighbors.KDTree(points), points, 2)

    d = sum(sum(d))

    return 1 / (0.815 * (d / size(points, 2)))
end

function computeShapeParameter(
    method::Rippa,
    points::Array,
    starting_points::Points,
    centers::Array;
    verbose::Bool=false,
)
    distance_matrix = survialSignatureDistanceMatrix(starting_points.coordinates)
    distance_matrix = distance_matrix ./ maximum(distance_matrix) # normalize

    cost_function =
        ϵ -> begin
            cost = LOOCV(
                distance_matrix,
                starting_points.coordinates,
                starting_points.solution,
                centers,
                ϵ,
            )
            return cost
        end
    result = Optim.optimize(cost_function, 0.0, 10.0; method=Optim.Brent())
    ϵ_opt = Optim.minimizer(result)

    return ϵ_opt
end

function computeShapeParameter(
    method::DirectAMLS,
    points::Array,
    starting_points::Points,
    centers::Array;
    verbose::Bool=false,
)::Float64
    distance_matrix = survialSignatureDistanceMatrix(starting_points.coordinates)
    distance_matrix = distance_matrix ./ maximum(distance_matrix) # normalize
    cost_function =
        ϵ -> begin
            cost = directAMLS(
                distance_matrix,
                starting_points.coordinates,
                method.order,
                starting_points.solution,
                ϵ,
                method.max_iterations,
                method.tolerance;
                verbose=verbose,
            )
            return cost
        end
    result = Optim.optimize(cost_function, 0.0, 10.0; method=Optim.Brent())
    ϵ_opt = Optim.minimizer(result)
    return ϵ_opt
end

function computeShapeParameter(
    method::IterativeAMLS,
    points::Array,
    starting_points::Points,
    centers::Array;
    verbose::Bool=false,
)::Float64
    distance_matrix = survialSignatureDistanceMatrix(starting_points.coordinates)
    distance_matrix = distance_matrix ./ maximum(distance_matrix) # normalize
    cost_function =
        ϵ -> begin
            cost = optimized_iterativeAMLS(
                distance_matrix,
                starting_points.coordinates,
                method.order,
                starting_points.solution,
                ϵ,
                method.max_iterations,
                method.tolerance;
                verbose,
            )
            return cost
        end
    result = Optim.optimize(cost_function, 0.0, 10.0; method=Optim.Brent())
    ϵ_opt = Optim.minimizer(result)
    return ϵ_opt
end
# ==============================================================================

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

function LOOCV(
    distance_matrix::Matrix{Float64},
    coordinates::Array{Float64},
    solutions::Vector{Float64},
    centers::Array,
    shape_parameter::Float64,
)
    A = BasisFunction.basis(Gaussian(), shape_parameter, distance_matrix)

    if isSquare(A)
        return optimized_LOOCV(distance_matrix, solutions, shape_parameter)
    else
        return explict_LOOCV(coordinates, solutions, centers, shape_parameter)
    end
end

function optimized_LOOCV(
    distance_matrix::Matrix{Float64}, solutions::Vector{Float64}, shape_parameter::Float64
)
    # this optimization only works for square matrices

    # Compute the basis function
    A = BasisFunction.basis(Gaussian(), shape_parameter, distance_matrix)
    # if a matrix is almost sigular, use the pseudoinverse

    # Invert the interpolation matrix A
    pinvA = pinv(A)

    # Solve for the RBF coefficients
    c = pinvA * solutions

    # Compute the error components e_k as |c_k / A_inv_kk|
    errors = abs.(c ./ diag(pinvA))

    # Return the Euclidean norm of the error vector
    return LinearAlgebra.norm(errors)
end

function explict_LOOCV(
    coordinates::Matrix{Float64},
    solutions::Vector{Float64},
    centers::Matrix{Float64},
    shape_parameter::Float64,
)
    N = size(coordinates, 2)  # Number of points (columns)
    errors = zeros(Float64, N)  # Array to store the errors for each leave-one-out iteration

    # Iterate over each data point
    for k in 1:N
        # Create a new matrix excluding the k-th column without modifying the original matrix
        coords_loo = hcat(coordinates[:, 1:(k - 1)], coordinates[:, (k + 1):end])
        solutions_loo = vcat(solutions[1:(k - 1)], solutions[(k + 1):end])

        # Compute the interpolation matrix for the remaining data points
        A_loo = BasisFunction.basis(Gaussian(), shape_parameter, coords_loo, centers)

        # Solve for the RBF coefficients c using the pseudoinverse
        c_loo = pinv(A_loo) * solutions_loo

        # Compute the RBF values at the left-out point x_k relative to all centers
        x_k = coordinates[:, k]  # The left-out point
        rbf_values = zeros(Float64, size(centers, 2))  # Initialize RBF values

        # Calculate the RBF value for each center
        for i in 1:size(centers, 2)
            distance = norm(x_k - centers[:, i])  # Euclidean norm
            rbf_values[i] = BasisFunction.basis(Gaussian(), shape_parameter, distance)
        end

        # Compute the interpolated value at x_k using the calculated RBF values and the coefficients
        interpolated_value = dot(rbf_values, c_loo)

        # Calculate the error for the left-out point
        errors[k] = abs(solutions[k] - interpolated_value)
    end
    # Return the maximum error (infinity norm of the error vector)
    return maximum(errors)
end

# ==============================================================================

function directAMLS(
    distance_matrix::Matrix{Float64},
    coordinates::Union{Matrix,Vector},
    order::Int,
    solutions::Vector{Float64},
    shape_parameter::Float64,
    max_iterations::Int,
    tolerance::Float64;
    verbose::Bool=false,
)::Float64
    # the Direct AMLS method necessitates A be a square matrix, however this is 
    # only the case if the number of centers matches the number of starting points.

    A = BasisFunction.basis(
        LaguerreGaussian(order), coordinates, shape_parameter, distance_matrix
    )
    I = Matrix(LinearAlgebra.I(size(A, 1)))

    v_prev = solutions # initializing
    M_prev = I
    cost_prev = Inf
    cost = 0.0
    min_change = Inf
    for iter in 1:max_iterations
        v = (I - A) * v_prev .+ solutions

        # eigen dicomposition 
        eig = LinearAlgebra.eigen(I - A)
        Λ = eig.values
        X = eig.vectors

        M = Λ .* M_prev .+ I

        # cost vector
        e = v ./ diag(X * M * X')

        cost = LinearAlgebra.norm(e)

        change = abs(cost - cost_prev)

        if change < min_change
            min_change = change
        end

        if change < tolerance
            if verbose
                println("\tConverged after $iter iterations")
            end
            return cost
        end

        # convergence
        if change < tolerance
            return cost
        end

        cost_prev = cost
        M_prev = M
        v_prev = v
    end

    if verbose
        println("\tMin. Change: $min_change - Tolerance: $tolerance")
        println("\tWarning: Indirect AMLS did not converge within max_iterations")
    end

    return cost
end

function squareMatrix(A::Matrix)::Matrix
    # the purpose of this function is to make a square-matrix from a non-square matrix
    if isSquare(A)
        # if the matrix is already square, this is unnessesary
        return A
    else
        return A * A'
    end
end

function isSquare(matrix::Matrix)::Bool
    return size(matrix, 1) == size(matrix, 2)
end

# ==============================================================================
function optimized_iterativeAMLS(
    distance_matrix::Matrix{Float64},
    coordinates::Union{Matrix,Vector},
    order::Int,
    solutions::Vector{Float64},
    shape_parameter::Float64,
    max_iterations::Int,
    tolerance::Float64;
    verbose::Bool=false,
)::Float64
    N = length(solutions)

    # Step 1: Construct matrix A using RBFs
    A = BasisFunction.basis(
        LaguerreGaussian(order), coordinates, shape_parameter, distance_matrix
    )

    # Step 2: Perform eigen-decomposition of A
    Λ_values, X = eigen(A)  # Get eigenvalues (Λ_values) and eigenvectors (X)
    Λ = Diagonal(Λ_values)

    # Step 3: Initialize
    E0 = repeat(reshape(solutions, 1, N), N, 1)  # Ensure solutions is treated as a row vector
    S = X \ E0  # S^{(0)} = X^{-1} * E^{(0)}
    D_prev = Diagonal(diag(E0))  # D^{(0)} is the diagonal of E0

    # Extract e_prev (the vector form of D_prev)
    e_prev = diag(D_prev)
    cost_prev = norm(e_prev)  # Initial cost based on the norm of e^{(0)}

    min_change = Inf
    for iter in 1:max_iterations

        # Perform iterative update
        S = (I - Λ) * S + Λ * (inv(X) * D_prev)
        S = clamp.(S, -1e12, 1e12)
        # Compute D_new as a diagonal matrix
        D_new = Diagonal(diag(X * S))

        # Extract e_new (the vector form of D_new)
        e_new = diag(D_new)

        # Compute cost based on the norm of e_new
        cost = norm(e_new)

        change = abs(cost - cost_prev)

        if change < min_change
            min_change = change
        end

        if change < tolerance
            if verbose
                println("\tConverged after $iter iterations")
            end
            return cost
        end

        # Prepare for the next iteration
        D_prev = D_new
        e_prev = e_new
        cost_prev = cost
    end

    # If max_iterations reached without convergence
    if verbose
        println("\tMin. Change: $min_change - Tolerance: $tolerance")
        println("\tWarning: Indirect AMLS did not converge within max_iterations")
    end
    return cost_prev
end

# ==============================================================================
end