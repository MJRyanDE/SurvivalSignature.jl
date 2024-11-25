module BasisFunction

__precompile__()

# ==============================================================================

using IterTools
using LinearAlgebra
using JuMP

# ==============================================================================

using ..Structures: Gaussian, LaguerreGaussian
using ..SurvivalSignatureUtils

# ==============================================================================
export basis
# ==============================================================================

function basis(
    method::Gaussian,
    shape_parameter::Float64,
    coordinates::Union{Matrix,Vector},
    centers::Union{Matrix,Vector},
)
    dist = [
        LinearAlgebra.norm(x .- c) for
        (x, c) in IterTools.product(eachcol(coordinates), eachcol(centers))
    ]

    Ψ = exp.(-dist .^ 2 ./ (2 * shape_parameter^2))

    # normalization - eps() avoid division by zero
    return Ψ ./ (sum(Ψ; dims=2) .+ eps())
end

function basis(method::Gaussian, shape_parameter::Float64, dist::Float64)

    # used in indirect AMLS shape parameter method
    Ψ = exp.(-dist .^ 2 ./ (2 * shape_parameter^2))

    # normalization not necessary since Ψ is scalar
    return Ψ
end

function basis(method::Gaussian, shape_parameter::Float64, distance_matrix::Matrix{Float64})
    # from Fasshauer and Zhang - Program 2
    Ψ = exp.(-(shape_parameter .* distance_matrix) .^ 2)
    return Ψ ./ (sum(Ψ; dims=2) .+ eps())
end

# ==============================================================================

function laguerre(n::Int, alpha::Float64, x::Float64)
    sum = 0.0
    for k in 0:n
        term = ((-1)^k / factorial(k)) * binomial(n + alpha, n - k) * x^k
        sum += term
    end
    return sum
end

function basis(
    method::LaguerreGaussian,
    coordinates::Union{Matrix,Vector},
    shape_parameter::Float64,
    distance_matrix::Matrix,
)
    alpha = size(coordinates, 1) / 2

    return exp.(-shape_parameter^2 * distance_matrix .^ 2) .*
           laguerre.(method.order, alpha, shape_parameter^2 * distance_matrix .^ 2)
end

function basis(
    method::LaguerreGaussian,
    shape_parameter::Float64,
    coordinates::Union{Matrix,Vector},
    centers::Union{Matrix,Vector},
)
    dist = [
        LinearAlgebra.norm(x .- c) for
        (x, c) in IterTools.product(eachcol(coordinates), eachcol(centers))
    ]

    distance_matrix = reshape(dist, size(coordinates, 2), size(centers, 2))

    alpha = size(coordinates, 1) / 2

    return exp.(-shape_parameter^2 * distance_matrix .^ 2) .*
           laguerre.(method.order, alpha, shape_parameter^2 * distance_matrix .^ 2)
end

end