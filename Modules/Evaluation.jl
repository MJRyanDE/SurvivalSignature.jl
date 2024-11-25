module Evaluation
# ==============================================================================
using ProgressMeter # @showprogress
# ==============================================================================

using SurvivalSignature

#include("Structures.jl")
using ..Structures: System, Simulation, Methods, Points, Model
using ..SurvivalSignatureUtils
using ..BasisFunction

# access to exactentry, approximateentry, numberofcombinations
include("../src/signature.jl")

# ==============================================================================
export computeSurvivalSignatureEntry, generateStateVectors, evaluate, evaluateSurrogate
# ==============================================================================

function evaluate(model::Model)::Model
    for idx in CartesianIndices(model.Phi.coordinates)
        @assert [Tuple(idx)...] == model.Phi.coordinates[idx]

        #if isinf(model.Phi.solution[idx])   # only evaluate non-evaluated points
        model.Phi.solution[idx], _ = evaluateSurrogate(
            model.Phi.coordinates[idx],
            model.model.weights,
            model.model.shape_parameter,
            model.model.centers,
            model.method,
        )
        # end
    end

    return model
end

function evaulate(
    coordinates::Union{Array{Float64},Array{Int}},
    weights::Array{Float64},
    shape_parameter::Float64,
    centers::Matrix{Float64},
    method::Method,
)
    solutions = zeros(Float64, size(coordinates, 2))

    for (idx, coordinate::Vector) in enumerate(eachcol(coordinates))
        solutions[idx], _ = evaluateSurrogate(
            coordinate, weights, shape_parameter, centers, method
        )
    end
    return solutions
end

function evaluateSurrogate(
    state_vector::Vector,
    weights::Array,
    shape_parameter::Float64,
    centers::Matrix,
    method::Methods,
)::Tuple{Float64,Float64}
    # Compute the Gaussian basis functions for the candidate point (state_vector)
    basis_values = BasisFunction.basis(
        method.basis_function_method, shape_parameter, state_vector, centers
    )

    # Surrogate prediction (weighted sum of basis functions)
    prediction = (basis_values * weights)[1]

    # Clamp the prediction between 0 and 1
    prediction = min(max(prediction, 0.0), 1.0)

    # the ideal Gaussian basis funciton sums to 1, so comparing the calculated
    # sum to 1 gives the RMSE.
    # the sum must be clamped to avoid negative values which occur due to the 
    # addition of eps() in the basis function normalization calculation.
    # the clamped value at eps() instead of 0 is to avoid division by zero
    # when the rmse is used to calculate the Expected Improvement.
    rmse = sqrt(max(eps(), 1 - sum(basis_values)))
    # naturally, this falls apart if the basis function isnt Gaussian, however, 
    # that functionality is not yet implemented.

    # Return both prediction and RMSE
    return prediction, rmse
end

function computeSurvivalSignatureEntry(
    sys::System, sim::Simulation, state_vector::Matrix{<:Number}; verbose::Bool=false
)
    components_per_type = groupComponents(sys.types)

    progress = verbose ? Progress(size(state_vector, 2); desc="\tComputing...") : nothing

    y = []
    for x in eachcol(state_vector)
        entry = CartesianIndex(Int.(x)...)
        num_combinations = numberofcombinations(components_per_type, entry)

        if num_combinations <= sim.samples
            result = SurvivalSignature.exactentry(
                entry, sys.adj, sys.types, sys.connectivity
            ),
            0.0
        else
            result = SurvivalSignature.approximateentry(
                entry,
                sys.adj,
                sys.types,
                sys.connectivity,
                sim.samples,
                sim.variation_tolerance,
            )
        end

        push!(y, result)

        if verbose
            next!(progress)
        end
    end
    if verbose
        finish!(progress)
    end

    # unpack the tuple
    solution = float.(getindex.(y, 1))
    coefficient_variations = getindex.(y, 2)

    return solution, coefficient_variations
end

function computeSurvivalSignatureEntry(
    sys::System, sim::Simulation, state_vector::Vector{<:Number}; verbose::Bool=false
)
    components_per_type = groupComponents(sys.types)

    entry = CartesianIndex(Int.(state_vector)...)
    num_combinations = numberofcombinations(components_per_type, entry)

    if num_combinations <= sim.samples
        # Exact value with coefficient of variation 0
        y = SurvivalSignature.exactentry(entry, sys.adj, sys.types, sys.connectivity), 0.0
    else
        # Approximate value with calculated coefficient of variation
        y = SurvivalSignature.approximateentry(
            entry,
            sys.adj,
            sys.types,
            sys.connectivity,
            sim.samples,
            sim.variation_tolerance,
        )
    end

    # unpack the tuple
    solution = float.(y[1])
    coefficient_variation = y[2]

    return solution, coefficient_variation
end

# move this function to a more appropriate module
function generateStateVectors(sys::System)::Tuple{Array,Array,Float64}
    components_per_type = groupComponents(sys.types)

    # cartersian coordinates of number of each type. 
    Ω = mapreduce(
        t -> [t...]', vcat, Iterators.product([1:c for c in components_per_type]...)
    )
    if sys.percolation
        fc = SurvivalSignature.percolation(sys.adj)
        threshold = sum(components_per_type .- 1) * (1 - fc)
    else
        threshold = 0.0
    end

    # percolate based on threshold
    percolated_state_vectors = [
        Ω[i, :] for i in 1:size(Ω, 1) if sum(Ω[i, :] .- 1) >= threshold
    ]

    #full version
    full_state_vectors = [Ω[i, :] for i in 1:size(Ω, 1) if sum(Ω[i, :] .- 1) >= 0.0]

    full_state_vectors = hcat([x for x in eachcol(float.(hcat(full_state_vectors...)))]...)
    percolated_state_vectors = hcat(
        [x for x in eachcol(float.(hcat(percolated_state_vectors...)))]...
    )

    return full_state_vectors, percolated_state_vectors, threshold
end

# move this function to a more appropriate module
function groupComponents(types::Dict{Int64,Vector{Int64}})
    components_per_type = ones(Int, length(types)) # initalize variable

    for (type, components) in types
        components_per_type[type] += length(components)
    end

    return components_per_type
end

# ==============================================================================

end

# ==============================================================================
