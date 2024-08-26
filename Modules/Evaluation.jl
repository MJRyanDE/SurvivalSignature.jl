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

        if isinf(model.Phi.solution[idx])   # only evaluate non-evaluated points
            model.Phi.solution[idx] = evaluateSurrogate(
                model.Phi.coordinates[idx],
                model.model.weights,
                model.model.shape_parameter,
                model.model.centers,
                model.method,
            )
        end
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
        solutions[idx] = evaluateSurrogate(
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
)::Float64
    basis = BasisFunction.basis(
        method.basis_function_method, shape_parameter, state_vector, centers
    )

    s = (basis * weights)[1]

    # fit between 0 and 1 
    return min(max(s, 0.0), 1.0)
end

function computeSurvivalSignatureEntry(
    sys::System, sim::Simulation, state_vector::Array{<:Number}; verbose::Bool=false
)
    components_per_type = groupComponents(sys.types)

    progress = verbose ? Progress(size(state_vector, 2); desc="\tComputing...") : nothing

    y = map(eachcol(state_vector)) do x
        entry = CartesianIndex(Int.(x)...)
        num_combinations = numberofcombinations(components_per_type, entry)

        if num_combinations <= sim.samples
            # Exact value with coefficient of variation 0
            return SurvivalSignature.exactentry(entry, sys.adj, sys.types, sys.connectivity),
            0.0
        else
            # Approximate value with calculated coefficient of variation
            return SurvivalSignature.approximateentry(
                entry,
                sys.adj,
                sys.types,
                sys.connectivity,
                sim.samples,
                sim.variation_tolerance,
            )
        end

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
    full_state_vectors = [Ω[i, :] for i in 1:size(Ω, 1) if sum(Ω[i, :] .- 1) >= 0]

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
