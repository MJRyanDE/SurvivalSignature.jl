module Systems

__precompile__()

# ==============================================================================
using LinearAlgebra
# ==============================================================================
using ..Structures: System, GridSystem

using ..SurvivalSignatureUtils

# needed for grid-network
include("../src/util.jl")
# needed for s_t_connectivity
include("../src/structurefunctions.jl")
# ==============================================================================

export generateSystem

# ==============================================================================
function generateSystem(method::GridSystem; percolation_bool::Bool=true)
    adj = gridnetwork(method.dims...)
    #  distance_matrix = distanceMatrix(method, adj)

    connectivity = s_t_connectivity([1:prod(method.dims);], [1], [prod(method.dims)])
    types = Dict(1 => collect(1:2:prod(method.dims)), 2 => collect(2:2:prod(method.dims))) # alternate every other type

    return System(adj, connectivity, types, percolation_bool)
end

# ==============================================================================
end
