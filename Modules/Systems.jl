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

# function distanceMatrix(method::GridSystem, adjacency_matrix::Matrix{Int64})

#     # creates a distance matrix using the adjacency matrix. 
#     # since there are no given coordinates, the distances are calculated as 
#     # a product of the index number of the adjacency matrix.

#     is_symmetric = adjacency_matrix == adjacency_matrix'

#     # function to convert linear index to grid coordinates
#     # only functions for 2-D grids.
#     function index_to_coordinates(idx::Int, dims::Tuple{Int,Int})
#         n, m = dims
#         row = div(idx - 1, m) + 1
#         col = mod(idx - 1, m) + 1
#         return (row, col)
#     end

#     distance_matrix = fill(NaN, size(adjacency_matrix))

#     # determine the coodinates of each node from the adj matrix
#     equivalent_node = [
#         index_to_coordinates(i, method.dims) for i in 1:size(adjacency_matrix, 1)
#     ]

#     if is_symmetric
#         # Calculate distances only for the upper triangle if symmetric
#         for i in 1:size(adjacency_matrix, 1)
#             for j in i:size(adjacency_matrix, 2)
#                 if isnan(distance_matrix[i, j])
#                     if adjacency_matrix[i, j] == 1
#                         row1, col1 = equivalent_node[i]
#                         row2, col2 = equivalent_node[j]
#                         distance = sqrt((row1 - row2)^2 + (col1 - col2)^2)  # Euclidean distance#

#                         distance_matrix[i, j] = distance
#                         distance_matrix[j, i] = distance # Use symmetry
#                     else
#                         distance_matrix[i, j] = Inf  # Set distance to infinity if not connected
#                         distance_matrix[j, i] = Inf  # Use symmetry
#                     end
#                 else
#                     continue
#                 end
#             end
#         end
#     else
#         # Handle non-symmetric (directed) graph case
#         for i in 1:size(adjacency_matrix, 1)
#             for j in 1:size(adjacency_matrix, 2)
#                 if adjacency_matrix[i, j] == 1
#                     row1, col1 = equivalent_node[i]
#                     row2, col2 = equivalent_node[j]
#                     distance = sqrt((row1 - row2)^2 + (col1 - col2)^2)  # Euclidean distance
#                     distance_matrix[i, j] = distance
#                 else
#                     distance_matrix[i, j] = Inf  # Set distance to infinity if not connected
#                 end
#             end
#         end
#     end

#     return distance_matrix
# end

# ==============================================================================
end
