using Plots

using LinearAlgebra
using Random
using LaTeXStrings

abstract type Method end
struct Intersite <: Method end
struct Projection <: Method end
struct IntersiteProjection <: Method end
struct IntersiteProjectionThreshold <: Method end

# Function to calculate intersite projection for all points (without threshold)
function intersite_projection(method::IntersiteProjection, X, x, dmin)
    n = size(X, 2) # Number of starting points

    # Calculate intersite (Euclidean distance)
    intersite =
        ((sqrt(n + 1) - 1) / 2) * minimum([LinearAlgebra.norm(x - X[:, i], 2) for i in 1:n])

    # Calculate projection (Chebyshev distance)
    projection = minimum([LinearAlgebra.norm(x - X[:, i], -Inf) for i in 1:n])

    return intersite + projection
end

function intersite_projection(method::Intersite, X, x, dmin)
    n = size(X, 2) # Number of starting points

    # Calculate intersite (Euclidean distance)
    intersite =
        ((sqrt(n + 1) - 1) / 2) * minimum([LinearAlgebra.norm(x - X[:, i], 2) for i in 1:n])

    return intersite
end
function intersite_projection(method::Projection, X, x, dmin)
    n = size(X, 2) # Number of starting points

    # Calculate intersite (Euclidean distance)
    projection = minimum([LinearAlgebra.norm(x - X[:, i], -Inf) for i in 1:n])

    return projection
end

# Function to calculate intersite projection with threshold method
function intersite_projection(method::IntersiteProjectionThreshold, X, x, dmin)::Float64
    n = size(X, 2)

    projection = minimum([LinearAlgebra.norm(x - X[:, i], -Inf) for i in 1:n])

    if projection >= dmin
        return minimum([LinearAlgebra.norm(x - X[:, i], 2) for i in 1:n])
    else
        return eps()
    end
end

# Calculate intersite projection for a grid for a specific method
function calculateIntersiteProjection(method::Method, X, grid_size)

    # Create the grid (discretize the [0,1] domain)
    x_vals = range(0, 1; length=grid_size)
    y_vals = range(0, 1; length=grid_size)

    dmin = (2 * 0.05) / size(X, 2)  # Define dmin (can be adjusted as needed)

    # Compute distances using the appropriate method
    distances = [intersite_projection(method, X, [x; y], dmin) for x in x_vals, y in y_vals]

    return x_vals, y_vals, distances
end

# Plot the heatmap
function plotIntersiteProjection(X, x_vals, y_vals, distances, color, levels)

    # Create a heatmap of the calculated distances
    contourf(
        x_vals,
        y_vals,
        distances;
        levels=levels,
        lw=0.25,
        colorbar=false,
        color=color,
        xlabel=L"x_1",
        ylabel=L"x_2",
    )

    # Overlay the starting points as white dots
    return scatter!(X[2, :], X[1, :]; color=:white, markersize=3, label="")
end

# Main function
function main()
    default(;
        framestyle=:box,
        label=nothing,
        grid=true,
        widen=false,
        size=(450, 450),
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
    # Generate 10 random starting points in the [0,1] x [0,1] domain
    Random.seed!(420)
    X = rand(2, 10)  # 10 starting points in 2D

    grid_size = 50  # Grid resolution (increase for finer plot)

    x_vals, y_vals, distances = calculateIntersiteProjection(Intersite(), X, grid_size)
    plt = plotIntersiteProjection(X, x_vals, y_vals, distances, color, levels)
    display(plt)
    savefig(plt, "intersite.pdf")

    x_vals, y_vals, distances = calculateIntersiteProjection(Projection(), X, grid_size)
    plt = plotIntersiteProjection(X, x_vals, y_vals, distances, color, levels)
    display(plt)
    savefig(plt, "proj.pdf")

    x_vals, y_vals, distances = calculateIntersiteProjection(
        IntersiteProjection(), X, grid_size
    )
    plt = plotIntersiteProjection(X, x_vals, y_vals, distances, color, levels)
    display(plt)
    savefig(plt, "intersite-proj.pdf")

    x_vals, y_vals, distances = calculateIntersiteProjection(
        IntersiteProjectionThreshold(), X, grid_size
    )
    plt = plotIntersiteProjection(X, x_vals, y_vals, distances, color, levels)
    display(plt)
    savefig(plt, "intersite-proj-th.pdf")

    return println("Done.")
end

main()
