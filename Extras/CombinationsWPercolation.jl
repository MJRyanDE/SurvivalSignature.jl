using Plots

default(;
    framestyle=:box,
    label=nothing,
    grid=true,
    legend=:topleft,
    legend_font_halign=:left,
    size=(560, 300),
    titlefontsize=8,
    guidefontsize=8,
    legendfontsize=7,
    tickfontsize=8,
    fontfamily="Computer Modern",
    margin=:1 * Plots.mm,
    dpi=300,
)

function countCombinations(state_vector::Vector{Int64}, full_vector::Vector{Int64})::BigInt
    @assert length(state_vector) == length(full_vector)

    combinations = [
        binomial(BigInt(full_vector[i]), BigInt(state_vector[i])) for
        i in 1:length(state_vector)
    ]

    return prod(combinations)
end

function generateStateVectors(
    full_vector::Vector{Int64}
)::Tuple{Vector{Vector{Int64}},Int64}
    state_vectors = Vector{Vector{Int64}}()

    max_states = [0:full_vector[i] for i in 1:length(full_vector)]
    for state_vector in Iterators.product(max_states...)
        push!(state_vectors, collect(state_vector))
    end

    return state_vectors, length(state_vectors)
end

function distribute_components(num_components::Int64, num_types::Int64)::Vector{Int64}
    base_value::Int64 = div(num_components, num_types)
    remainder::Int64 = num_components % num_types

    component_vector = fill(base_value, num_types)

    for i in 1:remainder
        component_vector[i] += 1
    end

    return component_vector
end

percolations = [false, true, false, true]
approximations = [false, false, true, true]
labels = ["Without Optimization", "Percolation", "Approximation", "Both"]
new_colors = ["#364652", :tomato, :dodgerblue, "#7DCE82"]

fc = 0.35
samples = 10000

# Define ranges for num_components and num_types
num_components = [1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
num_types = 3

# Initialize containers for the plot data
combinations_results = []
state_vectors_results = []
num_components_all = []

println("Calculating Combinations and State Vectors...")

# Outer loop: iterate over different numbers of types
for (percolation, approximation) in zip(percolations, approximations)
    println("\tNumber of Types: $num_types")
    # Initialize results storage for each type
    combinations_count = fill(BigInt(0), length(num_components))
    state_vectors_count = Vector{BigInt}(undef, length(num_components))

    # Inner loop: iterate over different numbers of components
    for (i, num_component) in enumerate(num_components)
        println("\t\tNumber of Components: $num_component")
        total_components = distribute_components(num_component, num_types)

        if percolation
            threshold = (1 - fc) * sum(total_components)
            println("\t\t\tThreshold: $threshold")
        end

        state_vectors, num_state_vectors = generateStateVectors(total_components)
        state_vectors_count[i] = BigInt(num_state_vectors)
        percolated_state_vectors = 0
        for state_vector in state_vectors
            if percolation
                if sum(state_vector) < threshold
                    percolated_state_vectors += 1
                    continue
                end
            end

            count = countCombinations(state_vector, total_components)

            if approximation
                count = min(count, samples)
            end

            combinations_count[i] += count
        end
        if percolation
            println("\t\t\tPercolated State Vectors: $percolated_state_vectors")
            println("\t\t\tTotal State Vectors: $num_state_vectors")
            println(
                "\t\t\tPercent Percolated: $(round((percolated_state_vectors / num_state_vectors); digits=4) * 100)%",
            )
            println(" ")
        end
    end

    # Store results for plotting
    push!(combinations_results, combinations_count)
    push!(state_vectors_results, state_vectors_count)
    push!(num_components_all, num_components)
end

println("Combinations and State Vectors Calculated.\n")

println("Plotting Results...")

# Use Plots package to plot the results
# First plot: Combinations vs Components
p1 = plot(;
    xlabel="Components",
    ylabel="Combinations",
    yscale=:log10,
    legend=:topleft,
    xlim=[0, 150],
)
for (i, (num_components, combinations_result)) in
    enumerate(zip(num_components_all, combinations_results))
    plot!(p1, num_components, combinations_result; label=labels[i], color=new_colors[i])
end

# Second plot: State Vectors vs Components
p2 = plot(
    num_components,
    state_vectors_results[1];
    xlabel="Components",
    ylabel="State Vectors",
    yscale=:log10,
    color=:black,
    xlim=[0, 150],
)

# Combine plots into one figure with 2 subplots
plot_combined = plot(p2, p1; layout=(1, 2))

# Save the figure to a PDF file
savefig(plot_combined, "combinations_v_components_w_optimization.pdf")

println("Plotting Complete.")
