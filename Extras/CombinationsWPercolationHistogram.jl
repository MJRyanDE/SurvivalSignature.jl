using Plots
using StatsPlots

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
    margin=:3 * Plots.mm,
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

# Define ranges for num_components and num_types

sample_limit = 10000
fc = 0.5

num_components = 20
num_types = 2

total_components = distribute_components(num_components, num_types)

state_vectors, num_state_vectors = generateStateVectors(total_components)

# Initialize containers for the plot data
combination_amounts = Vector{BigInt}()

println("Counting Combinations...")
for state_vector in state_vectors
    push!(combination_amounts, countCombinations(state_vector, state_vectors[end]))
end
println("Combinations Counted.")

threshold = (1 - fc) * sum(total_components)

println(threshold)

# Convert combination_amounts to Float64 for plotting
combination_amounts_float = map(Float64, combination_amounts)

threshold_bool = [sum(sv) >= threshold for sv in state_vectors]

percolated_combinations = combination_amounts_float .* threshold_bool

sample_limited = [min(sample_limit, c) for c in combination_amounts_float]

percolated_sample_limted = sample_limited .* threshold_bool

# Create string labels for the state vectors
state_vector_labels = [string(sv) for sv in state_vectors]

# Select five positions evenly spaced across the range
tick_positions = round.(Int, LinRange(1, length(combination_amounts_float), 8))

# Select the corresponding labels from state_vector_labels
tick_labels = state_vector_labels[tick_positions]

# Create a bar plot
bar(
    1:length(combination_amounts_float),  # X-axis: Indices of the state vectors
    combination_amounts_float;            # Y-axis: Combination counts
    xlabel="State Vector",                # Label for X-axis
    ylabel="Combinations",      # Label for Y-axis
    legend=:topleft,                         # No legend needed
    linecolor="#364652",                # Remove the borders of the bars
    color="#364652",                    # Change the bar color
    label="Without Optimization",
    xticks=(tick_positions, tick_labels), # Set the X-axis ticks and labels
    xrotation=45,                         # Rotate the X-axis labels
)

# Overlay the second bar chart

# Overlay the third bar chart
bar!(
    1:length(percolated_combinations),  # X-axis: Indices of the state vectors
    percolated_combinations;            # Y-axis:
    label="Percolated",
    linecolor=:tomato,                  # Remove the borders of the bars
    color=:tomato,
)

bar!(
    1:length(sample_limited),  # X-axis: Indices of the state vectors
    sample_limited;            # Y-axis:
    label="Approximation",
    linecolor=:dodgerblue,      # Remove the borders of the bars
    color=:dodgerblue,
)

# Overlay the fourth bar chart
bar!(
    1:length(percolated_sample_limted),  # X-axis: Indices of the state vectors
    percolated_sample_limted;            # Y-axis:
    label="Both",
    linecolor="#7DCE82",                # Remove the borders of the bars
    color="#7DCE82",
)

# Display the plot
display(current())

# Save the figure to a PDF file
savefig("optimization.pdf")

println("Plotting Complete.")
