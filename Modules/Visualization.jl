module Visualization
# the purpose of this module is to house the functions related to plotting and
# and text outputs

# ==============================================================================

using Plots, CairoMakie, LaTeXStrings

using DataFrames
using PrettyTables, ColorSchemes
using Statistics

# ==============================================================================

using ..Structures: System, Simulation, Methods, Model

using ..Structures: SimulationType, MonteCarloSimulation, IntervalPredictorSimulation
using ..Structures: CentersMethod, Grid, SparseGrid, Greedy, GeometricGreedy, Leja
using ..Structures: AdaptiveRefinementMethod, SFCVT
using ..Structures: DirectAMLS, IterativeAMLS

using ..SurvivalSignatureUtils

# ==============================================================================

export printDetails,
    plotError, plotTime, plotCentersSubPlot, printUltimateTes, plotSurrogateComparison

# ============================= TEXT ===========================================

function printUltimateTest(
    signatures::Union{Matrix{Model},Vector{Model},Model}; latex::Bool=false
)
    method_data = []
    timing_data = []
    for signature in signatures
        push!(
            method_data,
            (
                nameof(typeof(signature.method.centers_method)),
                nameof(typeof(signature.method.shape_parameter_method)),
                nameof(typeof(signature.method.adaptive_refinement_method)),
                signature.metrics.iterations,
                round(signature.errors["RMSE"]; digits=4),
                round(signature.errors["RAE"]; digits=4),
                signature.metrics.time.total,
            ),
        )
        push!(
            timing_data,
            (
                nameof(typeof(signature.method.centers_method)),
                nameof(typeof(signature.method.shape_parameter_method)),
                nameof(typeof(signature.method.adaptive_refinement_method)),
                signature.metrics.time.total,
                signature.metrics.time.centers,
                signature.metrics.time.shape_parameter,
                signature.metrics.time.adaptive_refinement,
                signature.metrics.iterations,
            ),
        )
    end

    errors_rmse = [d[5] for d in method_data]
    errors_rae = [d[6] for d in method_data]
    total_times = [d[7] for d in method_data]
    iterations = [d[8] for d in timing_data]

    function normalize(arr)
        min_val = minimum(arr)
        max_val = maximum(arr)
        range = max_val - min_val
        return range > 0 ? (arr .- min_val) ./ range : zeros(size(arr))
    end

    function robust_normalize(arr)
        q1 = Statistics.quantile(arr, 0.25)  # 25th percentile
        q3 = Statistics.quantile(arr, 0.75)  # 75th percentile
        iqr = q3 - q1              # Interquartile range
        return (arr .- Statistics.median(arr)) ./ iqr
    end

    function modified_z_score(arr)
        med = median(arr)                                  # Calculate the median of the array
        mad_value = median(abs.(arr .- med))              # Calculate the median absolute deviation (MAD)
        return mad_value > 0 ? (0.6745 * (arr .- med) ./ mad_value) : zeros(size(arr))  # Modified z-score normalization
    end
    norm_rmse = modified_z_score(errors_rmse)
    norm_rae = modified_z_score(errors_rae)
    norm_total_times = modified_z_score(total_times)
    norm_iterations = modified_z_score(iterations)

    # Scores
    alpha = 85
    beta = 5
    gamma = 10

    @assert alpha + beta + gamma == 100

    scores = []
    for i in 1:length(method_data)
        score =
            (alpha / 2) * (1 - norm_rmse[i]) +
            (alpha / 2) * (1 - norm_rae[i]) +
            beta * (1 - norm_total_times[i]) +
            gamma * (1 - norm_iterations[i])
        push!(scores, score)
    end

    df_method = DataFrame(;
        Centers=[d[1] for d in method_data],
        ShapeParameter=[d[2] for d in method_data],
        AdaptiveRefinement=[d[3] for d in method_data],
        Iterations=[d[4] for d in method_data],
        RMSEErrors=[d[5] for d in method_data],
        RAEErrors=[d[6] for d in method_data],
        Total=[d[7] for d in method_data],
        Scores=scores,
    )

    # Add a new column with the sum of RMSEErrors and RAEErrors
    df_method[!, :ErrorSum] = df_method.RMSEErrors .+ df_method.RAEErrors

    # Sort the DataFrame by the new column
    df_method = sort!(df_method, :Scores; rev=true)

    # Drop the ErrorSum column as it's no longer needed
    df_method = select!(df_method, Not(:ErrorSum))

    df_timing = DataFrame(;
        Centers=[d[1] for d in method_data],
        Centers_Time=[d[5] for d in timing_data],
        ShapeParameter=[d[2] for d in method_data],
        ShapeParameter_Time=[d[6] for d in timing_data],
        AdaptiveRefinement=[d[3] for d in method_data],
        AdaptiveRefinement_Time=[d[7] for d in timing_data],
        Iterations=[d[8] for d in timing_data],
        Total=[d[4] for d in timing_data],
    )

    min_method_indices = (argmin(df_method.RMSEErrors), argmin(df_method.RAEErrors))

    # Define highlighter functions for entire rows

    if !latex
        hl_heatmap = Highlighter(
            (data, i, j) -> j in [8],  # Apply to all cells
            (h, data, i, j) -> begin
                # Normalize the data values between 0 and 1 for each column
                min_val = minimum(data[:, j])
                max_val = maximum(data[:, j])
                normalized_val = (data[i, j] - min_val) / (max_val - min_val + eps())  # Avoid division by zero
                # Get the color from the colorscheme, here using 'coolwarm'
                color = get(colorschemes[:magma], normalized_val)
                return Crayon(;
                    foreground=(
                        round(Int, color.r * 255),
                        round(Int, color.g * 255),
                        round(Int, color.b * 255),
                    ),
                )
            end,
        )

        hl_heatmap_reverse = Highlighter(
            (data, i, j) -> j in [5, 6, 7],  # Apply to all cells
            (h, data, i, j) -> begin
                # Normalize the data values between 0 and 1 for each column
                min_val = minimum(data[:, j])
                max_val = maximum(data[:, j])
                normalized_val = (data[i, j] - min_val) / (max_val - min_val + eps())  # Avoid division by zero
                # Get the color from the colorscheme, here using 'coolwarm'
                color = get(reverse(colorschemes[:magma]), normalized_val)
                return Crayon(;
                    foreground=(
                        round(Int, color.r * 255),
                        round(Int, color.g * 255),
                        round(Int, color.b * 255),
                    ),
                )
            end,
        )

        hl_rmse = Highlighter(
            (data, i, j) -> (i == min_method_indices[1]), crayon"light_red"
        )

        hl_rae = Highlighter(
            (data, i, j) -> (i == min_method_indices[2]), crayon"light_blue"
        )

        hl_same = Highlighter(
            (data, i, j) -> (i == min_method_indices[1] && (i == min_method_indices[2])),
            crayon"fg:white bold bg:dark_gray",
        )
    else
        hl_rmse = LatexHighlighter((data, i, j) -> (i == min_method_indices[1]), ["textbf"])

        hl_rae = LatexHighlighter((data, i, j) -> (i == min_method_indices[2]), ["textbf"])
    end

    method_header = (
        [
            "Centers",
            "Shape Parameter",
            "Adaptive Refinement",
            "Iterations",
            "Error",
            "Error",
            "Time",
            "Score",
        ],
        ["", "", "", "", "RMSE", "RAE", "[s]", ""],
    )

    timing_header = (
        [
            "Centers",
            "Centers",
            "Shape Parameter",
            "Shape Parameter",
            "Adaptive Refinement",
            "Adaptive Refinement",
            "Adaptive Refinement",
            "Total",
        ],
        ["", "[s]", "", "[s]", "", "[s]", "Iterations", "[s]"],
    )

    if !latex
        pretty_table(
            df_method;
            formatters=ft_printf("%5.4f", 5:8),
            header=method_header,
            header_crayon=crayon"yellow bold",
            highlighters=(hl_same, hl_rmse, hl_rae, hl_heatmap, hl_heatmap_reverse),
            tf=tf_unicode_rounded,
            show_row_number=true,
            row_number_column_title="Rank",
            row_number_header_crayon=crayon"yellow bold",
        )
        # timing table
        pretty_table(
            df_timing;
            header=timing_header,
            header_crayon=crayon"yellow bold",
            tf=tf_unicode_rounded,
        )

    else
        pretty_table(
            df_method;
            formatters=ft_printf("%5.4f", 5:8),
            header=method_header,
            highlighters=(hl_rmse, hl_rae),
            backend=Val(:latex),
            show_row_number=true,
        )

        println("")

        pretty_table(df_timing; header=timing_header, backend=Val(:latex))
    end

    return nothing
end
# needs to be completly rewritten for refactorization
# make multiple for each simulation type
function printDetails(sys::System, sim::Simulation, method::Methods)
    println("==================================================")
    println("   Mode: $(nameof(typeof(method.simulation_method)))")
    println("--------------------------------------------------")
    println("   Parameters")
    println("..................................................")
    println("\tSamples: $(sim.samples)")
    println("\tVariation Tolerance: $(sim.variation_tolerance)")

    println("--------------------------------------------------")

    if typeof(method.simulation_method) == MonteCarloSimulation
        nothing # monte-carlo doesnt use these methods
    else
        println("   Methods:")
        println("..................................................")
        println("\tShape Parameter: $(nameof(typeof(method.shape_parameter_method)))")
        if typeof(method.shape_parameter_method) == IterativeAMLS ||
            typeof(method.shape_parameter_method) == DirectAMLS
            println("\t\tOrder: $(method.shape_parameter_method.order)")
        end

        println("\tBasis Function: $(nameof(typeof(method.basis_function_method)))")

        println("\tStarting Points: $(nameof(typeof(method.starting_points_method)))")
        println("\tCenter Points: $(nameof(typeof(method.centers_method)))")
        if typeof(method.centers_method) == Grid ||
            typeof(method.centers_method) == SparseGrid
            println("\t\tCenters Interval: $(method.centers_method.centers_interval)")
        elseif typeof(method.centers_method) == Greedy ||
            typeof(method.centers_method) == GeometricGreedy ||
            typeof(method.centers_method) == Leja
            println("\t\tNumber of Centers: $(method.centers_method.nCenters)")
        end
        println(
            "\tAdaptive Refinement: $(nameof(typeof(method.adaptive_refinement_method)))"
        )
        if typeof(method.adaptive_refinement_method) == SFCVT
            println(
                "\t\tOptimizer: $(nameof(typeof(method.adaptive_refinement_method.method)))"
            )
        end
        println("\tWeight Change: $(nameof(typeof(method.weight_change_method)))")
        println("\t\tWeight Change Tolerance: $(sim.weight_change_tolerance)")
        println("==================================================")
        println("")
    end

    return nothing
end

# ============================ PLOTS ===========================================
Plots.default(;
    framestyle=:box,
    label=nothing,
    grid=true,
    #legend=:bottomleft,
    #legend_font_halign=:left,
    size=(600, 450),
    titlefontsize=8,
    guidefontsize=8,
    legendfontsize=7,
    tickfontsize=8,
    left_margin=Plots.mm,
    bottom_margin=Plots.mm,
    fontfamily="Computer Modern",
    dpi=600,
)

function looped_variable(structs...)
    # Get the field names of the first struct
    fields = fieldnames(typeof(structs[1]))

    # Ensure all structs have the same fields
    for s in structs[2:end]
        if fieldnames(typeof(s)) != fields
            error("Structs have different fields")
        end
    end

    # Find the differing field
    differing_field = nothing
    for field in fields
        values = getfield.(structs, field)
        unique_values = unique(values)

        if length(unique_values) > 1
            if isnothing(differing_field)
                differing_field = (field, unique_values)
            else
                error("More than one differing field found")
            end
        end
    end

    if isnothing(differing_field)
        return false
    else
        return differing_field
    end
end

function extractError(
    errors::Union{Matrix{Dict{String,Float64}},Vector{Dict{String,Float64}}}
)
    # assumes all dicts have the same keys
    key_names = collect(Base.keys(errors[1]))

    extracted_errors = Vector{Tuple{String,Array{Float64}}}(undef, length(key_names))

    for (i, key) in enumerate(key_names)
        extracted_errors[i] = extractError(errors, key)
    end

    return extracted_errors
end

function extractError(errors::Vector{Dict{String,Float64}}, key::String)
    rows = length(errors)
    error_matrix = Array{Float64}(undef, rows)

    for i in 1:rows
        error_matrix[i] = get(errors[i], key, NaN)
    end

    return (key, error_matrix)
end

function extractError(errors::Matrix{Dict{String,Float64}}, key::String)
    rows, cols = size(errors)
    error_matrix = Array{Float64}(undef, rows, cols)

    for i in 1:rows
        for j in 1:cols
            error_matrix[i, j] = get(errors[i, j], key, NaN)
        end
    end

    return (key, error_matrix)
end

function removeDashAndCapitalize(str::String)
    return join([titlecase(word) for word in split(str, "-")], " ")
end

function removeUnderscoreAndCapitalize(str::String)
    return join([titlecase(word) for word in split(str, "_")], " ")
end

function latexSpaces(str::String)
    return replace(str, " " => "\\ ")
end

# ========================== ERROR PLOTS =======================================

function count_repetitions(v::Vector{String})
    occurrences = Dict{String,Int}()
    result = [occurrences[element] = get(occurrences, element, 0) + 1 for element in v]
    return result
end

function looped_variable(structs...)
    # Get the field names of the first struct
    fields = fieldnames(typeof(structs[1]))

    # Ensure all structs have the same fields
    for s in structs[2:end]
        if fieldnames(typeof(s)) != fields
            error("Structs have different fields")
        end
    end

    # Find the differing field
    differing_field = nothing
    for field in fields
        values = getfield.(structs, field)
        unique_values = unique(values)

        if length(unique_values) > 1
            if isnothing(differing_field)
                differing_field = (field, unique_values)
            else
                error("More than one differing field found")
            end
        end
    end

    if isnothing(differing_field)
        return false
    else
        return differing_field
    end
end

function extractError(
    errors::Union{Matrix{Dict{String,Float64}},Vector{Dict{String,Float64}}}
)
    # assumes all dicts have the same keys
    key_names = collect(Base.keys(errors[1]))

    extracted_errors = Vector{Tuple{String,Array{Float64}}}(undef, length(key_names))

    for (i, key) in enumerate(key_names)
        extracted_errors[i] = extractError(errors, key)
    end

    return extracted_errors
end

function extractError(errors::Vector{Dict{String,Float64}}, key::String)
    rows = length(errors)
    error_matrix = Array{Float64}(undef, rows)

    for i in 1:rows
        error_matrix[i] = get(errors[i], key, NaN)
    end

    return (key, error_matrix)
end

function extractError(errors::Matrix{Dict{String,Float64}}, key::String)
    rows, cols = size(errors)
    error_matrix = Array{Float64}(undef, rows, cols)

    for i in 1:rows
        for j in 1:cols
            error_matrix[i, j] = get(errors[i, j], key, NaN)
        end
    end

    return (key, error_matrix)
end

function removeDashAndCapitalize(str::String)
    return join([titlecase(word) for word in split(str, "-")], " ")
end

function removeUnderscoreAndCapitalize(str::String)
    return join([titlecase(word) for word in split(str, "_")], " ")
end

function latexSpaces(str::String)
    return replace(str, " " => "\\ ")
end

# ========================== ERROR PLOTS =======================================

function count_repetitions(v::Vector{String})
    occurrences = Dict{String,Int}()
    result = [occurrences[element] = get(occurrences, element, 0) + 1 for element in v]
    return result
end

function plotError(
    axis::Makie.Axis,
    error::Vector{Float64},
    looped_variable::Vector{<:Number};
    labels::Union{String,Nothing}=nothing,
)
    return CairoMakie.lines!(
        axis,
        looped_variable,
        error;
        label=isnothing(labels) ? nothing : LaTeXStrings.latexstring(labels),
    )
end

function plotError(
    axis::Makie.Axis,
    error::Matrix{Float64},
    looped_variable::Vector{<:Number};
    labels::Union{Vector{String},Nothing}=nothing,
)
    if !isnothing(labels)
        repetitons = count_repetitions(labels)
    end

    for (i, row::Vector) in enumerate(eachrow(error))
        if repetitons[i] > 1
            plotError(
                axis,
                row,
                looped_variable;
                labels=isnothing(labels) ? nothing : "$(labels[i]) - $(repetitons[i])",
            )
        else
            plotError(
                axis, row, looped_variable; labels=isnothing(labels) ? nothing : labels[i]
            )
        end
    end
end

function plotError(
    signatures::Union{Model,Array{Model}}, errors::Array{Dict{String,Float64}}
)
    sim_variable = looped_variable(getfield.(signatures, :sim)...)
    method_variable = looped_variable(getfield.(signatures, :method)...)

    extracted_errors = extractError(errors)

    plt = Figure(;
        size=(600, length(extracted_errors) * 200), fonts=(; regular="CMU Serif")
    )
    axes = Axis[]
    for (i, (key, error)) in enumerate(extracted_errors)
        if extracted_errors[2][end] != error
            ax = Axis(
                plt[i, 1];
                title=LaTeXStrings.latexstring(key),
                xgridstyle=:dash,
                ygridstyle=:dash,
                xtickalign=1,
                xticksize=5,
                ytickalign=1,
                yticksize=5,
                xscale=log10,
                xminorticksvisible=true,
                yscale=log10,
                yminorticksvisible=true,
            )
        else
            ax = Axis(
                plt[i, 1];
                title=LaTeXStrings.latexstring(key),
                xlabel=LaTeXStrings.latexstring(uppercase(string((sim_variable[1])))),
                xgridstyle=:dash,
                ygridstyle=:dash,
                xtickalign=1,
                xticksize=5,
                ytickalign=1,
                yticksize=5,
                xscale=log10,
                xminorticksvisible=true,
                yscale=log10,
                yminorticksvisible=true,
            )
        end

        push!(axes, ax)

        if typeof(method_variable) == Bool
            plotError(ax, error, sim_variable[2])
        else
            plotError(
                ax,
                error,
                sim_variable[2];
                labels=string.(nameof.(typeof.(method_variable[2]))),
            )
        end
    end

    if typeof(method_variable) != Bool
        Legend(plt[1:length(axes), 2], axes[1]; unique=true, merge=true)
    end
    return plt
end

# ============================ TIMING PLOTS ====================================

function plotTime(
    axis::Makie.Axis,
    time::Vector{Float64},
    looped_variable::Vector{<:Number};
    labels::Union{String,Nothing}=nothing,
)
    return CairoMakie.lines!(
        axis,
        looped_variable,
        time;
        label=isnothing(labels) ? nothing : LaTeXStrings.latexstring(labels),
    )
end

function plotTime(
    axis::Makie.Axis,
    time::Matrix{Float64},
    looped_variable::Vector{<:Number};
    labels::Union{Vector{String},Nothing}=nothing,
)
    if !isnothing(labels)
        repetitons = count_repetitions(labels)
    end

    for (i, row::Vector) in enumerate(eachrow(time))
        if repetitons[i] > 1
            plotTime(
                axis,
                row,
                looped_variable;
                labels=isnothing(labels) ? nothing : "$(labels[i]) - $(repetitons[i])",
            )
        else
            plotTime(
                axis, row, looped_variable; labels=isnothing(labels) ? nothing : labels[i]
            )
        end
    end
end

function plotTime(
    signatures::Union{Model,Array{Model}}, times::Array{Float64}; title::String="Times"
)
    sim_variable = looped_variable(getfield.(signatures, :sim)...)
    method_variable = looped_variable(getfield.(signatures, :method)...)

    plt = Figure(; size=(600, 200), fonts=(; regular="CMU Serif"))

    ax = Axis(
        plt[1, 1];
        xlabel=LaTeXStrings.latexstring(uppercase(string((sim_variable[1])))),
        ylabel=L"Time\ (s)",
        xgridstyle=:dash,
        ygridstyle=:dash,
        xtickalign=1,
        xticksize=5,
        ytickalign=1,
        yticksize=5,
        xscale=log10,
        xminorticksvisible=true,
        yminorticksvisible=true,
    )

    if typeof(method_variable) == Bool
        plotTime(ax, times, sim_variable[2])
    else
        plotTime(
            ax, times, sim_variable[2]; labels=string.(nameof.(typeof.(method_variable[2])))
        )
    end
    if typeof(method_variable) == Bool
    else
        Legend(plt[1, 2], ax; unique=true, merge=true)
    end

    return plt
end

function plotCentersSubPlot(signatures::Union{Model,Array{Model}})
    # Create the figure
    fig = Figure(; fonts=(; regular="CMU Serif"))

    # Create the axes
    ax1 = Axis(fig[1, 1]; title="RMSE", xlabel="Number of Centers", ylabel="RMSE")

    # Replace ax2 with a stacked area plot for times
    ax2 = Axis(
        fig[1, 2]; title="Stacked Times", xlabel="Number of Centers", ylabel="Time (s)"
    )

    ax3 = Axis(
        fig[2, 1];
        title="Adaptive Refinement",
        xlabel="Number of Centers",
        ylabel="Time (s)",
    )
    ax4 = Axis(fig[2, 1]; yaxisposition=:right, ylabel="Iterations", yticksvisible=false)

    # Loop through the signatures for relevant information
    errors = Vector{Float64}(undef, length(signatures))
    centers_time = Vector{Float64}(undef, length(signatures))
    shape_parameter_time = Vector{Float64}(undef, length(signatures))
    adaptive_refinement_time = Vector{Float64}(undef, length(signatures))
    evaluation_time = Vector{Float64}(undef, length(signatures))

    num_centers = Vector{Int}(undef, length(signatures))
    adaptive_refinement_iterations = Vector{Int}(undef, length(signatures))

    for (i, signature) in enumerate(signatures)
        errors[i] = get(signature.errors, "RMSE", NaN)
        centers_time[i] = signature.metrics.time.centers
        shape_parameter_time[i] = signature.metrics.time.shape_parameter
        adaptive_refinement_time[i] = signature.metrics.time.adaptive_refinement
        adaptive_refinement_iterations[i] = signature.metrics.iterations
        evaluation_time[i] = signature.metrics.time.evaluation
        num_centers[i] = size(signature.model.centers, 2)
    end

    # Compute the stacked times
    cumulative_centers = centers_time
    cumulative_shape_parameter = cumulative_centers .+ shape_parameter_time
    cumulative_adaptive_refinement = cumulative_shape_parameter .+ adaptive_refinement_time
    cumulative_evaluation = cumulative_adaptive_refinement .+ evaluation_time

    # Plot the data for RMSE
    lines!(ax1, num_centers, errors)

    # Plot the stacked times on ax2 using bands
    band!(ax2, num_centers, zeros(length(num_centers)), cumulative_centers)
    band!(ax2, num_centers, cumulative_centers, cumulative_shape_parameter)
    band!(ax2, num_centers, cumulative_shape_parameter, cumulative_adaptive_refinement)
    band!(ax2, num_centers, cumulative_adaptive_refinement, cumulative_evaluation)

    # Plot the dual-axis data for adaptive refinement
    lines!(ax3, num_centers, adaptive_refinement_time)
    lines!(ax4, num_centers, adaptive_refinement_iterations)

    return fig
end
function plotSurrogateComparison(times...)
    labels = [L"RBF", L"Kriging"]
    colors = [colorant"#557ec1", colorant"#ac2c71"]

    plt = Figure(; size=(600, 200), fonts=(; regular="CMU Serif"))
    ax = Axis(
        plt[1, 1];
        xgridstyle=:dash,
        xlabel=L"ITERATIONS",
        ylabel=L"Time\ (s)",
        ygridstyle=:dash,
        xtickalign=1,
        xticksize=5,
        ytickalign=1,
        yticksize=5,
        #xscale=log10,
        xminorticksvisible=true,
        yscale=log10,
        yminorticksvisible=true,
    )
    for (i, (time, color)) in enumerate(zip(times, colors))
        lines!(ax, time; label=labels[i], color=color, linewidth=1.5)
    end
    Legend(plt[1, 2], ax; unique=true, merge=true)
    return plt
end

# ==============================================================================
end