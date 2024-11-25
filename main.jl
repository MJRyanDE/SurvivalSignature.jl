using SurvivalSignature
using ProgressMeter
using Statistics
using JLD2              # needed for load
using Plots, CairoMakie
# ==============================================================================

if !isdefined(Main, :Import)
    include("./Modules/Import.jl")
end

using Revise
Revise.track("./Modules/Import.jl") # only recompiles changed files.

using .Import

using ..SurvivalSignatureUtils
using ..Simulate
using ..Structures
using ..Structures: System, Simulation, Model, Methods
using ..Structures: SystemMethod, GridSystem
using ..Structures: SimulationType, MonteCarloSimulation, IntervalPredictorSimulation
using ..Structures: BasisFunctionMethod, Gaussian
using ..Structures: ShapeParameterMethod, Hardy, Rippa, DirectAMLS, IterativeAMLS
using ..Structures: CentersMethod, Grid, SparseGrid, Greedy, GeometricGreedy, Leja
using ..Structures: StartingMethod, GridStart
using ..Structures: ErrorType, RMSE, RAE, NORM, NRMSE
using ..Structures: AdaptiveRefinementMethod, None, TEAD, MIPT, SFCVT, MEPE, EI, EIGF
using ..Structures:
    SFCVT_Method, EnumerationX, BlackBoxX, OptimX, SimulatedAnnealingX, EvolutionX
using ..StructureCompilation
using ..Error: calculateError
using ..Systems
using ..Visualization: plotError

function main()
    # ================================ INPUTS ==================================
    warmup::Bool = true         # used to precompile the code for more comparable timings. hopefully.
    verbose::Bool = true        # used to turn on and off print statements during 'simulate'
    timing::Bool = true         # used to turn on and off timing of the simulation

    plot_sample_space::Bool = true
    plot_results::Bool = true
    save::Bool = true      # saves the plots.

    # [ GridSystem() ]
    percolation::Bool = true
    system_type::SystemMethod = GridSystem((10, 10))

    # Simulation Parameters
    samples::Union{Int,Vector{Int}} = 100                        # number of samples
    covtol::Float64 = 1e-3                                       # coefficient of variation tolerance
    wtol::Float64 = 1e-3                                         # weight change tolerance

    ci::Vector{Int} = [10, 10]                                   # centers interval - dims must match 
    #                                                            # number of types
    loops = 1

    # =========================== METHODS ======================================
    # [MonteCarloSimulation(), IntervalPredictorSimulation()]
    simulation_method::SimulationType = IntervalPredictorSimulation()
    # [ GridStart() ]
    starting_points_method::Union{Vector{<:StartingMethod},StartingMethod} = GridStart()
    # [Grid(), SparseGrid(), Greedy(), GeometricGreedy(), Leja()]
    centers_method::Union{Vector{<:CentersMethod},CentersMethod} = [
        Grid(ci, true), Grid(ci, false)
    ]

    # [ NORM(), NRMSE() ]
    weight_change_method::Union{Vector{<:ErrorType},ErrorType} = NORM()
    # [ Hardy(), Rippa(), DirectAMLS(), IterativeAMLS() ] # can be a Vector
    shape_parameter_method::Union{Vector{<:ShapeParameterMethod},ShapeParameterMethod} = Rippa()
    # [ Gaussian() ] 
    basis_function_method::Union{Vector{<:BasisFunctionMethod},BasisFunctionMethod} = Gaussian()

    # [ None(), MIPT(), SFCVT(), TEAD(), EI(), EIGF() ]
    adaptive_refinement_method::Union{Vector{<:AdaptiveRefinementMethod},AdaptiveRefinementMethod} = None()

    # Error 
    # [RMSE(), RAE()]
    error_type::Union{Vector{ErrorType},ErrorType} = [RMSE(), RAE()]

    # ======================== STRUCT REFINEMENT ===============================

    sys::System = Systems.generateSystem(system_type; percolation_bool=percolation)

    sims::Union{Vector{Simulation},Simulation} = StructureCompilation.compileSimulation(
        samples, covtol, wtol
    )

    if verbose
        println("Compiling Methods....")
    end
    methods::Union{Vector{Methods},Methods} = StructureCompilation.compileMethods(
        simulation_method,
        starting_points_method,
        centers_method,
        weight_change_method,
        shape_parameter_method,
        basis_function_method,
        adaptive_refinement_method,
    )

    if verbose
        println("\tNumber of Methods: ", length(methods))
        println("Methods Compiled.")
    end
    # =============================== WARMUP =================================
    # just for warmup_purposes, such that the first method timing isnt effected 
    # by the JIT compilation. 
    if warmup
        if verbose
            println("Performing Warmup Simulation...")
        end
        warmup_sim = Simulation(1, 1.0, 1.0)
        Simulate.simulate(methods, sys, warmup_sim; verbose=false)
        if verbose
            println("Warmup Complete.")
        end
    end

    # =============================== SIMULATE =================================

    errors_vector = Union{Vector{Dict{String,Float64}},Matrix{Dict{String,Float64}}}[]
    total_times_vector = []
    centers_times_vector = []
    shape_parameter_times_vector = []
    adaptive_refinement_times_vector = []
    iterations_vector = []
    iteration_times_vector = []
    shape_parameters_vector = []

    signatures::Union{Matrix{Model},Vector{Model},Model,Nothing} = nothing

    for i in 1:loops
        signatures = Simulate.simulate(
            methods,
            sys,
            sims;
            verbose=verbose,
            save=save,
            timing=timing,
            plot_sample_space=plot_sample_space,
        )

        # ============================= "TRUE SOLUTION" ============================
        println("Loop: ", i, "/", loops)
        if system_type.dims != (15, 15)
            sim_mc = Structures.Simulation(250, covtol, wtol)
            method_mc = Structures.Methods(
                MonteCarloSimulation(), nothing, nothing, nothing, nothing, nothing, nothing
            )
            signature_mc = Simulate.simulate(method_mc, sys, sim_mc; verbose=verbose)

        else
            #true values - apparently this outputs a Φ - which is the true value solutions
            @load "demo/data/grid-network-15x15-MC-10000.jld2"
        end
        # =============================== ERROR ====================================
        println("Calculating Error...")

        if system_type.dims != (15, 15)
            signatures, errors = Error.calculateError(
                error_type, signatures, signature_mc.Phi.solution; verbose=false
            )
        else
            signatures, errors = Error.calculateError(
                error_type, signatures, Φ; verbose=false
            )
        end

        push!(errors_vector, errors)

        #SurvivalSignatureUtils._print(errors)

        println("Errors Calculated.\n")

        # ============================== Timings ===================================
        if timing
            #println("Total Times (s):")

            metrics = Array{Metrics}(undef, size(signatures))
            for (i, signature) in enumerate(signatures)
                metrics[i] = signature.metrics
            end

            total_times = Array{Float64}(undef, size(metrics))
            for (i, metric) in enumerate(metrics)
                total_times[i] = metric.time.total
            end
            #SurvivalSignatureUtils._print(total_times; digits=6)
            push!(total_times_vector, total_times)
            #println("")

            #println("Center Times (s):")
            center_times = Array{Float64}(undef, size(metrics))
            for (i, metric) in enumerate(metrics)
                center_times[i] = metric.time.centers
            end
            #SurvivalSignatureUtils._print(center_times; digits=6)
            push!(centers_times_vector, center_times)

            #println("")
            #println("Shape Parameter Times (s):")
            shape_parameter_times = Array{Float64}(undef, size(metrics))
            for (i, metric) in enumerate(metrics)
                shape_parameter_times[i] = metric.time.shape_parameter
            end
            #SurvivalSignatureUtils._print(shape_parameter_times; digits=6)
            push!(shape_parameter_times_vector, shape_parameter_times)

            #println(" ")

            #println("Adaptive Refinement Times (s):")
            adaptive_refinement_times = Array{Float64}(undef, size(metrics))
            for (i, metric) in enumerate(metrics)
                adaptive_refinement_times[i] = metric.time.adaptive_refinement
            end
            #SurvivalSignatureUtils._print(adaptive_refinement_times; digits=6)
            push!(adaptive_refinement_times_vector, adaptive_refinement_times)
            #println("")
            # =========================== Iterations ===================================
            #println("Iterations:")
            iterations = Array{Int}(undef, size(metrics))
            for (i, metric) in enumerate(metrics)
                iterations[i] = metric.iterations
            end
            #SurvivalSignatureUtils._print(iterations)
            push!(iterations_vector, iterations)
            #println("")

            #println("Time per Iteration (s):")
            iteration_times = adaptive_refinement_times ./ iterations

            #SurvivalSignatureUtils._print(iteration_times; digits=6)
            push!(iteration_times_vector, iteration_times)
            #println(" ")
        end
        # =========== Shape Parameters =============================================
        #println("Shape Parameters:")

        shape_parameters = Array{Float64}(undef, size(metrics))
        for (i, signature) in enumerate(signatures)
            shape_parameters[i] = signature.model.shape_parameter
        end

        #SurvivalSignatureUtils._print(shape_parameters; digits=6)
        push!(shape_parameters_vector, shape_parameters)
        #println(" ")
    end

    # ============================ AVERAGES =====================================

    errors = SurvivalSignatureUtils.average_dictionaries(errors_vector)

    total_times = Statistics.mean(total_times_vector)
    center_times = Statistics.mean(centers_times_vector)
    shape_parameter_times = Statistics.mean(shape_parameter_times_vector)
    adaptive_refinement_times = Statistics.mean(adaptive_refinement_times_vector)
    iterations = Statistics.mean(iterations_vector)
    iteration_times = Statistics.mean(iteration_times_vector)
    shape_parameters = Statistics.mean(shape_parameters_vector)

    # ============================ PRINTS ========================================

    SurvivalSignatureUtils._print(errors)
    println("")
    if timing
        println("Total Times (s):")
        SurvivalSignatureUtils._print(total_times; digits=6)
        println("")
        println("Center Times (s):")
        SurvivalSignatureUtils._print(center_times; digits=6)
        println("")
        println("Shape Parameter Times (s):")
        SurvivalSignatureUtils._print(shape_parameter_times; digits=6)
        println("")
        println("Adaptive Refinement Times (s):")
        SurvivalSignatureUtils._print(adaptive_refinement_times; digits=6)
        println("")
        println("Iterations:")
        SurvivalSignatureUtils._print(iterations)
        println("")
        println("Time per Iteration (s):")
        SurvivalSignatureUtils._print(iteration_times; digits=6)
        println("")
    end
    println("Shape Parameters:")
    SurvivalSignatureUtils._print(shape_parameters; digits=6)
    println("")
    # ============================ Plot ========================================

    if plot_results
        plt = Visualization.plotError(signatures, errors)
        if save
            title = "error_plot.pdf"
            file_path = joinpath("figures", "errors", title)
            CairoMakie.save(file_path, plt)
        end
        CairoMakie.display(plt)

        plt = Visualization.plotTime(signatures, total_times; title="Total Time")
        if save
            title = "total_time_plot.pdf"
            file_path = joinpath("figures", "times", title)
            CairoMakie.save(file_path, plt)
        end
        CairoMakie.display(plt)

        plt = Visualization.plotTime(signatures, center_times; title="Centers Time")
        if save
            title = "center_time_plot.pdf"
            file_path = joinpath("figures", "times", title)
            CairoMakie.save(file_path, plt)
        end
        CairoMakie.display(plt)

        plt = Visualization.plotTime(
            signatures, shape_parameter_times; title="Shape Parameter Time"
        )
        if save
            title = "shape_parameter_time_plot.pdf"
            file_path = joinpath("figures", "times", title)
            CairoMakie.save(file_path, plt)
        end
        CairoMakie.display(plt)

        plt = Visualization.plotTime(
            signatures, adaptive_refinement_times; title="Adaptive Refinement Time"
        )
        if save
            title = "adaptive_refinement_time_plot.pdf"
            file_path = joinpath("figures", "times", title)
            CairoMakie.save(file_path, plt)
        end
        CairoMakie.display(plt)

        plt = Visualization.plotTime(
            signatures, iteration_times; title="Time per Iteration"
        )
        if save
            title = "time_per_iteration_plot.pdf"
            file_path = joinpath("figures", "times", title)
            CairoMakie.save(file_path, plt)
        end
        CairoMakie.display(plt)
    end
    return nothing
end
# =============================== RUN ==========================================
main()
