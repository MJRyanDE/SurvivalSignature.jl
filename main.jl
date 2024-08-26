using SurvivalSignature
using ProgressMeter
using JLD2              # needed for load
# ==============================================================================

include("Modules/Import.jl")
using .Import

using ..SurvivalSignatureUtils

using ..Structures
using ..Structures: System, Simulation, Model, Methods
using ..Structures: SystemMethod, GridSystem
using ..Structures: SimulationType, MonteCarloSimulation, IntervalPredictorSimulation
using ..Structures: BasisFunctionMethod, Gaussian
using ..Structures: ShapeParameterMethod, Hardy, Rippa, DirectAMLS, IndirectAMLS
using ..Structures: CentersMethod, GridCenters, Greedy, GeometricGreedy, Leja
using ..Structures: StartingMethod, GridStart
using ..Structures: ErrorType, RMSE, RAE, NORM, NRMSE
using ..Structures: AdaptiveRefinementMethod, TEAD, MIPT, SFCVT, MEPE, EI
using ..Structures:
    SFCVT_Method, EnumerationX, BlackBoxX, OptimX, SimulatedAnnealingX, EvolutionX
using ..StructureCompilation
using ..Error: calculateError
using ..Systems
using ..Visualization: plotError

function main()
    # ================================ INPUTS ==================================
    percolation::Bool = true
    verbose::Bool = true        # used to turn on and off print statements during 'simulate'
    timing::Bool = true         # used to turn on and off timing of the simulation

    plot_sample_space::Bool = false
    plot_results::Bool = true

    # [ GridSystem() ]
    system_type::SystemMethod = GridSystem((15, 15))

    # Simulation Parameters
    samples::Union{Int,Vector{Int}} = [100, 500, 1000]        # number of samples
    covtol::Float64 = 1e-3                       # coeficient of varriation tolerance
    wtol::Float64 = 1e-3                         # weight change tolerance

    ci::Vector{Int} = [15, 15]                   # centers interval - dims must match 
    #                                            # number of types

    # =========================== METHODS ======================================
    # [MonteCarloSimulation(), IntervalPredictorSimulation()]
    simulation_method::SimulationType = IntervalPredictorSimulation()
    # [ GridStart() ]
    starting_points_method::StartingMethod = GridStart()
    # [ GridCenters(), Greedy(), GeometricGreedy(), Leja() ]
    centers_method::Union{Vector{CentersMethod},CentersMethod} = [Leja(), GridCenters(ci)]

    # [ NORM(), NRMSE() ]
    weight_change_method::Union{Vector{ErrorType},ErrorType} = NORM()
    # [ Hardy(), Rippa(), DirectAMLS(), IndirectAMLS() ] # can be a Vector
    shape_parameter_method::Union{Vector{ShapeParameterMethod},ShapeParameterMethod} = Rippa()

    # [ Gaussian() ] 
    basis_function_method::Union{Vector{BasisFunctionMethod},BasisFunctionMethod} = Gaussian()
    # [ MIPT(), SFCVT(), TEAD() ]
    adaptive_refinement_method::Union{Vector{<:AdaptiveRefinementMethod},AdaptiveRefinementMethod} = TEAD()

    # Error
    error_type::Union{Vector{ErrorType},ErrorType} = [RMSE(), RAE()]

    # ======================== STRUCT REFINEMENT ===============================

    sys::System = Systems.generateSystem(system_type; percolation_bool=percolation)

    sims::Union{Vector{Simulation},Simulation} = StructureCompilation.compileSimulation(
        samples, covtol, wtol
    )

    methods::Union{Vector{Methods},Methods} = StructureCompilation.compileMethods(
        simulation_method,
        starting_points_method,
        centers_method,
        weight_change_method,
        shape_parameter_method,
        basis_function_method,
        adaptive_refinement_method,
    )

    # =============================== SIMULATE =================================
    # just for warmup_purposes, such that the first method timing isnt effected 
    # by the JIT compilation. this doesnt seem to work correctly, however. 

    # warmup_sim = Simulation(1, 1.0, 1.0)
    # Simulate.simulate(methods[1], sys, warmup_sim; verbose=false)

    signatures::Union{Matrix{Model},Vector{Model},Model} = Simulate.simulate(
        methods,
        sys,
        sims;
        verbose=verbose,
        timing=timing,
        plot_sample_space=plot_sample_space,
    )

    # ============================= "TRUE SOLUTION" ============================
    if system_type.dims != (15, 15)
        sim_mc = Structures.Simulation(500, covtol, wtol)
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
        signatures, errors = Error.calculateError(error_type, signatures, Φ; verbose=false)
    end

    SurvivalSignatureUtils._print(errors)

    println("Errors Calculated.\n")

    # ============================== Timings ===================================
    if timing
        println("Total Times (s):")

        metrics = Array{Metrics}(undef, size(signatures))
        for (i, signature) in enumerate(signatures)
            metrics[i] = signature.metrics
        end

        total_times = Array{Float64}(undef, size(metrics))
        for (i, metric) in enumerate(metrics)
            total_times[i] = metric.time.total
        end
        SurvivalSignatureUtils._print(total_times; digits=6)
        println("")

        println("Center Times (s):")
        center_times = Array{Float64}(undef, size(metrics))
        for (i, metric) in enumerate(metrics)
            center_times[i] = metric.time.centers
        end
        SurvivalSignatureUtils._print(center_times; digits=6)

        println("")
        println("Shape Parameter Times (s):")
        shape_parameter_times = Array{Float64}(undef, size(metrics))
        for (i, metric) in enumerate(metrics)
            shape_parameter_times[i] = metric.time.shape_parameter
        end
        SurvivalSignatureUtils._print(shape_parameter_times; digits=6)
        println(" ")

        println("Adaptive Refinement Times (s):")
        adaptive_refinement_times = Array{Float64}(undef, size(metrics))
        for (i, metric) in enumerate(metrics)
            adaptive_refinement_times[i] = metric.time.adaptive_refinement
        end
        SurvivalSignatureUtils._print(adaptive_refinement_times; digits=6)
        # =========================== Iterations ===================================
        println("Iterations:")
        iterations = Array{Int}(undef, size(metrics))
        for (i, metric) in enumerate(metrics)
            iterations[i] = metric.iterations
        end
        SurvivalSignatureUtils._print(iterations)
        println("")

        println("Time per Iteration (s):")
        iteration_times = adaptive_refinement_times ./ iterations

        SurvivalSignatureUtils._print(iteration_times; digits=6)
        println(" ")
    end
    # =========== Shape Parameters =============================================
    println("Shape Parameters:")

    shape_parameters = Array{Float64}(undef, size(metrics))
    for (i, signature) in enumerate(signatures)
        shape_parameters[i] = signature.model.shape_parameter
    end

    SurvivalSignatureUtils._print(shape_parameters; digits=6)
    println(" ")

    # ============================ Plot ========================================

    if plot_results
        plt = Visualization.plotError(signatures, errors)
        display(plt)

        plt = Visualization.plotTime(signatures, total_times; title="Total Time")
        display(plt)

        plt = Visualization.plotTime(signatures, center_times; title="Centers Time")
        display(plt)

        plt = Visualization.plotTime(
            signatures, shape_parameter_times; title="Shape Parameter Time"
        )
        display(plt)

        plt = Visualization.plotTime(
            signatures, adaptive_refinement_times; title="Adaptive Refinement Time"
        )
        display(plt)
    end

    return nothing
end
# =============================== RUN ==========================================
main()
