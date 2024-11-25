using SurvivalSignature
using ProgressMeter
using JLD2              # needed for save and load
# ==============================================================================

println(pwd())

if !isdefined(Main, :Import)
    include("../Modules/Import.jl")
end

using Revise
Revise.track("Modules/Import.jl") # only recompiles changed files.

using .Import

using ..SurvivalSignatureUtils

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

    plot_sample_space::Bool = false
    save::Bool = false

    # [ GridSystem() ]
    percolation::Bool = true
    system_type::SystemMethod = GridSystem((10, 10))

    # Simulation Parameters
    samples::Int = 5000                            # number of samples
    covtol::Float64 = 1e-3                       # coefficient of variation tolerance
    wtol::Float64 = 1e-3                         # weight change tolerance

    ci::Vector{Int} = [15, 15]                   # centers interval - dims must match 
    #                                            # number of types

    max_adaptive_iterations::Int = 250

    # =========================== METHODS ======================================
    # [MonteCarloSimulation(), IntervalPredictorSimulation()]
    simulation_method::SimulationType = IntervalPredictorSimulation()
    # [ GridStart() ]
    starting_points_method::Union{Vector{<:StartingMethod},StartingMethod} = GridStart()
    # [ Grid(), SparseGrid(), Greedy(), GeometricGreedy(), Leja() ]
    centers_method::Union{Vector{<:CentersMethod},CentersMethod} = [
        Grid(ci), SparseGrid(ci), GeometricGreedy(prod(ci)), Leja(prod(ci))
    ]

    # [ NORM(), NRMSE() ]
    weight_change_method::Union{Vector{<:ErrorType},ErrorType} = NORM()
    # [ Hardy(), Rippa(), DirectAMLS(), IterativeAMLS() ] # can be a Vector
    shape_parameter_method::Union{Vector{<:ShapeParameterMethod},ShapeParameterMethod} = [
        Rippa(), DirectAMLS(), IterativeAMLS()
    ]

    # [ Gaussian() ] 
    basis_function_method::Union{Vector{<:BasisFunctionMethod},BasisFunctionMethod} = Gaussian()
    # [ None(), MIPT(), SFCVT(), TEAD(), EI(), EIGT() ]
    adaptive_refinement_method::Union{Vector{<:AdaptiveRefinementMethod},AdaptiveRefinementMethod} = [
        EIGF(max_adaptive_iterations),
        MEPE(max_adaptive_iterations),
        MIPT(0.25, 10e-4, max_adaptive_iterations),
        TEAD(false, max_adaptive_iterations),
    ]

    # Error 
    # [RMSE(), RAE(), SMAPE()]
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
        println("Methods Compiled.\n")
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
            println("Warmup Complete.\n")
        end
    end

    # =============================== SIMULATE =================================
    signatures::Union{Matrix{Model},Vector{Model},Model} = Simulate.simulate(
        methods,
        sys,
        sims;
        verbose=verbose,
        save=save,
        timing=timing,
        plot_sample_space=plot_sample_space,
    )

    # ============================= "TRUE SOLUTION" ============================
    if system_type.dims != (15, 15)
        sim_mc = Structures.Simulation(5000, covtol, wtol)
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

    println("Errors Calculated.\n")

    # ============================== SAVE ======================================

    save_object("ultimate_signatures_10x10_$(samples).JLD2", signatures)

    # ============================== Table =====================================

    Visualization.printUltimateTest(signatures)

    # ============================== Timings ===================================
    return nothing
end
# =============================== RUN ==========================================
main()
