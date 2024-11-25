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
using ..Structures: CentersMethod, Grid, SparseGrid, Greedy, GeometricGreedy, Leja
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
    samples::Union{Int,Vector{Int}} = 100        # number of samples
    covtol::Float64 = 1e-3                       # coefficient of variation tolerance
    wtol::Float64 = 1e-3                         # weight change tolerance

    ci::Vector{Int} = [15, 15]                   # centers interval - dims must match 
    #                                            # number of types

    # =========================== METHODS ======================================
    # [MonteCarloSimulation(), IntervalPredictorSimulation()]
    simulation_method::SimulationType = IntervalPredictorSimulation()
    # [ GridStart() ]
    starting_points_method::StartingMethod = GridStart()
    # [ GridCenters(), Greedy(), GeometricGreedy(), Leja() ]
    centers_method = Grid # just the name
    centers_spread = 5:1:50

    centers_methods = Vector{CentersMethod}(undef, length(centers_spread))

    if centers_method == Grid
        for (i, spread) in enumerate(centers_spread)
            ci = [spread, spread]
            centers_methods[i] = Grid(ci)
        end
    elseif centers_method == SparseGrid
        for (i, spread) in enumerate(centers_spread)
            ci = [spread, spread]
            centers_methods[i] = SparseGrid(ci)
        end
    elseif centers_method == GeometricGreedy
        for (i, spread) in enumerate(centers_spread)
            ci = [spread, spread]
            centers_methods[i] = GeometricGreedy(prod(ci))
        end
    elseif centers_method == Leja
        for (i, spread) in enumerate(centers_spread)
            ci = [spread, spread]
            centers_methods[i] = Leja(prod(ci))
        end
    end

    # [ NORM(), NRMSE() ]
    weight_change_method::ErrorType = NORM()
    # [ Hardy(), Rippa(), DirectAMLS(), IndirectAMLS() ] # can be a Vector
    shape_parameter_method::ShapeParameterMethod = Rippa()

    # [ Gaussian() ] 
    basis_function_method::BasisFunctionMethod = Gaussian()
    # [ MIPT(), SFCVT(), TEAD(), EI(), MEPE() ]
    adaptive_refinement_method::AdaptiveRefinementMethod = TEAD()

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
        centers_methods,
        weight_change_method,
        shape_parameter_method,
        basis_function_method,
        adaptive_refinement_method,
    )

    # =============================== SIMULATE =================================
    # just for warmup_purposes, such that the first method timing isnt effected 
    # by the JIT compilation. this doesnt seem to work correctly, however. 

    warmup_sim = Simulation(1, 1.0, 1.0)
    Simulate.simulate(methods[1], sys, warmup_sim; verbose=false)

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

    # ============================ Plot ========================================

    if plot_results && timing
        plt = Visualization.plotCentersSubPlot(signatures)
        display(plt)
    end

    return nothing
end
# =============================== RUN ==========================================
main()
