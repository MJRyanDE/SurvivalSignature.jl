module Structures

__precompile__()

# ==============================================================================

using ..SurvivalSignatureUtils

# ==============================================================================
export System, Simulation, Methods

export SimulationModel, MonteCarloModel, PredictorModel, Model
export SimulationType, MonteCarloSimulation, IntervalPredictorSimulation

export BasisFunctionMethod, Gaussian
export ShapeParameterMethod, Hardy, Rippa, DirectAMLS, IndirectAMLS
export CentersMethod, GridCenters, Greedy, GeometricGreedy, Leja
export StartingMethod, GridStart
export SystemMethod, GridSystem
export AdaptiveRefinementMethod, TEAD, MEP, LOLA, MIPT, SFCVT, MASA, EI
export SFCVT_Method, EnumerationX, BlackBoxX, OptimX, SimulatedAnnealingX, EvolutionX
export Metrics, Times

export ErrorType, RMSE, RAE, NORM, NRMSE

# ==============================================================================
struct System
    adj::Matrix{Int}
    connectivity::Any
    types::Dict{Int64,Vector{Int64}}
    percolation::Bool

    # Constructor with default values 
    function System(
        adj::Matrix{Int},
        connectivity::Any,
        types::Dict{Int,Vector{Int}},
        percolation::Bool=true,
    )
        return new(adj, connectivity, types, percolation)
    end
end

mutable struct Simulation
    samples::Int
    variation_tolerance::Float64
    weight_change_tolerance::Float64
    threshold::Union{Float64,Nothing}  # nothing prior to percolation

    # Constructor with default values
    function Simulation(
        samples::Int=1000,
        variation_tolerance::Float64=1e-3,
        weight_change_tolerance::Float64=1e-3,
        threshold::Union{Float64,Nothing}=nothing,
    )
        return new(samples, variation_tolerance, weight_change_tolerance, threshold)
    end
end

mutable struct Points
    coordinates::Union{Array,Nothing}
    idx::Union{Number,Vector,Matrix,Nothing}
    solution::Union{Float64,Vector,Matrix,Nothing}
    confidence::Union{Float64,Vector,Matrix,Nothing}
end

# ============================== BASIS FUNCTIONS ===============================

abstract type BasisFunctionMethod end

struct Gaussian <: BasisFunctionMethod end

# ============================== SIMULATIONS ===================================

abstract type SimulationType end

struct MonteCarloSimulation <: SimulationType end
struct IntervalPredictorSimulation <: SimulationType end

# ============================= SHAPE PARAMETERS ================================

abstract type ShapeParameterMethod end
struct Hardy <: ShapeParameterMethod end
struct Rippa <: ShapeParameterMethod end

struct DirectAMLS <: ShapeParameterMethod
    max_iterations::Int
    tolerance::Float64

    function DirectAMLS(
        max_iterations::Union{Nothing,Int}=1000, tolerance::Union{Nothing,Float64}=1e-6
    )
        return new(max_iterations, tolerance)
    end
end

struct IndirectAMLS <: ShapeParameterMethod
    max_iterations::Int
    tolerance::Float64

    function IndirectAMLS(
        max_iterations::Union{Nothing,Int}=1000, tolerance::Union{Nothing,Float64}=1e-5
    )
        return new(max_iterations, tolerance)
    end
end

# struct Franke <: ShapeParameterMethod end   # only proven with MQ 
# struct Kuo <: ShapeParameterMethod end      # only proven with MQ

# ================================ SYSTEMS =====================================

abstract type SystemMethod end

struct GridSystem <: SystemMethod
    dims::Tuple
end

# ================================ SYSTEMS =====================================

abstract type AdaptiveRefinementMethod end

struct TEAD <: AdaptiveRefinementMethod end
struct MIPT <: AdaptiveRefinementMethod
    alpha::Float64
    beta::Float64
    num_initial_points::Int
    num_additional_points::Int

    # default values based on Combecq et al. 2011
    function MIPT(
        alpha::Float64=0.5,
        beta::Float64=0.3,
        num_initial_points::Int=150,
        num_additional_points::Int=50,
    )
        return new(alpha, beta, num_initial_points, num_additional_points)
    end
end

struct MEPE <: AdaptiveRefinementMethod end
struct LOLA <: AdaptiveRefinementMethod end
struct EI <: AdaptiveRefinementMethod end

abstract type SFCVT_Method end
struct BlackBoxX <: SFCVT_Method end
struct OptimX <: SFCVT_Method end
struct EnumerationX <: SFCVT_Method end
struct SimulatedAnnealingX <: SFCVT_Method end
struct EvolutionX <: SFCVT_Method end

struct SFCVT <: AdaptiveRefinementMethod
    method::SFCVT_Method

    function SFCVT(method::Union{SFCVT_Method,Nothing}=EnumerationX())
        return new(method)
    end
end

struct MASA <: AdaptiveRefinementMethod end

# ================================ CENTERS =====================================

abstract type CentersMethod end

struct GridCenters <: CentersMethod
    centers_interval::Vector{Int}       # number of centers in each dimension

    function GridCenters(centers_interval::Vector{Int}=[15, 15])
        return new(centers_interval)
    end
end

struct Greedy <: CentersMethod
    nCenters::Int

    function Greedy(nCenters::Int=225)
        return new(nCenters)
    end
end

struct GeometricGreedy <: CentersMethod
    nCenters::Int

    function GeometricGreedy(nCenters::Int=225)
        return new(nCenters)
    end
end

struct Leja <: CentersMethod
    nCenters::Int

    function Leja(nCenters::Int=225)
        return new(nCenters)
    end
end

# ================================ CENTERS =====================================

abstract type ErrorType end

struct RMSE <: ErrorType end
struct RAE <: ErrorType end
struct NORM <: ErrorType end
struct NRMSE <: ErrorType end

# ================================ STARTING ====================================

abstract type StartingMethod end

struct GridStart <: StartingMethod end

# ==============================================================================

struct Methods
    simulation_method::Union{SimulationType,Nothing}
    starting_points_method::Union{StartingMethod,Nothing}
    centers_method::Union{CentersMethod,Nothing}
    weight_change_method::Union{ErrorType,Nothing}
    shape_parameter_method::Union{ShapeParameterMethod,Nothing}
    basis_function_method::Union{BasisFunctionMethod,Nothing}
    adaptive_refinement_method::Union{AdaptiveRefinementMethod,Nothing}
end
# ================================ DATA ========================================

struct Times
    total::Float64
    centers::Float64
    shape_parameter::Float64
    adaptive_refinement::Float64

    function Times(
        total::Float64=0.0,
        centers::Float64=0.0,
        shape_parameter::Float64=0.0,
        adaptive_refinement::Float64=0.0,
    )
        return new(total, centers, shape_parameter, adaptive_refinement)
    end
end

struct Metrics
    time::Times
    iterations::Union{Nothing,Int}

    function Metrics(time::Times, iterations::Union{Nothing,Int}=nothing)
        return new(time, iterations)
    end
end

# =========================== MODELS ===========================================
abstract type SimulationModel end

struct MonteCarloModel <: SimulationModel end

struct PredictorModel <: SimulationModel
    evaluated_points::Points
    centers::Array
    shape_parameter::Float64
    weights::Array
    w_u::Vector{Float64}
    w_l::Vector{Float64}
end

mutable struct Model
    Phi::Points
    model::SimulationModel                          # Specific to the type of simulation
    sys::System                                     # For post-processing access
    sim::Simulation
    method::Methods
    metrics::Union{Metrics,Nothing}
    errors::Union{Dict{String,Float64},Nothing}     # starts nothing, then populated
end

# ==============================================================================

end
