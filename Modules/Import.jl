module Import
# ==============================================================================

export SurvivalSignatureUtils,
    Structures,
    StructureCompilation,
    Visualization,
    Error,
    Systems,
    Evaluation,
    BasisFunction,
    ShapeParameter,
    StartingPoints,
    Centers,
    IntervalPredictorModel,
    AdaptiveRefinement,
    Simulate,
    monotonicity_constraints, #from final include
    lsqr                      #from final include
# ==============================================================================
include("SurvivalSignatureUtils.jl")
using .SurvivalSignatureUtils

include("Structures.jl")
using .Structures

include("StructureCompilation.jl")
using .StructureCompilation

include("Visualization.jl")
using .Visualization

include("BasisFunction.jl")
using .BasisFunction

include("Error.jl")
using .Error

include("Systems.jl")
using .Systems

include("Evaluation.jl")
using .Evaluation

include("ShapeParameter.jl")
using .ShapeParameter

include("StartingPoints.jl")
using .StartingPoints

include("Centers.jl")
using .Centers

include("IntervalPredictorModel.jl")
using .IntervalPredictorModel

include("AdaptiveRefinement.jl")
using .AdaptiveRefinement

include("Simulate.jl")
using .Simulate

# convert this to a Module: RadialBasisFunctions (?)
# needed for monotonicity_constraints and lsqr
include("../src/rbf/radialbasisfunctions.jl")
# ==============================================================================
end