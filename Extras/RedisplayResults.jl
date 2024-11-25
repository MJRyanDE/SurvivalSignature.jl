using JLD2
using PrettyTables
using DataFrames

if !isdefined(Main, :Import)
    include("../Modules/Import.jl")
end

using Revise
Revise.track("Modules/Import.jl")

using .Import

using ..Structures
using ..Visualization

println("Loading Signatures...")
signatures = load_object("ultimate_signatures_10x10_5000.JLD2")
println("Signatures loaded.\n")

Visualization.printUltimateTest(signatures; latex=false)