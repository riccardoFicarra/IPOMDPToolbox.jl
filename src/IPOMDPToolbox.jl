module IPOMDPToolbox

using POMDPs
using IPOMDPs
using SARSOP
using POMDPModelTools
using BeliefUpdaters
import Base: (==)
export
    pomdpModel,
    ipomdpModel,

    printPOMDP,

    DiscreteInteractiveBelief,
    DiscreteInteractiveUpdater,

    ReductionSolver,
    ReductionPolicy,

    IBPISolver,
    IBPIPolicy,

    #temporary
    BPIPolicy

    include("interactivebelief.jl")
    include("gpomdp.jl")
    include("reductionsolver.jl")
    include("ipomdpstoolbox.jl")
    include("functions.jl")
end
