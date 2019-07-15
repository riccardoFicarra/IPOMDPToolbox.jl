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
    BPIPolicy,
    solve_fresh!,
    continue_solving,
    print_solver_stats,
    load_policy,
    IBPIsimulate

    include("interactivebelief.jl")
    include("gpomdp.jl")
    include("reductionsolver.jl")
    include("ibpisolver.jl")
    include("ipomdpstoolbox.jl")
    include("functions.jl")
end

#for thesis
#single agent (POMDP)
#interactive: i coop with j optimal
#understand i's strategy
#try i with the three frame type
#add level and go to level 3
#implement random pomdp -> 1/3 1/3 1/3
#change reward to discounted but do multiple simulations until its stable
#compare converged controllers
#try 4x4 wumpus world with just wumpus
