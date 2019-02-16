#=
ibpisolver.jl:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
include("ibpipolicyutils.jl")

struct IBPISolver
    # Here should go some settings
    timeout::Float64
end

struct IBPIPolicy{A, W}
    #temporary, find a way to store multiple controllers for frames and other agents
    controllers::Vector{Controller{A, W}}
end

struct BPIPolicy{A, W}
    controller::Controller{A, W}
end


"""
    Return the policy type used by the solver. Since ReductionSolver is an online solver, the policy doesn't really exist.
    It is used as a container to maintain data through time
    solve(solver::ReductionSolver, ipomdp::IPOMDP{S,A,W})
Return:
    ReductionPolicy{S,A,W}
"""
function IPOMDPs.solve(solver::IBPISolver, ipomdp::IPOMDP{S,A,W}) where {S,A,W}
    # Create the folder used by the action function
    try
    mkdir("./tmp")
    catch
    # Already present
    end
    return IBPIPolicy(ipomdp, solver.timeout)
end

"""
    Uses the policy to compute the best action.
    action(policy::ReductionPolicy{S,A,W}, b::DiscreteInteractiveBelief{S,A,W})
Return:
    action::A
"""
function IPOMDPs.action(policy::IBPIPolicy{S,A,W}, b::DiscreteInteractiveBelief{S,A,W}) where {S,A,W}

    return :none
end
