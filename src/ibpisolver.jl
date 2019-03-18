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

    function BPIPolicy(pomdp::POMDP{A, W}, force::Int64) where {A, W}
        BPIPolicy(Controller(pomdp, force))
    end

    function IPOMDPs.Model(pomdp::POMDP;depth=0, solvertype = :IBPI, force = 0)
        # Timeout
        t = 10.0
        for i = 1:depth
            t = t/10
        end
        name = hash(pomdp)
        policy = BPIPolicy(pomdp, force)
        solver = IBPISolver(t)
        updater = BeliefUpdaters.DiscreteUpdater(pomdp)
        belief = BeliefUpdaters.uniform_belief(pomdp)

        return pomdpModel(belief, pomdp, updater, policy, depth)
    end
    #=
    function IPOMDPs.Model(ipomdp::IPOMDP;depth=0)
        t = 10.0
        for i = 1:depth
            t = t/10
        end
        solver = ReductionSolver(t)
        updater = DiscreteInteractiveUpdater(ipomdp)
        policy = IPOMDPs.solve(solver, ipomdp)
        belief = IPOMDPs.initialize_belief(updater; depth=depth)

        return ipomdpModel(belief, ipomdp, updater, policy, depth)
    end
    =#
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
