#=
ibpisolver.jl:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#

abstract type AbstractController end

    include("bpipolicyutils.jl")
    include("ibpi.jl")
    struct IBPISolver
        # Here should go some settings
        timeout::Float64
    end


    struct IBPIPolicy{S, A, W}
        #so far it's controller level -> controller
        #level 0 is the pompdp, max level is the chosen agent controller
        controllers::Dict{Int64, AbstractController}
    end

    function IBPIPolicy(ipomdp::IPOMDP{S, A, W}, pomdp::POMDP{A, W}, maxlevel::Int64; force = 0) where {S, A, W}
        controllers = init_controllers(ipomdp, pomdp, maxlevel, force)
        return IBPIPolicy{S, A, W}(controllers)
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
        #try
        #mkdir("./tmp")
        #catch
        # Already present
        #end
        return IBPIPolicy(ipomdp)
    end

    function eval_and_improve!(policy::IBPIPolicy, level::Int64)
    	improved = false
    	if level >= 1
    		improved = eval_and_improve!(policy, level-1)
    	end
    	if level == 0
    		evaluate!(policy.controllers[0], policy.controllers[0].pomdp)
    		improved = partial_backup!(policy.controllers[0], policy.controllers[0].pomdp)
    	else
    		evaluate!(policy.controllers[level], policy.controllers[level-1])
    		improved = partial_backup!(policy.controllers[level], policy.controllers[level-1])
    	end
    	return improved
    end
