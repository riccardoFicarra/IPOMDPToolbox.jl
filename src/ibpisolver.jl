#=
ibpisolver.jl:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
module IBPI
	using POMDPs
	using IPOMDPs
	using IPOMDPToolbox
	"""
	Abstract type used for any controller, interactive or not.
	"""
	abstract type AbstractController end
    """
    Snippet to have debug utility. Use @deb(String) to print debug info
    Modulename.debug[] = true to enable, or just debug[] = true if you are in the module
    """
    global debug = [false]
    macro deb(str)
        :( debug[] && println($(esc(str))))
    end

    include("bpipolicyutils.jl")
    include("ibpi.jl")
    struct IBPISolver
        # Here should go some settings
		#TODO add solver parameters here and pass this to all functions (or find a way to make all functions see this object)
		#minval
		#force
		#partial_backup_add_one
		#escape_optima_add_one
		#maxrep
		#timeout
        timeout::Float64
    end

    function Base.println(controller::AbstractController)
        for (id,node) in controller.nodes
            println(node)
        end
    end


    struct IBPIPolicy{S, A, W}
        #so far it's controller level -> controller
        #level 0 is the pompdp, max level is the chosen agent controller
        controllers::Dict{Int64, AbstractController}
    end

    function IBPIPolicy(ipomdp::IPOMDP{S, A, W}, pomdp::POMDP{A, W}, maxlevel::Int64; force = 0) where {S, A, W}
		if force == 0
			controllers = init_controllers(ipomdp, pomdp, maxlevel)
		else
			controllers = init_controllers(ipomdp, pomdp, maxlevel; force=force)
		end
        return IBPIPolicy{S, A, W}(controllers)
    end


    """
        Return the policy type used by the solver. Since ReductionSolver is an online solver, the policy doesn't really exist.
        It is used as a container to maintain data through time
        solve(solver::ReductionSolver, ipomdp::IPOMDP{S,A,W})
    Return:
        ReductionPolicy{S,A,W}
    """
    function IPOMDPs.solve(solver::IBPISolver, ipomdp::IPOMDP{S,A,W}) where {S,A,W}
        
        return IBPIPolicy(ipomdp)
    end

    function eval_and_improve!(policy::IBPIPolicy, level::Int64, maxlevel::Int64)
        println("called @level $level")
        improved = false
    	if level >= 1
    		improved, tangent_b_vec = eval_and_improve!(policy, level-1, maxlevel)
    	end
        println("evaluating level $level")
    	if level == 0
            tangent_b_vec = Vector{Dict{Int64, Array{Float64}}}(undef, maxlevel+1)
            println("Level 0")

    		evaluate!(policy.controllers[0])
            println(policy.controllers[0])

    		improved, tangent_b_vec[1]  = partial_backup!(policy.controllers[0] ; minval = 1e-10)

            if improved
                println("Improved level 0")
                println(policy.controllers[0])
            end
    	else
            println("Level $level")
    		evaluate!(policy.controllers[level], policy.controllers[level-1])
            println(policy.controllers[level])

    		improved, tangent_b_vec[level+1] = partial_backup!(policy.controllers[level], policy.controllers[level-1]; minval = 1e-10)
            if improved
                println("Improved level $level")
                println(policy.controllers[level])
            end
    	end
    	return improved, tangent_b_vec
    end
    function ibpi!(policy::IBPIPolicy, maxlevel::Int64, max_iterations::Int64)
        iterations = 0
        escaped = true
		#full backup part to speed up
		evaluate!(policy.controllers[0])
		full_backup_stochastic!(policy.controllers[0])
		println("Level0 after full backup")
		println(policy.controllers[0])
		for level in 1:maxlevel
			println("Level $level after full backup")
			evaluate!(policy.controllers[level], policy.controllers[level-1])
			full_backup_stochastic!(policy.controllers[level], policy.controllers[level-1])
			println(policy.controllers[level])
		end
		#start of the actual algorithm
        while escaped  && iterations <= max_iterations
            escaped = false
            improved = true
            tangent_b_vec = nothing
            while improved && iterations <= max_iterations
                println("Iteration $iterations / $max_iterations")
                improved, tangent_b_vec = eval_and_improve!(policy, maxlevel, maxlevel)
                iterations += 1
            end
            for level in maxlevel:-1:1
                evaluate!(policy.controllers[level], policy.controllers[level-1])
                escaped_single = escape_optima_standard!(policy.controllers[level], policy.controllers[level-1], tangent_b_vec[level+1]; minval = 1e-10)
                escaped = escaped || escaped_single
                if escaped_single
                    println("Level $level: escaped")
                    println(policy.controllers[level])
                    println(" ")

                end
            end
            escaped = escape_optima_standard!(policy.controllers[0], tangent_b_vec[1]; minval = 1e-10)
            if escaped
                println("Level 0: escaped")
                println(policy.controllers[0])
                println(" ")
            end
        end
    end
	include("bpigraph.jl")
end
