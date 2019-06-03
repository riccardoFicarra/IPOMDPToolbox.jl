#=
ibpisolver.jl:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
	using POMDPs
	using IPOMDPs
	using IPOMDPToolbox
	"""
	Abstract type used for any controller, interactive or not.
	"""
	abstract type AbstractController end

    """
    Snippet to have debug utility. Use @deb(String) to print debug info
    """
    global debug = Set{Symbol}()
	macro deb(str)
		:( :data in debug && println($(esc(str))) )
    end
    macro deb(str, type)
		:( $type in debug && println($(esc(str))) )
    end

    include("bpipolicyutils.jl")
    include("ibpi.jl")

	"""
	Structure used for global parameters.


	"""
    struct IBPISolver
        # Here should go some settings
		#TODO add solver parameters here and pass this to all functions (or find a way to make all functions see this object)
		#minval
		force::Int64
		#partial_backup_add_one
		#escape_optima_add_one
		maxrep::Int64
		#timeout
		minval::Float64
        timeout::Int64
    end
	"""
	Default config values
	"""
	config = IBPISolver(0, 10, 1e-15, 300)


	"""
	functions to overwrite default values
	# Arguments
	- `force::Int64`: force all controllers to be initialized with a node with the same action.
	- `maxrep::Int64`: maximum number of iterations after which IBPI stops
	- `minval::Float64`: if a number is below minval it is considered as zero.
	- `timeout::Int64`: number of seconds after which the algorithm stops.
	"""
	function set_solver_params(force::Int64, maxrep::Int64, minval::Float64, timeout::Int64)
		config = IBPISolver(force, maxrep, minval, timeout)
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
        @deb("called @level $level", :flow)
        improved = false
    	if level >= 1
    		improved, tangent_b_vec = eval_and_improve!(policy, level-1, maxlevel)
    	end
        @deb("evaluating level $level", :flow)
    	if level == 0
            tangent_b_vec = Vector{Dict{Int64, Array{Float64}}}(undef, maxlevel+1)
            @deb("Level 0", :flow)

    		evaluate!(policy.controllers[0])
            @deb(policy.controllers[0], :data)

    		improved, tangent_b_vec[1]  = partial_backup!(policy.controllers[0] ; minval = config.minval)

            if improved
                @deb("Improved level 0", :flow)
                @deb(policy.controllers[0], :data)
            end
    	else
            @deb("Level $level", :flow)
    		evaluate!(policy.controllers[level], policy.controllers[level-1])
            @deb(policy.controllers[level], :data)

    		improved, tangent_b_vec[level+1] = partial_backup!(policy.controllers[level], policy.controllers[level-1]; minval = config.minval)
            if improved
                @deb("Improved level $level", :flow)
                @deb(policy.controllers[level], :data)
            end
    	end
    	return improved, tangent_b_vec
    end

    function ibpi!(policy::IBPIPolicy, maxlevel::Int64, max_iterations::Int64)
        iterations = 0
        escaped = true
		#full backup part to speed up
		evaluate!(policy.controllers[0])
		full_backup_stochastic!(policy.controllers[0]; minval = config.minval)
		@deb("Level0 after full backup", :data)
		@deb(policy.controllers[0], :data)
		for level in 1:maxlevel
			@deb("Level $level after full backup", :data)
			evaluate!(policy.controllers[level], policy.controllers[level-1])
			full_backup_stochastic!(policy.controllers[level], policy.controllers[level-1]; minval = config.minval)
			@deb(policy.controllers[level], :data)
		end
		#start of the actual algorithm
        while escaped  && iterations <= max_iterations
            escaped = false
            improved = true
            tangent_b_vec = nothing
            while improved && iterations <= max_iterations
                @deb("Iteration $iterations / $max_iterations", :flow)
                improved, tangent_b_vec = eval_and_improve!(policy, maxlevel, maxlevel)
                iterations += 1
            end
            for level in maxlevel:-1:1
                evaluate!(policy.controllers[level], policy.controllers[level-1])
                escaped_single = escape_optima_standard!(policy.controllers[level], policy.controllers[level-1], tangent_b_vec[level+1]; minval = config.minval)
                escaped = escaped || escaped_single
                if escaped_single
                    @deb("Level $level: escaped", :flow)
                    @deb(policy.controllers[level], :data)
                    @deb(" ")

                end
            end
            escaped = escape_optima_standard!(policy.controllers[0], tangent_b_vec[1]; minval = 1e-10)
            if escaped
                @deb("Level 0: escaped", :flow)
                @deb(policy.controllers[0], :data)
                @deb(" ")
            end
        end
    end
	include("bpigraph.jl")
