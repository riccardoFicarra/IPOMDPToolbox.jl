#=
ibpisolver.jl:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
	using POMDPs
	using IPOMDPs
	using IPOMDPToolbox
	using Dates
	using JLD2

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
		:( $type in debug && println($(esc(str)))  )
    end

	include("statistics.jl")
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
	config = IBPISolver(0, 10, 1e-10, 300)


	"""
	functions to overwrite default values
	# Arguments
	- `force::Int64`: force all controllers to be initialized with a node with the same action.
	- `maxrep::Int64`: maximum number of iterations after which IBPI stops
	- `minval::Float64`: if a number is below minval it is considered as zero.
	- `timeout::Int64`: number of seconds after which the algorithm stops.
	"""
	function set_solver_params(force::Int64, maxrep::Int64, minval::Float64, timeout::Int64)
		global config = IBPISolver(force, maxrep, minval, timeout)
	end

    function Base.println(controller::AbstractController)
		for (id,node) in controller.nodes
            println(node)
        end
		println("$(length(controller.nodes)) nodes")
    end


    struct IBPIPolicy{S, A, W}
        #so far it's controller level -> controller
        #level 0 is the pompdp
        controllers::Array{Array{AbstractController,1}}
		maxlevel::Int64
    end

	"""
	maxlevelframe is the frame of the agent we want to build
	add as many frame arrays as levels, in descending order.
	"""
    function IBPIPolicy(maxlevelframe::IPOMDP{S, A, W}, emulated_frames...; force = 0) where {S, A, W}
		#i dont consider maxlevelframe because we're starting from lv0
		maxlevel = length(emulated_frames)+1
		@deb("maxlevel = $maxlevel", :multiple)
		controllers = Array{Array{AbstractController, 1}}(undef, maxlevel)
		max_level_controller = InteractiveController(maxlevel, maxlevelframe; force = force)
		controllers[maxlevel] = [max_level_controller]
		#emulated_frames is a tuple{vector{IPOMDP}}
		for i in 1:maxlevel-2
			controllers[maxlevel-i] = [InteractiveController(maxlevel-i, emulated_frames[i][frame]; force = force) for frame in 1:length(emulated_frames[i])]
		end
		#pomdp part, level 1
		controllers[1] = [Controller( emulated_frames[maxlevel-1][frame]; force = force) for frame in 1:length(emulated_frames[maxlevel-1])]
        return IBPIPolicy{S, A, W}(controllers, maxlevel)
    end

	function Base.println(policy::IBPIPolicy)
		for l in 1:policy.maxlevel
			println("Level $l")
			for frame_index in 1:length(policy.controllers[l])
				println("Frame $frame_index:  $(typeof(policy.controllers[l][frame_index].frame))")
			    println(policy.controllers[l][frame_index])
			end
		end
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

    function eval_and_improve!(policy::IBPIPolicy, level::Int64)
		debug = Set([:flow])
		@deb("called @level $level", :flow)
		maxlevel = length(policy.controllers)
        improved = false
    	if level >= 2
    		improved, tangent_b_vec = eval_and_improve!(policy, level-1)
    	end
        @deb("evaluating level $level", :flow)
    	if level == 1
            tangent_b_vec = Vector{Dict{Int64, Array{Float64}}}(undef, maxlevel)
			for controller in policy.controllers[1]
    			evaluate!(controller)
				println("Level 1 : $(length(controller.nodes)) nodes")
				@deb(controller, :data)
				start_time = datetime2unix(now())

	    		improved, tangent_b_vec[1]  = partial_backup!(controller ; minval = config.minval, add_one = true)
				@deb("Elapsed time for level $level: $(datetime2unix(now()) - start_time)",:stats)
	            if improved
	                @deb("Improved level 1", :flow)
	                @deb(policy.controllers[1], :data)
	            else
					@deb("Did not improve level 1", :flow)
					escaped = escape_optima_standard!(controller, tangent_b_vec[1]; add_one = false, minval = 1e-10)
					improved == improved || escaped
				end
			end
    	else
			for controller in policy.controllers[level]
	            println("Level $level : $(length(controller.nodes)) nodes")
	    		evaluate!(controller, policy.controllers[level-1])

	            @deb(controller, :data)
				start_time = datetime2unix(now())

	    		improved_single, tangent_b_vec[level] = partial_backup!(controller, policy.controllers[level-1]; minval = config.minval, add_one = true)

				@deb("Elapsed time for level $level: $(datetime2unix(now()) - start_time)", :stats)

				if improved_single
	                @deb("Improved level $level", :flow)
	                @deb(controller, :data)
	            else
					@deb("Did not improve level $level", :flow)
					escaped_single = escape_optima_standard!(controller, policy.controllers[level-1], tangent_b_vec[level];add_one = false, minval = config.minval)
					improved_single = improved_single || escaped_single

					improved = improved || improved_single

				end
			end
    	end
    	return improved, tangent_b_vec
    end

    function ibpi!(policy::IBPIPolicy)
		iterations = 0
        escaped = true
		#full backup part to speed up
		start_time = datetime2unix(now())
		for controller in policy.controllers[1]
			evaluate!(controller)
			if length(controller.nodes) <= 1
				full_backup_stochastic!(controller)
			end
			@deb("Level0 after full backup", :flow)
			@deb(controller, :flow)
		end

		for level in 2:policy.maxlevel
			for controller in policy.controllers[level]
				evaluate!(controller, policy.controllers[level-1])
				checkController(controller, config.minval)
				if length(controller.nodes) <= 1
					full_backup_stochastic!(controller, policy.controllers[level-1]; minval = config.minval)
					@deb("Level $level after full backup", :flow)
					@deb(controller, :flow)
				end
			end
		end
		#start of the actual algorithm
        while escaped  && iterations <= config.maxrep
            escaped = false
            improved = true
            tangent_b_vec = nothing
            while improved && iterations <= config.maxrep && datetime2unix(now()) < start_time+config.timeout
                #@deb("Iteration $iterations / $max_iterations", :flow)
                improved, tangent_b_vec = eval_and_improve!(policy, policy.maxlevel)
                iterations += 1
				if !improved
					println("Algorithm stopped because it could not improve controllers anymore")
				end
				if iterations >= config.maxrep
					println("maxrep exceeded $iterations")
				end
            end
        end
    end

	function save_policy(policy::IBPIPolicy, name::String)
	    @save "savedcontrollers/$name.jld2" policy time_stats
	end

	function load_policy(name::String)
	    @load "savedcontrollers/$name.jld2" policy time_stats
	    return policy, time_stats
	end

	function solve_fresh(policy::IBPIPolicy, n_steps::Int64, step_length::Int64, maxsimsteps::Int64 ; save = "", benchmark = true)
		force = 3
		max_iterations = 1000
		# benchmark_data[1] = number of nodes in maxlevel
		#benchmark_data[2] = tot number of nodes
		# benchmark_data[3] = time
		# benchmark_data[4] = value
		benchmark_data = zeros(4, n_steps)
		reset_timers()
		for step in 1:n_steps
		    filename_dst = "$(save)_$(step*step)m"
		    set_solver_params(force,max_iterations,1e-10,step_length*60)

	        start_time("ibpi")
	        ibpi!(policy)
	        stop_time("ibpi")


			if benchmark
				value,stats_i, stats_j = IBPIsimulate(policy,maxsimsteps)

				old_benchmark_data

				for level in 1:policy.maxlevel
					if level == policy.maxlevel
						benchmark_data[1][step] = length(policy.controllers[level].nodes)
					end
					benchmark_data[2][step] += length(policy.controllers[level].nodes)
				end
				benchmark_data[3] = step * step_length
				benchmark_data[4] = value
			end

		    if save != ""
		        save_policy(policy, filename_dst)
		    end
		end
		return policy, time_Stats
	end

	function continue_solving(src_filename::String, n_steps::Int64, step_length::Int64; benchmark = true)
		force = 3
		max_iterations = 1000
		policy, saved_time_stats = load_policy(filename_src)
		global time_stats = saved_time_stats
		new_benchmark = zeros(4, n_steps)

		name = split(src_filename, "_")
		prefix = name[1]
		start_duration = parse(Int64,name[2])

		for step in 1:n_steps
		    filename_dst = "$(prefix)_$(start_duration + step*step)m"
		    set_solver_params(force,max_iterations,1e-10,step_length*60)

	        start_time("ibpi")
	        ibpi!(policy)
	        stop_time("ibpi")
	        println(policy)


			if benchmark
				value,stats_i, stats_j = IBPIsimulate(policy,maxsimsteps)
				for level in 1:policy.maxlevel
					if level == policy.maxlevel
						new_benchmark[1][step] = length(policy.controllers[level].nodes)
					end
					new_benchmark[2][step] += length(policy.controllers[level].nodes)
				end
				new_benchmark[3] = step * step_length
				new_benchmark[4] = value
			end

		    save_policy(policy, filename_dst)
		end
		old_benchmark_data = hcat(old_benchmark_data, new_benchmark)
		return policy, time_stats
	end
	include("./simulator.jl")
