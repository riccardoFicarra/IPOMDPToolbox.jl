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
		maxlevel = length(policy.controllers) -1
        improved = false
    	if level >= 1
    		improved, tangent_b_vec = eval_and_improve!(policy, level-1)
    	end
        @deb("evaluating level $level", :flow)
    	if level == 0
            tangent_b_vec = Vector{Dict{Int64, Array{Float64}}}(undef, maxlevel+1)
            println("Level 0 : $(length(policy.controllers[0].nodes)) nodes")

    		evaluate!(policy.controllers[0])
            @deb(policy.controllers[0], :data)
			start_time = datetime2unix(now())

    		improved, tangent_b_vec[1]  = partial_backup!(policy.controllers[0] ; minval = config.minval, add_one = true)
			@deb("Elapsed time for level $level: $(datetime2unix(now()) - start_time)",:stats)
            if improved
                @deb("Improved level 0", :flow)
                @deb(policy.controllers[0], :data)
            else
				@deb("Did not improve level 0", :flow)
				escaped = escape_optima_standard!(policy.controllers[0], tangent_b_vec[1]; add_one = false, minval = 1e-10)
				improved == improved || escaped
			end
    	else
            println("Level $level : $(length(policy.controllers[level].nodes)) nodes")
    		evaluate!(policy.controllers[level], policy.controllers[level-1])

            @deb(policy.controllers[level], :data)
			start_time = datetime2unix(now())

    		improved_single, tangent_b_vec[level+1] = partial_backup!(policy.controllers[level], policy.controllers[level-1]; minval = config.minval, add_one = true)

			@deb("Elapsed time for level $level: $(datetime2unix(now()) - start_time)", :stats)

			if improved_single
                @deb("Improved level $level", :flow)
                @deb(policy.controllers[level], :data)
            else
				@deb("Did not improve level $level", :flow)
				escaped_single = escape_optima_standard!(policy.controllers[level], policy.controllers[level-1], tangent_b_vec[level+1];add_one = false, minval = config.minval)
				improved_single = improved_single || escaped_single
			end
			# global debug = Set([])

			improved = improved || improved_single
    	end
    	return improved, tangent_b_vec
    end

    function ibpi!(policy::IBPIPolicy)
		maxlevel = length(policy.controllers) -1
		iterations = 0
        escaped = true
		#full backup part to speed up
		start_time = datetime2unix(now())

		evaluate!(policy.controllers[0])
		if length(policy.controllers[0].nodes) <= 1
			full_backup_stochastic!(policy.controllers[0]; minval = config.minval)
		end
		@deb("Level0 after full backup", :flow)
		@deb(policy.controllers[0], :flow)
		for level in 1:maxlevel
			evaluate!(policy.controllers[level], policy.controllers[level-1])
			checkController(policy.controllers[level], config.minval)
			if length(policy.controllers[level].nodes) <= 1
				full_backup_stochastic!(policy.controllers[level], policy.controllers[level-1]; minval = config.minval)
				@deb("Level $level after full backup", :flow)
				@deb(policy.controllers[level], :flow)
			end
		end
		#start of the actual algorithm
        while escaped  && iterations <= config.maxrep
            escaped = false
            improved = true
            tangent_b_vec = nothing
            while improved && iterations <= config.maxrep && datetime2unix(now()) < start_time+config.timeout
                #@deb("Iteration $iterations / $max_iterations", :flow)
                improved, tangent_b_vec = eval_and_improve!(policy, maxlevel)
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
		@save "savedcontrollers/$name.jld2" policy #time_stats
	end

	function load_policy(name::String)
		@load "savedcontrollers/$name.jld2" policy #time_stats
		return policy, time_stats
	end
	#include("bpigraph.jl")

	mutable struct IBPIAgent
		controller::AbstractController
		current_node::Node
		value::Float64
	end
	function IBPIAgent(controller::AbstractController, initial_belief::Array{Float64})
		best_node = nothing
		best_value = nothing
		for (id, node) in controller.nodes
			new_value = sum(initial_belief[i]*node.value[i] for i in 1:length(initial_belief))
			if best_node == nothing || new_value > best_value
				best_node = node
				best_value = new_value
			end
		end
		return IBPIAgent(controller, best_node, 0.0)
	end
	function best_action(agent::IBPIAgent)
		return chooseWithProbability(agent.current_node.actionProb)
	end
	function update_agent!(agent::IBPIAgent, action::A, observation::W) where {A, W}
		agent.current_node = chooseWithProbability(agent.current_node.edges[action][observation])
	end
	function compute_s_prime(state::S, ai::A, aj::A, frame::IPOMDP) where {S, A}
		dist = IPOMDPs.transition(frame, state, ai, aj)
		items = dist.probs
		randn = rand() #number in [0, 1)
		for i in 1:length(dist.vals)
			if randn <= items[i]
				return dist.vals[i]
			else
				randn-= items[i]
			end
		end
		error("Out of dict bounds while choosing items")
	end

	function compute_s_prime(state::S, ai::A, aj::A, frame::POMDP) where {S, A}
		dist = POMDPs.transition(frame, state, ai)
		items = dist.probs
		randn = rand() #number in [0, 1)
		for i in 1:length(dist.vals)
			if randn <= items[i]
				return dist.vals[i]
			else
				randn-= items[i]
			end
		end
		error("Out of dict bounds while choosing items")
	end

	function compute_observation(s_prime::S, ai::A, aj::A, frame::IPOMDP) where {S, A}
		dist = IPOMDPs.observation(ipomdp, s_prime, ai, aj)
		items = dist.probs
		randn = rand() #number in [0, 1)
		for i in 1:length(dist.vals)
			if randn <= items[i]
				return dist.vals[i]
			else
				randn-= items[i]
			end
		end
		error("Out of dict bounds while choosing items")
	end

	function compute_observation(s_prime::S, ai::A, aj::A, frame::POMDP) where {S, A}
		dist = POMDPs.observation(frame, s_prime, ai)
		items = dist.probs
		randn = rand() #number in [0, 1)
		for i in 1:length(dist.vals)
			if randn <= items[i]
				return dist.vals[i]
			else
				randn-= items[i]
			end
		end
		error("Out of dict bounds while choosing items")
	end

	function IBPIsimulate(policy::IBPIPolicy, maxsteps::Int64) where {S, A, W}
		stats_i = stats()
		stats_j = stats()
		maxlevel = length(policy.controllers) -1
		frame_i = policy.controllers[maxlevel].frame
		anynode = first(policy.controllers[maxlevel].nodes)[2]
		initial = ones(size(anynode.value))
		initial = initial ./ length(initial)
		agent_i = IBPIAgent(policy.controllers[maxlevel], initial)

		frame_j = policy.controllers[maxlevel-1].frame
		anynode_j = first(policy.controllers[maxlevel-1].nodes)[2]
		initial_j = ones(size(anynode_j.value))
		initial_j = initial_j ./ length(initial_j)
		if maxlevel - 1 == 0
			agent_j = IBPIAgent(policy.controllers[maxlevel-1], initial_j)

		else
			agent_j = IBPIAgent(policy.controllers[maxlevel-1], initial_j)
		end
		state = randn() > 0.5 ? :TL : :TR
		value = 0.0
		for i in 1:95
			print(" ")
		end
		println("end v")
		for i in 1:maxsteps
			if i % (maxsteps/100) == 0
				print("|")
			end
			ai = best_action(agent_i)
			aj = best_action(agent_j)
			@deb("state: $state -> ai: $ai, aj: $aj", :sim)

			value +=  IPOMDPs.reward(frame_i, state, ai, aj)
			@deb("value this step: $(IPOMDPs.reward(frame_i, state, ai, aj))", :sim)

			s_prime = compute_s_prime(state, ai, aj, frame_i)

			zi = compute_observation(s_prime, ai, aj, frame_i)
			zj = compute_observation(s_prime, aj, ai, frame_j)
			@deb("zi -> $zi, zj -> $zj", :sim)
			update_agent!(agent_i, ai, zi)
			update_agent!(agent_j, aj, zj)
			stats_i = computestats(stats_i, ai, aj, state, s_prime, zi, zj)
			stats_j = computestats(stats_j, aj, ai, state, s_prime, zj, zi)

			state = s_prime
		end
		println()
		return value/maxsteps , stats_i, stats_j
	end
