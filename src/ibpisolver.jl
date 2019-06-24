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

	function Base.println(policy::IBPIPolicy)
		for l in 0:length(policy.controllers)-1
		    println("Level $l")
		    println(policy.controllers[l])
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
            @deb("Level 0", :flow)

    		evaluate!(policy.controllers[0])
            @deb(policy.controllers[0], :data)

    		improved, tangent_b_vec[1]  = partial_backup!(policy.controllers[0] ; minval = config.minval)

            if improved
                @deb("Improved level 0", :flow)
                @deb(policy.controllers[0], :data)
            else
				@deb("Did not improve level 0", :flow)
				escaped = escape_optima_standard!(policy.controllers[0], tangent_b_vec[1]; minval = 1e-10)
				improved == improved || escaped
			end
    	else
            @deb("Level $level", :flow)
    		evaluate!(policy.controllers[level], policy.controllers[level-1])

            @deb(policy.controllers[level], :data)

    		improved_single, tangent_b_vec[level+1] = partial_backup!(policy.controllers[level], policy.controllers[level-1]; minval = config.minval)
			if improved_single
                @deb("Improved level $level", :flow)
                @deb(policy.controllers[level], :data)
            else
				@deb("Did not improve level $level", :flow)
				escaped_single = escape_optima_standard!(policy.controllers[level], policy.controllers[level-1], tangent_b_vec[level+1]; minval = config.minval)
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
		@save "savedcontrollers/$name.jld2" policy
	end

	function load_policy(name::String)
		@load "savedcontrollers/$name.jld2" policy
		return policy
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
		dist = IPOMDPs.transition(ipomdp, state, ai, aj)
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

	function execute_step(state::S, ai::A, aj::A, frame::IPOMDP) where {S, A}
		new_state = compute_s_prime(state, ai, aj, frame)
		observation = compute_observation(s_prime, ai, aj, frame)
		return new_state, observation
	end


	function IBPIsimulate(policy::IBPIPolicy, maxsteps::Int64) where {S, A, W}
		stats_i = stats()
		stats_j = stats()
		frame_i = policy.controllers[2].frame
		initial = ones(length(IPOMDPs.states(frame_i)), length(policy.controllers[1].nodes))
		initial = initial ./ length(initial)
		agent_i = IBPIAgent(policy.controllers[2], initial)

		frame_j = policy.controllers[1].frame
		initial_j = ones(length(IPOMDPs.states(frame_j)), length(policy.controllers[0].nodes))
		initial_j = initial_j ./ length(initial_j)
		agent_j = IBPIAgent(policy.controllers[1], initial_j)

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

	mutable struct stats

		#agent_i
		correct::Int64
		wrong::Int64
		listen::Int64

		correct_z_l::Int64
		correct_z_ol::Int64
		correct_z_or::Int64

		wrong_z_l::Int64
		wrong_z_ol::Int64
		wrong_z_or::Int64
	end

	stats() = stats(0,0,0,0,0,0,0,0,0)

	function computestats(stats::stats, ai::A, aj::A, state::S, s_prime::S, zi::W, zj::W) where {S, A, W}
		#action stats
		if ai != :L
			if (ai == :OL && state == :TR) || (ai == :OR && state == :TL)
				stats.correct += 1
			else
				stats.wrong += 1
			end
		else
			stats.listen += 1
		end

		#observation stats
		if ai == :L
			if aj == :L
				if (s_prime == :TL && zi == :GLS) || (s_prime == :TR && zi == :GRS)
					stats.correct_z_l += 1
				else
					stats.wrong_z_ol += 1
				end
			elseif aj == :OR
				if (s_prime == :TL && zi == :GLCR) || (s_prime == :TR && zi == :GRCR)
					stats.correct_z_or += 1
				else
					stats.wrong_z_or += 1
				end
			elseif aj == :OL
				if (s_prime == :TL && zi == :GLCL) || (s_prime == :TR && zi == :GRCL)
					stats.correct_z_ol += 1
				else
					stats.correct_z_ol += 1
				end
			end
		end
		return stats
	end


	function average_listens(stats::stats)
		avg_l = stats.listen / (stats.correct + stats.wrong)
		println("Average listens per opening: $avg_l")

	end

	function average_correct_obs(stats::stats)
		avg_l = stats.correct_z_l / (stats.wrong_z_l + stats.correct_z_l)
		avg_or = stats.correct_z_or / (stats.wrong_z_or + stats.correct_z_or)
		avg_ol = stats.correct_z_ol / (stats.wrong_z_ol + stats.correct_z_ol)
		println("avg_l: $avg_l")
		println("avg_or: $avg_or")
		println("avg_ol: $avg_ol")
	end
