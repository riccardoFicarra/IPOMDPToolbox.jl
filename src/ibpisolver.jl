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
	using Serialization

	"""
	Abstract type used for any controller, interactive or not.
	"""
	abstract type AbstractController end


	function frametype(controller::AbstractController)
		return split("$(typeof(controller.frame))", ".")[3]
	end

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
		force::Int64
		maxrep::Int64
		minval::Float64
		timeout::Int64
		min_improvement::Float64
		normalize::Bool
	end
	"""
	Default config values
	"""
	config = IBPISolver(0, 10, 1e-10, 300, 1e-10, true)


	"""
	functions to overwrite default values
	# Arguments
	- `force::Int64`: force all controllers to be initialized with a node with the same action.
	- `maxrep::Int64`: maximum number of iterations after which IBPI stops
	- `minval::Float64`: if a number is below minval it is considered as zero.
	- `timeout::Int64`: number of seconds after which the algorithm stops.
	"""
	function set_solver_params(force::Int64, maxrep::Int64, minval::Float64, timeout::Int64, min_improvement::Float64, normalize::Bool)
		global config = IBPISolver(force, maxrep, minval, timeout, min_improvement, normalize)
	end

	function Base.println(controller::AbstractController)
		for node in controller.nodes
			println(node)
		end
		println("$(length(controller.nodes)) nodes")
	end


	struct IBPIPolicy{S, A, W}
		#Controller level -> controller frames
		#level 1 is the pompdp
		name::String
		controllers::Array{Array{AbstractController,1}}
		maxlevel::Int64
	end

	"""
	maxlevelframe is the frame of the agent we want to build
	add as many frame arrays as levels, in descending order.
	"""
	function IBPIPolicy(name::String, maxlevelframe::IPOMDP{S, A, W}, emulated_frames...; force = 0) where {S, A, W}
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
		return IBPIPolicy{S, A, W}(name, controllers, maxlevel)
	end

	function Base.println(policy::IBPIPolicy)
		println("Policy $(policy.name)")
		for l in 1:policy.maxlevel
			println("Level $l")
			for frame_index in 1:length(policy.controllers[l])
				println("Frame $frame_index:  $(frametype(policy.controllers[l][frame_index]))")
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
		@deb("called @level $level", :flow)
		improved = false
		if level >= 2
			improved_lower = eval_and_improve!(policy, level-1)
		end
		if level == 1
			tangent_b  = Dict{Int64, Array{Float64}}()
			for controller in policy.controllers[1]
				#experimental: if controller of level 0 has converged skip it to avoid losing time
				if !controller.converged

					println("Level 1, frame $(frametype(controller)): $(length(controller.nodes)) nodes")
					evaluate!(controller)
					@deb(controller, :data)
					start_time = datetime2unix(now())

					((improved, tangent_b), time, mem, gc)  = @timed partial_backup!(controller ; add_one = true)
					@deb("Elapsed time for level $level: $(datetime2unix(now()) - start_time)",:stats)
					if improved
						@deb("Improved level 1", :flow)
						@deb(policy.controllers[1], :data)
					else
						@deb("Did not improve level 1", :flow)
						escaped = escape_optima_standard!(controller, tangent_b; add_one = false)
						improved == improved || escaped
						#if the controller failed both improvement and escape mark it as converged.
						controller.converged = !escaped
					end
					log_time_nodes(controller.stats, datetime2unix(now()), length(controller.nodes), mem)

				else
					log_time_nodes(controller.stats, datetime2unix(now()), length(controller.nodes), 0)
					println("Level 1, frame $(frametype(controller)): $(length(controller.nodes)) nodes has converged")
				end
			end
			# current_time = datetime2unix(now())
			# for controller in policy.controllers[1]
			# 	log_time_nodes(controller.stats,current_time,length(controller.nodes), mem)
			# end

		else
			for controller in policy.controllers[level]
				if !controller.converged
					println("Level $level, frame $(frametype(controller)) : $(length(controller.nodes)) nodes")
					evaluate!(controller, policy.controllers[level-1])

					@deb(controller, :data)
					start_time = datetime2unix(now())

					((improved, tangent_b), time, mem, gc)  = @timed partial_backup!(controller, policy.controllers[level-1]; add_one = true)

					@deb("Elapsed time for level $level: $(datetime2unix(now()) - start_time)", :stats)

					if improved
						@deb("Improved level $level", :flow)
						@deb(controller, :data)
					else
						if level == policy.maxlevel
							log_time_nodes(controller.stats, datetime2unix(now()), length(controller.nodes), mem)

							save_policy(policy, 1, length(controller.nodes))
						end
						@deb("Did not improve level $level", :flow)
						escaped = escape_optima_standard!(controller, policy.controllers[level-1], tangent_b ;add_one = false)
						improved = improved || escaped
					end

					#a controller has converged only if also all lower level controllers have converged
					controller.converged = !improved && !improved_lower
					if controller.converged
						println("Level $level, frame $(frametype(controller)) : $(length(controller.nodes)) nodes")
					end
				else
					log_time_nodes(controller.stats, datetime2unix(now()), length(controller.nodes), 0)
					println("Level $level, frame $(frametype(controller)): $(length(controller.nodes)) nodes has converged")
				end
			end
			# current_time = datetime2unix(now())
			# for controller in policy.controllers[level]
			# 	log_time_nodes(controller.stats, current_time, length(controller.nodes), mem)
			# end
		end
		return improved
	end

	function ibpi!(policy::IBPIPolicy, repetition::Int64; start_duration = 0,  n_backups = 0)
		#full backup part to speed up
		for controller in policy.controllers[1]
			evaluate!(controller)
			if length(controller.nodes) <= 1
				full_backup_stochastic!(controller)
				@deb("Level	1 after full backup", :flow)
				@deb(controller, :flow)
			end
		end

		for level in 2:policy.maxlevel
			for controller in policy.controllers[level]
				evaluate!(controller, policy.controllers[level-1])
				checkController(controller)
				if length(controller.nodes) <= 1
					full_backup_stochastic!(controller, policy.controllers[level-1])
					@deb("Level $level after full backup", :flow)
					@deb(controller, :flow)
				end
			end
		end
		start_time = datetime2unix(now())

		for level in 1:policy.maxlevel
			for controller in policy.controllers[level]
				if controller.stats.start_time == 0
					set_start_time(controller.stats, start_time)
				else
					set_start_time(controller.stats, last(controller.stats.data)[1])
				end
			end
		end
		#start of the actual algorithm

		iteration = 1
		step = 1
		if n_backups > 0
			#this is in seconds!!!
			timestep_duration = trunc(Int64, config.timeout / (n_backups))
		end
		while true
			@deb("Iteration $iteration", :flow)
			improved = eval_and_improve!(policy, policy.maxlevel)
			iteration += 1
			# if n_backups > 0 && datetime2unix(now()) - start_time >= timestep_duration * step
			# 	@deb("Saving...", :flow)
			# 	save_policy(policy, repetition, start_duration + step * timestep_duration)
			# 	step += 1
			# end
			if !improved
				println("Algorithm stopped because it could not improve controllers anymore")
				return true
			elseif config.maxrep >= 0 && iteration >= config.maxrep
				println("maxrep exceeded $iteration")
				break
			elseif	datetime2unix(now()) >= start_time+config.timeout
				println("timeout exceeded")
				break
			end
		end
		#only needed when evaluation is cut short
		#but it doesnt take that much time
		for level in 1:policy.maxlevel
			for controller in policy.controllers[level]
				if level == 1
					evaluate!(controller)
				else
					evaluate!(controller, policy.controllers[level-1])
				end
			end
		end
		return false
	end

	function save_policy(policy::IBPIPolicy, repetition::Int64, duration::Int64; converged = false)
		#@save "savedcontrollers/$name.jld2" policy
		for level in 1:policy.maxlevel
			for controller in policy.controllers[level]
				if level == 1
					evaluate!(controller)
				else
					evaluate!(controller, policy.controllers[level-1])
				end
			end
		end
		if !isdir("savedcontrollers/$(policy.name)")
			mkdir("savedcontrollers/$(policy.name)")
		end
		if !isdir("savedcontrollers/$(policy.name)/rep$rep")
			mkdir("savedcontrollers/$(policy.name)/rep$rep")
		end
		if converged
			serialize("savedcontrollers/$(policy.name)/rep$rep/$(policy.name)$(repetition)_conv.policy", policy)
		else
			serialize("savedcontrollers/$(policy.name)/rep$rep/$(policy.name)$(repetition)_$duration.policy", policy)
		end
	end

	function load_policy(name::String, repetition::Int64, duration::Int64; converged = false)
		#@load "savedcontrollers/$name.jld2" policy
		if converged
			if !isfile("savedcontrollers/$(name)/rep$repetition/$(name)$(repetition)_conv.policy")
				return nothing
			end
			policy = deserialize("savedcontrollers/$(name)/rep$repetition/$(name)$(repetition)_conv.policy")
		else
			if !isfile("savedcontrollers/$(name)/rep$repetition/$(name)$(repetition)_$duration.policy")
				return nothing
			end
			policy = deserialize("savedcontrollers/$(name)/rep$repetition/$(name)$(repetition)_$duration.policy")
		end
		return policy
	end


	function print_solver_stats(policy::IBPIPolicy)
		for level in 1:policy.maxlevel
			println("Level $level")
			for controller in policy.controllers[level]
				println("\tFrame $(frametype(controller))")
				print_solver_stats(controller.stats)
			end
		end
	end

	# """
	# function to create a custom initial belief, used to find which nodes of controller at level L-1
	# correspond to nodes at level L.
	# node = "node_id" -> find the node of level L corresponding to node of level L-1 with node.id = node_id
	# node = "any" -> find the node of level L that is visited when there's a possibility to be in any node of L-1
	# node = "all" -> find all nodes corresponding to level L
	# """
	# function create_initial_belief(policy::IBPIPolicy, controller_i_level::Int64, controller_i_index::Int64, controller_j_index::Int64; node = "any") where {S, A, W}
	# 	controller = policy.controllers[controller_i_level][controller_i_index]
	# 	controllers_j = policy.controllers[controller_i_level-1]
	# 	#compute at which index the nodes of controller j start in value vectors
	# 	controller_j_start_index = 0
	# 	for lower_controller_index in 1:controller_j_index-1
	# 		controller_j_start_index += length(controllers_j[lower_controller_index].nodes)
	# 	end
	# 	controller_j = policy.controllers[controller_i_level-1][controller_j_index]
	# 	#this will be used to iterate on all nodes without having to compute the end index
	# 	n_nodes_j = length(controller_j.nodes)
	# 	n_states = IPOMDPS.n_states(controller.frame)
	# 	anynode = controller.nodes[1]
	# 	initial = zeros(length(anynode_j.value))
	# 	initial_reshaped = reshape(initial, n_states, length(initial)/n_states)
	# 	if node == "any"
	# 		normalize = n_nodes_j * n_states
	# 		for state in 1:n_states
	# 			for node_j in 1:n_nodes:j
	# 				initial_reshaped = 1/normalize
	# 			end
	# 		end
	# 	end
	# end
	include("./simulator.jl")


	function compute_nodes_time(policy::IBPIPolicy)
		#sum together the length of all controllers in the same level
		#output running time average of one eval -> improv -> escape iteration
		for level in 1:policy.maxlevel
			time_nodes = Array{Tuple{Float64, Int64, Int64}}(undef, 0)
			for i in 1:length(policy.controllers[level][1].stats.data)
				sumnodes = 0
				time = 0.0
				maxmem = 0
				for controller in policy.controllers[level]
					maxmem = max(maxmem, controller.stats.data[i][3])
					sumnodes += controller.stats.data[i][2]
					time = max(time, controller.stats.data[i][1])
				end
				if length(time_nodes) == 0 || sumnodes != last(time_nodes)[2]
					push!(time_nodes, (time, sumnodes, maxmem))
				end
			end
			for t_n_m in time_nodes
				println("level $level $(t_n_m[1]): $(t_n_m[2]) $(t_n_m[3]/8000) kB")
			end
		end
	end

# """
# Print solver_stats to file in a latex readable .dat file
# """
# function print_stats_to_file(policy::IBPIPolicy, file_path::String)
#  	stats= policy.controllers[policy.maxlevel][1].stats
# 	len = length(stats.data[1])
# 	time = stats.data[1]
# 	nodes = stats.data[2]
# 	mem = stats.data[3]
# 	open("$file_path.dat", "w") do f
# 		write(f, "t\tn\tm\n")
# 		for i in
# 			t = time[i]
# 			n = nodes[i]
# 			m = mem[i]
# 			write(f, "$t\t$n\t$m\n")
# 		end
# 	end
# end

"""
Prints stats to screen so its easy to copy and paste them in case input from files is bugged
"""

# function print_stats_coordinates(policy::IBPIPolicy, window::Int64)
#  	stats= policy.controllers[policy.maxlevel][1].stats
# 	len = length(stats.data)
# 	time = stats.data
# 	nodes = stats.data
# 	mem = stats.data
# 	println("time, nodes")
# 	for i in 1:len
# 		t = round(time[i][1]; digits = 3)
# 		n = nodes[i][2]
# 		m = round(mem[i][3]/8000; digits = 3)
# 		print("($t, $n)")
# 	end
# 	println()
# 	println("time, memory")
# 	m_total = 0.0
# 	for i in 1:len
# 		t = round(time[i][1]; digits = 3)
# 		n = nodes[i][2]
# 		m = mem[i][3]/8000
# 		m_total = m_total + m
# 		if i > window + 1
# 			m_total -= (mem[i-window][3] / 8000)
# 		end
# 		if i > window + 1
# 			m_avg = round(m_total/window; digits = 3)
# 		else
# 			m_avg = round(m_total / i; digits = 3)
# 		end
# 		print("($t, $m_avg)")
# 	end
# 	println()
# end


function print_time_node_coordinates(policy::IBPIPolicy; fix = false)
	#sum together the length of all controllers in the same level
	#output running time average of one eval -> improv -> escape iteration
		time_nodes = Array{Tuple{Float64, Int64, Int64}}(undef, 0)
		controller = policy.controllers[policy.maxlevel][1]
		oldtime = 0
		starttime = 0
		for i in 1:length(controller.stats.data)
			maxmem = controller.stats.data[i][3]
			sumnodes = controller.stats.data[i][2]
			time = controller.stats.data[i][1]
			if fix
				time += starttime
			end
			if time < oldtime
				starttime = oldtime
			end
			oldtime = time
			if length(time_nodes) == 0 || sumnodes != last(time_nodes)[2]
				push!(time_nodes, (time, sumnodes, maxmem))
			end
		end
		for t_n_m in time_nodes
			print("($(t_n_m[1]), $(t_n_m[2]))")
		end
end

function print_time_value_coordinates(policy_name::String, agent_j_index::Int64, simreps::Int64, maxsimsteps::Int64, timestep::Int64, ntimesteps:: Int64)
	coords = Array{Tuple{Float64, Float64}}(undef, 0)
	for ts in 1:ntimesteps
		policy = load_policy(policy_name, 1, ts*timestep)
		if policy == nothing
			continue
		end
		avg_value = 0.0
		for rep in 1:simreps
			value, agent_i, agent_j, i = IBPIsimulate(policy.controllers[policy.maxlevel][1],  policy.controllers[policy.maxlevel-1][agent_j_index], maxsimsteps)
			avg_value += value
		end
		avg_value /= simreps
		time = last( policy.controllers[policy.maxlevel][1].stats.data)[1]
		push!(coords, (time, avg_value))
	end
	for coord in coords
		print("($(coord[1]), $(coord[2]))")
	end

end

function print_stats_coordinates(policy_name::String, agent_j_index::Int64, solverreps::Int64, simreps::Int64, maxsimsteps::Int64, timestep::Int64, ntimesteps:: Int64)
	#time, node, mem, value
	coords = Array{Tuple{Float64, Int64, Float64, Float64}}(undef, 0)
	old_stats_end_index = 1
	for ts in 1:ntimesteps
		policy = load_policy(policy_name, solverreps, ts*timestep)
		if policy == nothing
			continue
		end
		controller_i = policy.controllers[policy.maxlevel][1]
		avg_value = 0.0
		for rep in 1:simreps
			value, agent_i, agent_j, i = IBPIsimulate(policy.controllers[policy.maxlevel][1],  policy.controllers[policy.maxlevel-1][agent_j_index], maxsimsteps)
			avg_value += value
		end
		avg_value /= simreps
		nodes = length(controller_i.nodes)
		time = trunc(last( controller_i.stats.data)[1]; digits = 3)
		mem =  maximum(trunc(controller_i.stats.data[t][3]/8000; digits = 3) for t in old_stats_end_index:length(controller_i.stats.data))
		old_stats_end_index = length(controller_i.stats.data)
		push!(coords, (time, nodes, mem, avg_value))
	end
	#touch("/savedcontrollers/$(policy.name)/rep$solverreps/$(policy.name)_stats.policystats")
	open("savedcontrollers/$(policy.name)/rep$solverreps/$(policy.name)_stats.policystats", "w") do f

		write(f,"time, nodes\n")
		for coord in coords
			write(f, "($(coord[1]), $(coord[2]))")
		end
		write(f,"\ntime, mem\n")
		for coord in coords
			write(f, "($(coord[1]), $(coord[3]))")
		end
		write(f,"\nnodes, mem\n")
		for coord in coords
			write(f, "($(coord[2]), $(coord[3]))")
		end
		write(f,"\ntime, avg_val\n")
		for coord in coords
			write(f, "($(coord[1]), $(coord[4]))")
		end
		write(f, "\nnodes, avg_val\n")
		for coord in coords
			write(f, "($(coord[2]), $(coord[4]))")
		end
	end


end
