

#include("bpigraph.jl")
using Random
mutable struct IBPIAgent
	controller::AbstractController
	current_node::Node
	value::Float64
	stats::agent_stats
	visited::Array{Int64}
	history::Array{Any, 2}
end
function IBPIAgent(controller::AbstractController, initial_belief::Array{Float64})
	best_node, best_value = get_best_node(initial_belief, controller.nodes)
	return IBPIAgent(controller, best_node, 0.0, agent_stats(), zeros(Int64, length(controller.nodes)), Array{Any}(undef, 0, 5))
end
function best_action(agent::IBPIAgent)
	return chooseWithProbability(agent.current_node.actionProb)
end
function update_agent!(agent::IBPIAgent, action::A, observation::W) where {A, W}
	@deb("current node", :update)
	@deb(agent.current_node, :update)
	new_node_id = chooseWithProbability(agent.current_node.edges[action][observation])
	agent.current_node = agent.controller.nodes[new_node_id]
	agent.visited[agent.current_node.id]+=1
	return agent.current_node.id
end
function compute_s_prime(state::S, ai::A, aj::A, frame::IPOMDP; script = nothing) where {S, A}
	if script != nothing && script != :rand
		return script
	end
	dist = IPOMDPs.transition(frame, state, ai, aj)
	@deb("state = $state ai=$ai aj = $aj: $(dist.vals) $(dist.probs) ", :simprob)
	rn = RandomDevice()
	return POMDPModelTools.rand(rn, dist)
end

function compute_s_prime(state::S, ai::A, aj::A, frame::POMDP; script = nothing) where {S, A}
	if script != nothing && script != :rand
		return script
	end
	dist = POMDPs.transition(frame, state, ai)
	@deb("state = $state ai=$ai aj = $aj: $(dist.vals) $(dist.probs) ", :simprob)

	rn = RandomDevice()
	return POMDPModelTools.rand(rn, dist)
end

function compute_observation(s_prime::S, ai::A, aj::A, frame::IPOMDP; script = nothing) where {S, A}
	if script != nothing && script != :rand
		return script
	end
	dist = IPOMDPs.observation(frame, s_prime, ai, aj)
	@deb("state = $s_prime ai=$ai aj = $aj: $(dist.vals) $(dist.probs) ", :simprob)

	rn = RandomDevice()
	return POMDPModelTools.rand(rn, dist)
end

function compute_observation(s_prime::S, ai::A, aj::A, frame::POMDP; script = nothing) where {S, A}
	if script != nothing && script != :rand
		return script
	end
	dist = POMDPs.observation(frame, s_prime, ai)
	@deb("state = $s_prime ai=$ai aj = $aj: $(dist.vals) $(dist.probs)t ", :simprob)
	rn = RandomDevice()
	return POMDPModelTools.rand(rn, dist)
	# items = dist.probs
	# randn = rand() #number in [0, 1)
	# for i in 1:length(dist.vals)
	# 	if randn <= items[i]
	# 		return dist.vals[i]
	# 	else
	# 		randn-= items[i]
	# 	end
	# end
	# error("Out of dict bounds while choosing items")
end



function IBPIsimulate(controller_i::InteractiveController{S, A, W}, controller_j::AbstractController, maxsteps::Int64; trace=false, scenario = nothing) where {S, A, W}
	correlation = zeros(Int64, length(controller_i.nodes), length(controller_j.nodes))


	frame_i = controller_i.frame
	anynode = controller_i.nodes[1]
	initial = ones(length(anynode.value))
	initial = initial ./ length(initial)
	agent_i = IBPIAgent(controller_i, initial)

	frame_j = controller_j.frame
	anynode_j = controller_j.nodes[1]
	initial_j = ones(length(anynode_j.value))
	initial_j = initial_j ./ length(initial_j)
	agent_j = IBPIAgent(controller_j, initial_j)

	if trace
		println("Starting node for I:")
		println(agent_i.current_node)
		println("Starting node for J:")
		println(agent_j.current_node)
	end
	if scenario != nothing
		state = scenario.script[1][1]
	else
		state = randn() > 0.5 ? :TL : :TR

	end
	value = 0.0
	value_j = 0.0
	if scenario != nothing
		maxsteps = size(scenario.script, 1)
	end
	if !trace && maxsteps >= 100
		for i in 1:95
			print(" ")
		end
		println("end v")
	end

	#1 -> state 2-> action_i 3 -> obs_i 4 -> node 5 -> value
	agent_i.history = Array{Any}(undef, maxsteps, 5)
	agent_j.history = Array{Any}(undef, maxsteps, 5)
	for i in 1:maxsteps
		if !trace && maxsteps >= 100 && i % (maxsteps/100) == 0
			print("|")
		end
		ai = best_action(agent_i)
		aj = best_action(agent_j)
		if trace
			println("$i -> state: $state -> ai: $ai, aj: $aj")
		end
		value =  IPOMDPs.discount(frame_i) * value + IPOMDPs.reward(frame_i, state, ai, aj)

		if trace
			println("\tvalue this step: $(IPOMDPs.reward(frame_i, state, ai, aj))")
		end
		if scenario != nothing
			if i != maxsteps
				s_prime = compute_s_prime(state, ai, aj, frame_i; script = scenario.script[i+1,1])
			else
				s_prime = compute_s_prime(state, ai, aj, frame_i)
			end

			zi = compute_observation(s_prime, ai, aj, frame_i; script = scenario.script[i,2])
			zj = compute_observation(s_prime, aj, ai, frame_j; script = scenario.script[i,3])
		else
			s_prime = compute_s_prime(state, ai, aj, frame_i)

			zi = compute_observation(s_prime, ai, aj, frame_i)
			zj = compute_observation(s_prime, aj, ai, frame_j)
		end
		if trace
			println("\tzi -> $zi, zj -> $zj")
		end
		np_i = update_agent!(agent_i, ai, zi)
		np_j = update_agent!(agent_j, aj, zj)

		if trace
			println("\tnew current node for I: $np_i")
			@deb(agent_i.current_node, :simverb)
			println("\tnew current node for j: $np_j")
			@deb(agent_j.current_node, :simverb)
		end
		correlation[np_i, np_j] += 1
		computestats!(agent_i.stats, ai, aj, state, s_prime, zi, zj)
		computestats!(agent_j.stats, aj, ai, state, s_prime, zj, zi)
		agent_i.history[i,1] = state
		agent_i.history[i,2] = ai
		agent_i.history[i,3] = zi
		agent_i.history[i,4] = np_i
		agent_i.history[i,5] = value

		agent_j.history[i,1] = state
		agent_j.history[i,2] = aj
		agent_j.history[i,3] = zj
		agent_j.history[i,4] = np_j
		agent_j.history[i,5] = value

		state = s_prime
	end
	#println()
	#analyze_history(i_history, j_history)
	#analyze_correlation(correlation, controller_i, controller_j)
	return value, agent_i, agent_j, correlation
end


function analyze_correlation(correlation::Array{Int64, 2}, controller::InteractiveController{S, A, W}, controller_j:: AbstractController) where {S, A, W}
	for nj in 1:length(controller_j.nodes)
		maxCorr = maximum(correlation[:, nj])
		for ni in 1:length(controller.nodes)
			if correlation[ni, nj] > 0.5 * maxCorr
				println("$ni -> $nj = $(correlation[ni,nj])" )
			end
		end
	end

end
function get_avg_sim_value(controller_i::AbstractController, controller_j::AbstractController, maxsimsteps::Int64, simreps::Int64 )
	avg_value = zeros(Float64, maxsimsteps)
	for simrep in 1:simreps
		@deb("rep $simrep", :simrep)
		value, agent_i, agent_j = IBPIsimulate(controller_i, controller_j, maxsimsteps)
		avg_value = avg_value .+ agent_i.history[:, 5]
	end
	avg_value = avg_value ./ simreps
	@deb(avg_value, :simvalue)

	return avg_value
end
function compute_sim_step_values(policy_name::String, rep::Int64, backup_n::Int64, maxsimsteps::Int64, simreps::Int64)
	policy = load_policy(policy_name, rep, backup_n)
	#consider maxlevel controller_j
	@deb(policy.name, :simvalue)
	controller_i = policy.controllers[policy.maxlevel][1]

	for agent_j_index in  1:length(policy.controllers[policy.maxlevel-1])
		controller_j = policy.controllers[policy.maxlevel-1][agent_j_index]
		avg_value = get_avg_sim_value(controller_i, controller_j, maxsimsteps, simreps)
		open("savedcontrollers/$(policy_name)/rep$rep/$(policy_name)_agent$(agent_j_index).simvalue", "w") do f
			for i in 1:length(avg_value)
				v = trunc(avg_value[i]; digits = 3)
				write(f, "($i,$v)")
			end
		end
	end
end

function compute_level_values(policy_name::String, rep::Int64, backup_n::Int64, maxsimsteps::Int64, simreps::Int64)
	policy = load_policy(policy_name, rep, backup_n)
	#consider maxlevel controller_j
	@deb(policy.name, :simvalue)
	controller_j = policy.controllers[policy.maxlevel][1]
	for i_level in 2:policy.maxlevel
		controller_i = policy.controllers[i_level][1]
		avg_value = get_avg_sim_value(controller_i, controller_j, maxsimsteps, simreps)
		open("savedcontrollers/$(policy_name)/rep$rep/$(policy_name)_level$i_level.simvalue", "w") do f
			for i in 1:length(avg_value)
				v = trunc(avg_value[i]; digits = 3)
				write(f, "($i,$(v))")
			end
		end
	end
end

function analyze_history(i_history::Array{Any, 2}, j_history::Array{Any, 2})

	open_count = zeros(Int64, 2)
	open_diff = zeros(Int64, 2)
	#i is 1, j is 2
	last_opened = 1
	correct_obs = 0
	wrong_obs = 0

	for i in 1:size(i_history, 1)
		action = i_history[i, 2]
		state = i_history[i, 1]
		obs = i_history[i, 3]
		@deb("$action $state $obs", :i_history)


		if action == :L
			#if there's silence
			if obs == :GLS || obs == :GRS
				if correct_result(state, obs)
					correct_obs += 1
				else
					wrong_obs += 1
				end


			else
				#if the other agent opened (from obs)
				#set j as last opened
				last_opened = 2
				correct_obs = 0
				wrong_obs = 0
			end
		else
			#if it opened
			@deb("correct = $correct_obs, wrong = $wrong_obs, last = $last_opened", :i_history)

			open_count[last_opened] += 1
			open_diff[last_opened] += correct_obs - wrong_obs

			#set i as last opened
			last_opened = 1
			correct_obs = 0
			wrong_obs = 0
		end
	end

	println("Average obs difference after I opens: $(open_diff[1]/open_count[1])")
	println("Average obs difference after J opens: $(open_diff[2]/open_count[2])")

end

function correct_result(state::Symbol, other::Symbol)
	if state == :TL
		return other == :GLS || other == :GLCL || other == :GLCR || other == :OR || other == :GL
	else
		return other == :GRS || other == :GRCL || other == :GRCR || other == :OL || other == :GR
	end
end




#scenarios
struct Scenario
	name::String
	#State, zi, zj
	script::Array{Symbol, 2}
end
#TODO return scenarios with swapped script, so it is more sane when they are used (script[action, timestep])
function standard_scenario()
	script = [
				:TR :GRS :GR;
				:TR :GRS :GR;
				:TR :rand :rand;
				:TL :GLS :GL;
				:TL :GLS :GL;
				:TL :rand :rand
			 ]
	return Scenario("1_standard", script)
end

function immediate_false_creak()
	script = [
				:TR :GRCL :GR;
				:TR :GRS :GR;
				:TR :GRS :GR;
				:TL :GLS :GL;
				:TL :GLS :GL;
				:TL :rand :rand
			 ]
	return Scenario("2_immediate_false_creak", script)
end
function opposite_side_creak_after_one()
	script = [
				:TR :GRS :GR;
				:TR :GRCL :GL;
				:TR :GRS :GR;
				:TR :GRS :GR;
				:TL :GLS :GL;
				:TL :GLS :GL;
				:TL :rand :rand;

			 ]
	return Scenario("3a_opposite_side_creak_after_one", script)
end

function same_side_creak_after()
	script = [
				:TR :GRS :GR;
				:TR :GRCR :GL;
				:TR :GRS :GR;
				:TR :GRS :GR;
				:TL :GLS :GL;
				:TL :GLS :GL;
				:TL :rand :rand;

			 ]
	return Scenario("3b_same_side_creak_after_one", script)
end



function opposite_creak_opposite_growl_after_one()
	script = [
				:TR :GRS :GR;
				:TR :GLCL :GL;
				:TR :GRS :GR;
				:TR :GRS :GR;
				:TR :GRS :GR;
				:TL :GLS :GL;
				:TL :rand :rand;

			 ]
	return Scenario("3c_opposite_creak_opposite_growl_after_one", script)
end
function true_creak()
	script = [
				:TR :GRS :GR;
				:TR :GLS :GR;
				:TR :GLCL :rand;
				:TL :GLS :GL;
				:TL :GLS :GL;
				:TL :rand :rand;

			 ]
	return Scenario("4_true_creak", script)
end
function late_agent_j_scenario()
	script = [
				:TR :GLS :GR;
				:TR :GRS :GR;
				:TL :GLCL :rand;
				:TL :GLS :GL;
				:TR :GLS :GL;
				:TL :GLS :rand;
				:TL :rand :rand;
			 ]
	return Scenario("5_late_agent_j", script)
end
function all_scenarios()
	return [standard_scenario(), immediate_false_creak(),  opposite_side_creak_after_one(), same_side_creak_after(),opposite_creak_opposite_growl_after_one(),true_creak(), late_agent_j_scenario()]
end
function collect_scenario_stats(controller_i::AbstractController, controller_j::AbstractController, scenario::Scenario, maxsimsteps::Int64, simreps::Int64)
	occ = Dict{Vector{Tuple{Symbol, Symbol}}, Int64}()

	for simrep in 1:simreps
		value, agent_i, agent_j = IBPIsimulate(controller_i, controller_j, maxsimsteps; scenario = scenario)
		agent_i_actions = agent_i.history[:, 2]
		agent_j_actions = agent_j.history[:, 2]
		joint_actions = Vector{Tuple{Symbol, Symbol}}(undef, length(agent_i_actions))
		for i in 1:length(joint_actions)
			joint_actions[i] = (agent_i_actions[i], agent_j_actions[i])
		end
		if !haskey(occ, joint_actions)
			occ[joint_actions] = 1
		else
			occ[joint_actions] += 1
		end
	end
	sorted =  sort(collect(occ), by=x->x[2]; rev=true)
	@deb(sorted, :scenario)
	return sorted
end

function save_scenario_stats(policy::IBPIPolicy, scenario::Scenario, agent_j_index::Int64, maxsimsteps::Int64, simreps::Int64; mode= "w")
	controller_i = policy.controllers[policy.maxlevel][1]
	controller_j = policy.controllers[policy.maxlevel-1][agent_j_index]

	sorted_responses = collect_scenario_stats(controller_i, controller_j, scenario, maxsimsteps, simreps)
	@deb(sorted_responses, :scenario)
	path = "savedstats/scenario/$(scenario.name)"
	if(!isdir(path))
		mkdir(path)
	end
	# path *= "/$(policy.name)"
	# if(!isdir(path))
	# 	mkdir(path)
	# end
	open("$path/$(scenario.name).scenariostats", mode) do f
		if mode == "w"

			#steps line
			line = "Step &"
			for step in 1:size(scenario.script, 1)
				line *= "$step &"
			end
			line *= "\\%\\\\\n"
			@deb(line, :scenario)

			write(f, line)
			#state line
			line = "State &"
			for state in scenario.script[:, 1]
				if state == :rand
					state = :Any
				end
				line *= "$state &"
			end
			line *= "\\\\\n"
			@deb(line, :scenario)

			write(f, line)
			#obs_i line
			line = "Obs_i &"
			for obs in scenario.script[:, 2]
				if obs == :rand
					obs = :Any
				end
				line *= "$obs &"
			end
			line *= "\\\\\n"
			@deb(line, :scenario)

			write(f, line)

			#obs_i line
			line = "Obs_j &"
			for obs in scenario.script[:, 3]
				if obs == :rand
					obs = :Any
				end
				line *= "$obs &"
			end
			line *= "\\\\\n"
			@deb(line, :scenario)

			write(f, line)
		end

		header = true
		for resp in sorted_responses
			if header
				line = policy.name * " &"
				header = false
			else
				line = "&"
			end

			perc = trunc(resp[2]/simreps; digits = 3)
			for a in resp[1]
				line *=  "$(a[1]), $(a[2]) &"
			end
			line *= "$(perc) \\\\\n"
			@deb(line, :scenario)
			write(f, line)
		end
	end

end
