

#include("bpigraph.jl")

mutable struct IBPIAgent
	controller::AbstractController
	current_node::Node
	value::Float64
	stats::agent_stats
	visited::Array{Int64}
end
function IBPIAgent(controller::AbstractController, initial_belief::Array{Float64})
	best_node, best_value = get_best_node(initial_belief, controller.nodes)
	return IBPIAgent(controller, best_node, 0.0, agent_stats(), zeros(Int64, length(controller.nodes)))
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
	dist = IPOMDPs.observation(frame, s_prime, ai, aj)
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

function IBPIsimulate(controller_i::InteractiveController{S, A, W}, controller_j::AbstractController, maxsteps::Int64; trace=false) where {S, A, W}
	correlation = zeros(Int64, length(controller_i.nodes), length(controller_j.nodes))
	#1 -> state 2-> action 3 -> obs
	i_history = Array{Symbol}(undef, maxsteps, 3)
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
	state = randn() > 0.5 ? :TL : :TR
	value = 0.0
	if !trace
		for i in 1:95
			print(" ")
		end
		println("end v")
	end
	for i in 1:maxsteps
		if i % (maxsteps/100) == 0 && !trace
			print("|")
		end
		ai = best_action(agent_i)
		aj = best_action(agent_j)
		if trace
			println("state: $state -> ai: $ai, aj: $aj")
		end
		value =  IPOMDPs.discount(frame_i) * value + IPOMDPs.reward(frame_i, state, ai, aj)
		if trace
			println("value this step: $(IPOMDPs.reward(frame_i, state, ai, aj))")
		end
		s_prime = compute_s_prime(state, ai, aj, frame_i)

		zi = compute_observation(s_prime, ai, aj, frame_i)
		zj = compute_observation(s_prime, aj, ai, frame_j)
		if trace
			println("zi -> $zi, zj -> $zj")
		end
		np_i = update_agent!(agent_i, ai, zi)
		np_j = update_agent!(agent_j, aj, zj)

		if trace
			println("new current node for I: $np_i")
			@deb(agent_i.current_node, :sim)
			println("new current node for j: $np_j")
			@deb(agent_j.current_node, :sim)
		end
		correlation[np_i, np_j] += 1
		computestats!(agent_i.stats, ai, aj, state, s_prime, zi, zj)
		computestats!(agent_j.stats, aj, ai, state, s_prime, zj, zi)
		i_history[i,1] = state
		i_history[i,2] = ai
		i_history[i,3] = zi
		state = s_prime
	end
	println()
	analyze_history(i_history)
	#analyze_correlation(correlation, controller_i, controller_j)
	return value, agent_i, agent_j, i_history
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

function analyze_history(history::Array{Symbol, 2})

	open_count = zeros(Int64, 2)
	open_diff = zeros(Int64, 2)
	#i is 1, j is 2
	last_opened = 1
	correct_obs = 0
	wrong_obs = 0

	for i in 1:size(history, 1)
		action = history[i, 2]
		state = history[i, 1]
		obs = history[i, 3]
		@deb("$action $state $obs", :history)


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
			@deb("correct = $correct_obs, wrong = $wrong_obs, last = $last_opened", :history)

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
