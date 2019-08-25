#=
IBPIPolicyUtils:
- Julia version: 1.1.0
- Author: Riccardo Ficarra
- Date: 2019-02-11
=#
	using LinearAlgebra
	using DataStructures
	using JuMP
	using CPLEX
	using SparseArrays

	"""
	Basic data structure for controllers.

	"""

	struct Node{A, W}
		id::Int64
		actionProb::Dict{A, Float64}
		#action -> observation -> node_id -> prob
		edges::Dict{A, Dict{W, Vector{Pair{Int64, Float64}}}}
		value::Array{Float64}
		#needed to efficiently redirect edges during pruning
		#srcNode -> vectors of dictionaries that contains edge to this node
		#incomingEdgeDicts::Dict{Int64, Vector{Vector{Pair{Int64, Float64}}}}
	end

	Base.isequal(n1::Node, n2::Node) = Base.isequal(hash(n1), hash(n2))
	#overload display function to avoid walls of text when printing nodes
	Base.display(n::Node) = println(n)


	function Base.println(node::Node)
		for (a, a_prob) in node.actionProb
			obs = node.edges[a]
			for (obs, edges) in obs
				for (next_id, prob) in edges
					println("node_id=$(node.id), a=$a, $a_prob, $obs -> $(next_id), $prob")
				end
			end
		end
		#=
		if :data in debug
			for (src_node, dict_vect) in node.incomingEdgeDicts
				for dict in dict_vect
					for (next, prob) in dict
						println("from node $(src_node.id) p=$(prob) to node $(next.id)")
					end
				end
			end
		end
		=#
		println("Value vector = $(node.value)")
	end



	"""
	Receives the pomdp frame
	Builds a node with a random action chosen and with all observation edges
	pointing back to itself
	If force is defined the action is decided by the user (by index in 1:length(actions))
	"""
	# function InitialNode(pomdp::POMDP{A, W}; force = 0) where {A, W}
	# 	actions = POMDPs.actions(pomdp)
	# 	observations = POMDPs.observations(pomdp)
	#
	# 	actions_vector = []
	# 	if !(pomdp <: randomTiger)
	# 		if force == 0
	# 			actionindex = rand(1:length(actions))
	# 		else
	# 			if force > length(actions)
	# 				error("forced action outside of action vector length")
	# 			end
	# 			actionindex = force
	# 		end
	# 		actions_vector = [actions[actionindex]]
	# 	else
	# 		actions_vector = actions.vals
	# 	end
	# 	n = Node(1, [actions[actionindex]], observations)
	# 	obsdict = Dict{W, Vector{Pair{Int64, Float64}}}()
	# 	for a in keys(n.actionProb)
	# 		for obs in observations
	# 			edges = [(1 => 1.0)]
	# 			obsdict[obs] = edges
	# 		end
	# 		n.edges[a] = obsdict
	# 	return n
	# end
	function InitialNode(actions::Vector{A}, observations::Vector{W}) where {S, A, W}
		#build actionProb
		actionProb = Dict{A, Float64}()
		for i in 1:length(actions)
			actionProb[actions[i]] = 1/length(actions)
		end

		#build edges
		edges = Dict{A,Dict{W,Vector{Pair{Int64,Float64}}}}();

		for a in keys(actionProb)
			#for each action and observation add a loop edge
			obsdict = Dict{W, Vector{Pair{Int64, Float64}}}()
			for obs in observations
				obsdict[obs] = [(1 => 1.0)]
			end
			edges[a] = obsdict
		end
		return Node(1, actionProb, edges, Array{Float64}(undef, 0))
	end

	#interface function for pomdps
	function InitialNode(pomdp::POMDP{A, W}; force = 0) where {A, W}
		actions = POMDPs.actions(pomdp)
		observations = POMDPs.observations(pomdp)

		if typeof(pomdp) <: randomTiger
			return InitialNode(actions, observations)
		else
			if force == 0
				return InitialNode([random(actions)], observations)
			else
				if force > length(actions)
					error("Forced action outside of action vector bounds")
				else
					return InitialNode([actions[force]], observations)
				end
			end
		end
	end

	function checkNode(node::Node{A, W}, controller::AbstractController; normalize = false, checkDistinct = false) where {A, W}
		return
		obs_list = observations(controller.frame)
		#check actionProb
		tot = 0.0
		for (action, prob) in node.actionProb
			tot+= prob
		end
		if normalize && tot != 1.0
			@deb("Normalizing $tot", :checkNodes)
			for (action,prob) in node.actionProb
				node.actionProb[action] = prob/tot
			end
		end
		if !normalize && tot <= 1.0-config.minval || tot >= 1.0+config.minval
			error("Sum of actionProb == $tot")
		end

		#check edges
		for (action, obs_dict) in node.edges
			#check that all observations are there
			for obs in obs_list
				if !haskey(obs_dict, obs)
					error("Missing observation $obs")
				end
			end
			for (obs, next_dict) in obs_dict
				tot = 0.0
				for (next_id, prob_next) in next_dict
					if next_id > length(controller.nodes)
						error("Node $(next_id) not present in controller")
					end
					tot+= prob_next
				end
				if normalize && tot != 1.0
					@deb("Normalizing edges $tot", :checkNodes)
					for i in 1:length(next_dict)
						next_dict[i] = (next_dict[i][1] => next_dict[i][2]/tot)
					end
				end
				if !normalize && (tot <= 1.0-config.minval || tot >= 1.0+config.minval)
					error("Sum of edges  == $tot")
				end

			end
		end
		if checkDistinct
			for other_node in controller.nodes
				if node.id != other_node.id && nodeequal(node, other_node) && nodeequal(other_node, node)
					println("new node:")
					println(node)
					println("old node:")
					println(other_node)
					@warn("New node is already present as $(other_node.id)")
				end
			end
		end
	end

	function nodeequal(a::Node{A, W}, b::Node{A, W}) where {A, W}

		# for i in 1:length(a.value)
		# 	if abs(a.value[i] - b.value[i]) > config.minval
		# 		return false
		# 	end
		# end
		# @warn("Value vector of $(a.id) and $(b.id) are equal")

		for (action, prob) in a.actionProb
			if !haskey(b.actionProb, action) || (b.actionProb[action] - prob) > config.minval
				return false
			end
		end
		#disabled after tuple change
		#actions are the same
		# for (action, action_dict) in a.edges
		# 	for (obs, obs_dict) in action_dict
		# 		for (next_id, prob) in obs_dict
		# 			b_obs_dict = b.edges[action][obs]
		# 			if !haskey(b_obs_dict, next_id) || abs(b_obs_dict[next_id] - prob) > minval
		# 				return false
		# 			end
		# 		end
		# 	end
		# end
		return true
	end

	"""
	Randomly choose an action based on action probability given a node
	returns action::A
	"""
	function getAction(node::Node{A, W}) where {A, W}
		action = chooseWithProbability(node.actionProb)
		@deb("Chosen action $action")
		return action
	end


	"""
	Given a dictionary in the form item => probability
	Pick a random item based on the probability.
	probability must sum to 1.
	O(n)
	"""
	function chooseWithProbability(items::Dict)
		randn = rand() #number in [0, 1)
		for i in keys(items)
			@deb(i, :update)
			if randn <= items[i]
				return i
			else
				randn-= items[i]
			end
		end
		return last(collect(keys(items)))
	end

	function chooseWithProbability(items::Vector{Pair{Int64, Float64}})
		randn = rand() #number in [0, 1)
		for i in items
			@deb(i, :update)
			if randn <= i[2]
				return i[1]
			else
				randn-= i[2]
			end
		end
		#if out of bounds
		return last(items)[1]
	end

	"""
	Data structure for non-interactive controllers. MaxID is the highest ID currently in the controller.
	"""
	mutable struct Controller{A, W} <: AbstractController
		#level::Int64
		frame::POMDP{A, W}
		nodes::Vector{Node{A, W}}
		stats::solver_statistics
		converged::Bool
	end
	"""
	Initialize a controller with only one standard initial node
	"""
	function Controller(pomdp::POMDP{A,W}; force=0) where {A, W}

		newNode = InitialNode(pomdp; force = force)
		if typeof(pomdp) <: randomTiger
			#this way it's going to be skipped
			Controller{A, W}(pomdp, [newNode], solver_statistics(), true)
		else
			Controller{A, W}(pomdp, [newNode], solver_statistics(), false)
		end
	end

	function checkController(controller::AbstractController; checkDistinct = false)
		for node in controller.nodes
			checkNode(node, controller; checkDistinct = false)
		end
	end
	"""
	Hardcoded optimal tiger controller from Kaelbling's paper
	Currently not working for some reason
	"""
	function optimal_tiger_controller(pomdp::POMDP{A, W}) where {A, W}
		# controller = Controller(pomdp; force = 3)
		# controller.nodes[1].id = 1
		# #create the open left(2)- open right(3) nodes
		# controller.nodes[3] = InitialNode(pomdp; force = 1)
		# controller.nodes[3].id = 3
		# controller.nodes[2] = InitialNode(pomdp; force = 2)
		# controller.nodes[2].id = 2
		# for i in 5:10
		# 	controller.nodes[i] = InitialNode(pomdp; force = 3)
		# 	controller.nodes[i].id = i
		# end
		# controller.nodes[1].edges[:L][:GL] = Dict(controller.nodes[5] => 1.0)
		# controller.nodes[1].edges[:L][:GR] = Dict(controller.nodes[6] => 1.0)
		# controller.nodes[2].edges[:OR][:GL] = Dict(controller.nodes[1] => 1.0)
		# controller.nodes[2].edges[:OR][:GR] = Dict(controller.nodes[1] => 1.0)
		# controller.nodes[3].edges[:OL][:GL] = Dict(controller.nodes[1] => 1.0)
		# controller.nodes[3].edges[:OL][:GR] = Dict(controller.nodes[1] => 1.0)
		# controller.nodes[5].edges[:L][:GL] = Dict(controller.nodes[2] => 1.0)
		# controller.nodes[5].edges[:L][:GR] = Dict(controller.nodes[1] => 1.0)
		# controller.nodes[6].edges[:L][:GL] = Dict(controller.nodes[1] => 1.0)
		# controller.nodes[6].edges[:L][:GR] = Dict(controller.nodes[3] => 1.0)
		# controller.nodes[7].edges[:L][:GL] = Dict(controller.nodes[10] => 1.0)
		# controller.nodes[7].edges[:L][:GR] = Dict(controller.nodes[8] => 1.0)
		# controller.nodes[8].edges[:L][:GL] = Dict(controller.nodes[7] => 1.0)
		# controller.nodes[8].edges[:L][:GR] = Dict(controller.nodes[9] => 1.0)
		# controller.nodes[9].edges[:L][:GL] = Dict(controller.nodes[8] => 1.0)
		# controller.nodes[9].edges[:L][:GR] = Dict(controller.nodes[3] => 1.0)
		# controller.nodes[10].edges[:L][:GL] = Dict(controller.nodes[2] => 1.0)
		# controller.nodes[10].edges[:L][:GR] = Dict(controller.nodes[7] => 1.0)
		# controller.maxId = 10
		# evaluate!(controller)
		# if :data in debug
		# 	println("Optimal controller for tiger game:")
		# 	for (node_id, node) in controller.nodes
		# 		println(node)
		# 	end
		# end
		nodes = Array{Node{A, W}, 1}(undef, 9)
		nodes[1] = Node(1, Dict(:L => 1.0), Dict(:L => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(4 => 1.0)], :GR => [(5 => 1.0)])), Array{Float64}(undef, 0, 0))
		nodes[2] = Node(2, Dict(:OR => 1.0), Dict(:OR => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(1 => 1.0)], :GR => [(1 => 1.0)])), Array{Float64}(undef, 0, 0))
		nodes[3] = Node(3, Dict(:OL => 1.0), Dict(:OL => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(1 => 1.0)], :GR => [(1 => 1.0)])), Array{Float64}(undef, 0, 0))
		nodes[4] = Node(4, Dict(:L => 1.0), Dict(:L => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(2 => 1.0)], :GR => [(1 => 1.0)])), Array{Float64}(undef, 0, 0))
		nodes[5] = Node(5, Dict(:L => 1.0), Dict(:L => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(1 => 1.0)], :GR => [(3 => 1.0)])), Array{Float64}(undef, 0, 0))
		nodes[6] = Node(6, Dict(:L => 1.0), Dict(:L => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(2 => 1.0)], :GR => [(7 => 1.0)])), Array{Float64}(undef, 0, 0))
		nodes[7] = Node(7, Dict(:L => 1.0), Dict(:L => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(6 => 1.0)], :GR => [(8 => 1.0)])), Array{Float64}(undef, 0, 0))
		nodes[8] = Node(8, Dict(:L => 1.0), Dict(:L => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(7 => 1.0)], :GR => [(9 => 1.0)])), Array{Float64}(undef, 0, 0))
		nodes[9] = Node(9, Dict(:L => 1.0), Dict(:L => Dict{W, Array{Pair{Int64, Float64}, 1}}(:GL =>[(8 => 1.0)], :GR => [(3 => 1.0)])), Array{Float64}(undef, 0, 0))

		controller = Controller(pomdp,nodes, solver_statistics(), true)
		evaluate!(controller)
		return controller
	end


	"""
	Returns the next node given current node, action and observations
	"""
	function get_next_node(node::Node{A, W}, action::A, observation::W, controller::AbstractController) where {A, W}
		if !haskey(node.edges, action)
			error("Action has probability 0!")
		end
		edges = node.edges[action][observation]
		next_id = chooseWithProbability(edges)
		@deb("Chosen $(next_id) as next node", :flow )
		return controller.nodes[next_id]
	end

	"""
	Wrapper data structure for a non-interactive policy.
	"""
	struct BPIPolicy{A, W}
		name::String
		controller::Controller{A, W}
	end
	"""
	Create a BPIPolicy with a standard initial controller.
	"""
	function BPIPolicy(name::String, pomdp::POMDP{A, W}; force=0) where {A, W}
		if force == 0
			BPIPolicy(name, Controller(pomdp))
		else
			BPIPolicy(name, Controller(pomdp; force = force))
		end
	end

	# function build_node(node_id::Int64, actions::Vector{A}, actionProb::Vector{Float64}, observations::Vector{Vector{W}}, observation_prob::Vector{Vector{Float64}}, next_nodes::Vector{Vector{Node{A,W}}}, value::Array{Float64}) where {A, W}
	# 	if length(actions) != length(observations) || length(actions) != length(actionProb) || length(actions) != length(observation_prob) || length(actions) != length(next_nodes)
	# 		error("Length of action-level arrays are different")
	# 	end
	# 	edges = Dict{A, Dict{W, Vector{Pair{Int64, Float64}}}}()
	# 	d_actionprob = Dict{A, Float64}()
	# 	for a_i in 1:length(actions)
	# 		action = actions[a_i]
	# 		#fill actionprob dict
	# 		d_actionprob[action] = actionProb[a_i]
	# 		#vector of observations tied to action
	# 		a_obs = observations[a_i]
	# 		a_obs_prob = observation_prob[a_i]
	# 		a_next_nodes = next_nodes[a_i]
	# 		if length(a_obs) != length(a_obs_prob) || length(a_obs) != length(a_next_nodes)
	# 			error("Length of observation-level arrays are different")
	# 		end
	# 		new_obs = Dict{W, Vector{Pair{Int64, Float64}}}()
	# 		for obs_index in 1:length(a_obs)
	# 			obs = a_obs[obs_index]
	# 			new_obs[obs] = Vector{Pair{Int64, Float64}}(a_next_nodes[obs_index] => a_obs_prob[obs_index])
	# 		end
	# 		edges[action] = new_obs
	# 	end
	# 	return Node(node_id, d_actionprob, edges, value, Dict{Int64, Vector{Vector{Pair{Int64, Float64}}}}())
	# end
	"""
	Builds a node with action specified, a single edge for the specified observation going to next_node, and value vector value.
	"""
	function build_node(node_id::Int64, action::A, observation::W, next_node::Node{A, W}, value::Array{Float64}) where {A, W}
		actionprob = Dict{A, Float64}(action => 1.0)
		edges = Dict{A, Dict{W, Vector{Pair{Int64, Float64}}}}(action => Dict{W, Vector{Pair{Int64, Float64}}}(observation => [(next_node.id => 1.0)]))
		return Node(node_id, actionprob, edges, value)
	end
	"""
	Computes all the possible new nodes that can be added to the controller using Incremental pruning with stochastic filtering.
	"""
	function full_backup_generate_nodes(controller::Controller{A, W}) where {A, W}
		pomdp = controller.frame
		nodes = controller.nodes
		observations = POMDPs.observations(pomdp)
		#tentative from incpruning
		#prder of it -> actions, obs
		#for each a, z produce n new nodes (iterate on nodes)
		#for each node iterate on s and s' to produce a new node
		#new node is tied to old node?, action a and obs z
		#with stochastic pruning we get the cis needed
		new_nodes = Set{Node}()
		#new nodes counter used mainly for debugging, counts backwards (gets overwritten eventually)
		new_nodes_counter = -1
		for a in POMDPs.actions(pomdp)
			#this data structure has the set of nodes for each observation (new_nodes_z[obs] = Set{Nodes} generated from obs)
			new_nodes_z = Vector{Set{Node}}(undef, length(observations))
			for obs_index in 1:length(observations)
				obs = observations[obs_index]
				#this set contains all new nodes for action, obs for all nodes
				new_nodes_a_z = Set{Node}()
				for (n_id, node) in controller.nodes
					new_v = node_value(node, a, obs, pomdp)
					#do not set node id for now
					#new_node = build_node(new_nodes_counter, [a], [1.0], [[obs]], [[1.0]], [[node]], new_v)
					new_node = build_node(new_nodes_counter, a, obs, node, new_v)
					push!(new_nodes_a_z, new_node)
					new_nodes_counter -=1
				end
				if :data in debug
					println("New nodes created:")
					for node in new_nodes_a_z
						println(node)
					end
				end
				new_nodes_z[obs_index] = filterNodes(new_nodes_a_z, config.minval)
			end
			#set that contains all nodes generated from action a after incremental pruning
			new_nodes_counter, new_nodes_a = incprune(new_nodes_z, new_nodes_counter, config.minval)
			union!(new_nodes, new_nodes_a)
		end
		#all new nodes, final filtering
		return filterNodes(new_nodes, config.minval)
	end
	# """
	# Perform a full backup operation according to Pourpart and Boutilier's paper on Bounded finite state controllers
	# """
	# function full_backup_stochastic!(controller::Controller{A, W}; minval=1e-10) where {A, W}
	# 	pomdp = controller.frame
	# 	nodes = controller.nodes
	# 	observations = POMDPs.observations(pomdp)
	# 	#tentative from incpruning
	# 	#prder of it -> actions, obs
	# 	#for each a, z produce n new nodes (iterate on nodes)
	# 	#for each node iterate on s and s' to produce a new node
	# 	#new node is tied to old node?, action a and obs z
	# 	#with stochastic pruning we get the cis needed
	# 	new_nodes = full_backup_generate_nodes(controller, minval)
	# 	#before performing filtering with the old nodes update incomingEdge structure of old nodes
	# 	#also assign permanent ids
	# 	nodes_counter = controller.maxId+1
	# 	for new_node in new_nodes
	# 		@deb("Node $(new_node.id) becomes node $(nodes_counter)", :data)
	# 		new_node.id = nodes_counter
	# 		if :data in debug
	# 			println(new_node)
	# 		end
	# 		nodes_counter+=1
	# 		for (action, observation_map) in new_node.edges
	# 			for (observation, edge_map) in observation_map
	# 				#@deb("Obs $observation")
	# 				for (next, prob) in edge_map
	# 					#@deb("adding incoming edge from $(new_node.id) to $(next.id) ($action, $observation)")
	# 					if haskey(next.incomingEdgeDicts, new_node)
	# 						#@deb("it was the $(length(next.incomingEdgeDicts[new_node])+1)th")
	# 						push!(next.incomingEdgeDicts[new_node], edge_map)
	# 					else
	# 						#@deb("it was the first edge for $(new_node.id)")
	# 						next.incomingEdgeDicts[new_node] = [edge_map]
	# 					end
	# 				end
	# 			end
	# 		end
	# 	end
	# 	new_max_id = nodes_counter-1
	# 	#add new nodes to controller
	# 	#i want to have the new nodes first, so in case there are two nodes with identical value the newer node is pruned and we skip rewiring
	# 	#no need to sort, just have new nodes examined before old nodes
	# 	#the inner union is needed to have an orderedset as first parameter, as the result of union({ordered}, {not ordered}) = {ordered}
	# 	orderedset = union(union(OrderedSet{Node}(), new_nodes), Set{Node}(oldnode for oldnode in values(nodes)))
	# 	all_nodes = filterNodes(orderedset, minval)
	# 	new_controller_nodes = Dict{Int64, Node}()
	# 	for node in all_nodes
	# 		#add nodes to the controller
	# 		new_controller_nodes[node.id] = node
	# 		checkNode(node, controller, minval; checkDistinct = false)
	# 	end
	# 	controller.nodes = new_controller_nodes
	# 	controller.maxId = new_max_id
	# end

	function full_backup_stochastic!(controller::Controller{A, W}) where {A, W}
		if typeof(controller.frame) <: randomTiger
			return
		end
		initial_node = controller.nodes[1]
		already_present_action = first(controller.nodes[1].actionProb)[1]
		@deb("already_present = $already_present_action", :shortfull)
		for a in actions(controller.frame)
			if a != already_present_action
				@deb("adding node with action $a", :shortfull)
				actionProb = Dict{A, Float64}(a => 1)
				edges = Dict{A, Dict{W, Vector{Pair{Int64, Float64}}}}( a => Dict{W, Vector{Pair{Int64, Float64}}}() )
				for z in observations(controller.frame)
					edges[a][z] = [(initial_node.id => 1)]
				end
				new_node = Node(length(controller.nodes)+1, actionProb, edges, Array{Float64, 2}(undef, 0, 0) )
				push!(controller.nodes, new_node)
			end
		end
	end

	"""
	Returns the value vector for a node with a single action and a single observation edge going to next_node
	"""
	function node_value(next_node::Node{A, W}, action::A, observation::W, pomdp::POMDP) where {A, W}
		states = POMDPs.states(pomdp)
		n_states = length(states)
		n_observations = POMDPs.n_observations(pomdp)
		γ = POMDPs.discount(pomdp)
		new_V = Vector{Float64}(undef, n_states)
		for s_index in 1:n_states
			state = states[s_index]
			transition_dist = POMDPs.transition(pomdp, state, action)
			#for efficiency only iterate in s' that can be originated from s, a
			#else the p_s_prime would be zero
			sum = 0.0
			for s_prime in transition_dist.vals
				s_prime_index = POMDPs.stateindex(pomdp, s_prime)
				p_s_prime = POMDPModelTools.pdf(transition_dist, s_prime)
				p_obs = POMDPModelTools.pdf(POMDPs.observation(pomdp, s_prime, action), observation)
				@deb("$action, $state, $observation, $s_prime", :data)
				@deb("$(next_node.value[s_prime_index]) * $(p_obs) * $(p_s_prime)", :data)
				sum+= next_node.value[s_prime_index] * p_obs * p_s_prime
			end
			new_V[s_index] = (1/n_observations) * POMDPs.reward(pomdp, state, action) + γ*sum
		end
		return new_V
	end

	"""
	Perform incremental pruning on a set of nodes by computing the cross sum and filtering every time
	Follows Cassandra et al paper
	"""
	function incprune(nodeVec::Vector{Set{Node}}, startId::Int64, minval::Float64)
		@deb("Called incprune, startId = $startId", :full)
		id = startId
		id, xs = xsum(nodeVec[1], nodeVec[2], id)
		res = filterNodes(xs, minval)
		#=if :data in debug
			for node in res
				println(node)
			end
		end
		=#
		for i = 3:length(nodeVec)
			@deb("Summing set $i", :full)
			id, xs = xsum(xs, nodeVec[i], id)
			res = filterNodes(xs, minval)
			#@deb("Length $i = $(length(nodeVec[i]))")
		end
		#@deb("returned ID = $id")
		return id, res
	end
	"""
	Perform the cross sum between elements in set A and B.
	startId is the current highest ID in the controller.
	Returns a new set of the cross-summed nodes.
	"""
	function xsum(A::Set{Node}, B::Set{Node}, startId::Int64)
		@deb("Called xsum, startId = $startId", :full)
		@deb("$(length(A)) * $(length(B))", :full)
		X = Set{Node}()
		id = startId
		for a in A, b in B
			#each of the newly generated nodes only has one action!
			@assert length(a.actionProb) == length(b.actionProb) == 1 "more than one action in freshly generated node"
			a_action = collect(keys(a.actionProb))[1]
			b_action = collect(keys(b.actionProb))[1]
			@assert a_action == b_action "action mismatch"
			c = mergeNode(a, b, a_action, id)
			if :data in debug
				println("nodes merged: $id<- $(a.id), $(b.id)")
				println(c)
			end
			id-=1
			push!(X, c)
		end
		#@deb("returned id = $id")
		return id, X
	end
	"""
		Merges two nodes into a new node with id = id
	"""
	function mergeNode(a::Node{A, W}, b::Node{A, W}, action::A, id::Int64) where {A, W}
		a_observation_map = a.edges[action]
		b_observation_map = b.edges[action]

		#There is no way we have repeated observation because set[obs]
		#only contains nodes with obs
		actionProb = Dict{A, Float64}(action => 1.0)
		edges = Dict{A, Dict{W, Vector{Pair{Int64, Float64}}}}(action => Dict{W, Vector{Pair{Int64, Float64}}}())
		for (obs, node_map) in a_observation_map
			edges[action][obs] = Vector{Pair{Int64, Float64}}(undef, 0)
			for (node_id, prob) in node_map
				push!(edges[action][obs], (node_id => prob))
			end
		end
		for (obs, node_map) in b_observation_map
			if haskey(edges[action], obs)
				error("Observation already present")
			end
			edges[action][obs] = Vector{Pair{Int64, Float64}}(undef, 0)
			for (node_id, prob) in node_map
				push!(edges[action][obs], (node_id => prob))
			end
		end
		return Node(id, actionProb, edges, a.value+b.value)
	end

	"""
		Filtering function to remove dominated nodes.
		Minval is the minimum probability that an edge can have. values below minval are treated as zero, values above 1-minval are treated as 1
	"""
	function filterNodes(nodes::Set{Node}, minval::Float64)
		return nodes
		@deb("Called filterNodes, length(nodes) = $(length(nodes))", :full)
		if length(nodes) > 1000
			@warn "Called filterNodes on $(length(nodes))"
		end
		if length(nodes) == 0
			error("called FilterNodes on empty set")
		end
		if length(nodes) == 1
			#if there is only one node it is useless to try and prune anything
			return nodes
		end
		new_nodes = Dict{Int64, Node}()
		#careful, here dict key != node.id!!!!
		node_counter = 1
		for node in nodes
			new_nodes[node_counter] = node
			node_counter+=1
		end
		#this ensures that the value vector is flattened even if it has multiple dimension because of nodes/agents/frames
		n_states = length(new_nodes[1].value)
		for (temp_id, n) in new_nodes
			#remove the node we're testing from the node set (else we always get that a node dominates itself!)
			if length(new_nodes) == 1
				#only one node in the set, no reason to keep pruning
				break;
			end
			pop!(new_nodes, temp_id)
			#@deb("$(length(new_nodes))")
			lpmodel = JuMP.Model(with_optimizer(CPLEX.Optimizer))
			CplexSolver(CPX_PARAM_SCRIND=0)

			#define variables for LP. c(i)
			@variable(lpmodel, c[i=keys(new_nodes)] >= 0)
			#e to maximize
			@variable(lpmodel, e)
			@objective(lpmodel, Max, e)
			@constraint(lpmodel, con[s_index=1:n_states], n.value[s_index] + e <= sum(c[ni_id]*ni.value[s_index] for (ni_id, ni) in new_nodes))
			@constraint(lpmodel, con_sum, sum(c[i] for i in keys(new_nodes)) == 1)
			optimize!(lpmodel)
			if :data in debug
				println("node $(n.id) -> eps = $(JuMP.value(e))")
			end
			if JuMP.value(e) >= -1e-12
				#rewiring function here!
				if :data in debug
					for i in keys(new_nodes)
						print("c$(new_nodes[i].id) = $(JuMP.value(c[i])) ")
					end
					println("")
				end
				#rewiring starts here!
				#@deb("Start of rewiring")
				for (src_node, dict_vect) in n.incomingEdgeDicts
					#skip rewiring of edges from dominated node
					if src_node != n
						#@deb("length of dict_vect = $(length(dict_vect))")
						for dict in dict_vect
							#dict[n] is the probability of getting to the dominated node (n)
							old_prob = dict[n]
							for (dst_id,dst_node) in new_nodes
								#remember that dst_id (positive) is != dst_node.id (negative)
								#add new edges to edges structure
								v = JuMP.value(c[dst_id])
								#FIXME hack: sum of cis is no longer 1 (but it is really close)
								if v >= 1-minval
									v = 1
								end
								if v > minval
									#@deb("Added edge from node $(src_node.id) to $(dst_node.id)")
									dict[dst_node] = v*old_prob
									#update incomingEdgeDicts for new dst node
									if haskey(dst_node.incomingEdgeDicts, src_node)
										push!(dst_node.incomingEdgeDicts[src_node], dict)
									else
										dst_node.incomingEdgeDicts[src_node] = [dict]
									end
								end
							end
							#remove edge of dominated node
							#@deb("Removed edge from node $(src_node.id) to $(n.id)")
							delete!(dict, n)
							if length(dict) == 0
								#only happens if n was the last node pointed by that node for that action/obs pair
								#and all c[i]s = 0, which is impossible!
								@deb("length of dict coming from $(src_node.id) == 0 after deletion, removing", :flow)
							end
						end
					end
				end
				#@deb("End of rewiring")
				#end of rewiring, do not readd dominated node
				#remove incoming edge from pointed nodes
				for (action, observation_map) in n.edges
					for (observation, edge_map) in observation_map
						for (next, prob) in edge_map
							if haskey(next.incomingEdgeDicts, n)
								#@deb("removed incoming edge vector from $(n.id) to $(next.id) ($action, $observation)")
								delete!(next.incomingEdgeDicts, n)
							end
						end
					end
				end
				if :data in debug
					println("Deleting node $(n.id): obj value = $(JuMP.value(e))")
					println(n)
				end
				#set it to nothing just to be sure
				n = nothing
			else
				#if node is not dominated readd it to the dict!
				new_nodes[temp_id] = n
			end
		end

		return Set{Node}(node for node in values(new_nodes))
	end

	"""
		Filtering function to remove dominated nodes.
		Minval is the minimum probability that an edge can have. values below minval are treated as zero, values above 1-minval are treated as 1
		This version of filterNodes analyzes new nodes before old nodes to avoid rewiring in case of nodes with equal values
	"""
	function filterNodes(nodes::OrderedSet{Node}, minval::Float64)
		@deb("Called filterNodes, length(nodes) = $(length(nodes))", :data)
		if length(nodes) == 0
			error("called FilterNodes on empty set")
		end
		if length(nodes) == 1
			#if there is only one node it is useless to try and prune anything
			return nodes
		end
		new_nodes = Dict{Int64, Node}()
		#careful, here dict key != node.id!!!!
		node_counter = 1
		for node in nodes
			new_nodes[node_counter] = node
			@deb("temp = $node_counter, id = $(node.id)", :data)
			@deb(node, :data)
			node_counter+=1
		end
		n_states = length(new_nodes[1].value)
		#since we know temp_ids are contiguous we can iterate in order
		#this way new nodes are checked first without need for an ordered dict nor sorting
		for temp_id in 1:length(new_nodes)
			n = new_nodes[temp_id]
			#@deb("temp_id = $temp_id, n.id = $(n.id)")
			#remove the node we're testing from the node set (else we always get that a node dominates itself!)
			if length(new_nodes) == 1
				#only one node in the set, no reason to keep pruning
				break;
			end
			pop!(new_nodes, temp_id)
			#@deb("$(length(new_nodes))")
			lpmodel = JuMP.Model(with_optimizer(CPLEX.Optimizer))
			CplexSolver(CPX_PARAM_SCRIND=0)

			#define variables for LP. c(i)
			@variable(lpmodel, 0.0 <= c[i=keys(new_nodes)] <= 1.0)
			#e to maximize
			@variable(lpmodel, e)
			@objective(lpmodel, Max, e)
			@constraint(lpmodel, con[s_index=1:n_states], n.value[s_index] + e <= sum(c[ni_id]*ni.value[s_index] for (ni_id, ni) in new_nodes))
			@constraint(lpmodel, con_sum, sum(c[i] for i in keys(new_nodes)) == 1)
			optimize!(lpmodel)
			if :data in debug
				println("node $(n.id) -> eps = $(JuMP.value(e))")
			end
			if JuMP.value(e) >= -1e-10
				#rewiring function here!
				if :data in debug
					for i in keys(new_nodes)
						print("c$(new_nodes[i].id) = $(JuMP.value(c[i])) ")
					end
					println("")
				end
				#rewiring starts here!
				@deb("Start of rewiring")
				for (src_node, dict_vect) in n.incomingEdgeDicts
					#skip rewiring of edges from dominated node
					if src_node != n
						#@deb("length of dict_vect = $(length(dict_vect))")
						for dict in dict_vect
							#dict[n] is the probability of getting to the dominated node (n)
							old_prob = dict[n]
							for (dst_id,dst_node) in new_nodes
								#remember that dst_id (positive) is != dst_node.id (negative)
								#add new edges to edges structure
								v = JuMP.value(c[dst_id])
								if v >= 1-minval
									v = 1
								end
								if v > minval
									if haskey(dict, dst_node)
										@deb("updated probability of edge from node $(src_node.id) to $(dst_node.id)", :data)
										dict[dst_node]+= v*old_prob
										if dict[dst_node] > 1
											error("probability > 1 after redirection!")
										end
									else
										@deb("Added edge from node $(src_node.id) to $(dst_node.id)", :data)
										dict[dst_node] = v*old_prob
										#update incomingEdgeDicts for new dst node
										if haskey(dst_node.incomingEdgeDicts, src_node)
											push!(dst_node.incomingEdgeDicts[src_node], dict)
										else
											dst_node.incomingEdgeDicts[src_node] = [dict]
										end
									end
								end
							end
							#remove edge of dominated node
							@deb("Removed edge from node $(src_node.id) to $(n.id)")
							delete!(dict, n)
							if length(dict) == 0
								#only happens if n was the last node pointed by that node for that action/obs pair
								#and all c[i]s = 0, which is impossible!
								@deb("length of dict coming from $(src_node.id) == 0 after deletion, removing")
							end
						end
					end
				end
				@deb("End of rewiring")
				#end of rewiring, do not readd dominated node
				#remove incoming edge from pointed nodes
				for (action, observation_map) in n.edges
					for (observation, edge_map) in observation_map
						for (next, prob) in edge_map
							if haskey(next.incomingEdgeDicts, n)
								@deb("removed incoming edge vector from $(n.id) to $(next.id) ($action, $observation)")
								delete!(next.incomingEdgeDicts, n)
							end
						end
					end
				end
				if :data in debug
					println("Deleting node $(n.id): obj value = $(JuMP.value(e))")
					println(n)
				end
				#set it to nothing just to be sure
				n = nothing
			else
				#if node is not dominated readd it to the dict!
				new_nodes[temp_id] = n
			end
		end

		return Set{Node}(node for node in values(new_nodes))
	end

	"""
	Computes all value vectors of the nodes in a non-interactive controller.
	"""
	function evaluate!(controller::Controller{A,W}) where {A, W}
		#log_n_nodes(controller.stats, length(controller.nodes))
		#start_time(controller.stats, "eval")
		#solve V(n,s) = R(s, a(n)) + gamma*sumz(P(s'|s,a(n))Pr(z|s',a(n))V(beta(n,z), s'))
		#R(s,a(n)) is the reward function
		pomdp = controller.frame
		nodes = controller.nodes
		n_nodes = length(keys(controller.nodes))
		states = POMDPs.states(pomdp)
		n_states = POMDPs.n_states(pomdp)
		M = zeros(n_nodes, n_states, n_nodes, n_states)
		b = zeros(n_nodes, n_states)

		# #dictionary used for recompacting ids
		# temp_id = Dict{Int64, Int64}()
		# for (node_id, node) in nodes
		# 	temp_id[node_id] = length(temp_id)+1
		# end

		#compute coefficients for sum(a)[R(s|a)*P(a|n)+gamma*sum(z, n', s')[P(s'|s,a)*P(z|s',a)*P(a|n)*P(n'|z)*V(nz, s')]]
		#start_time(controller.stats, "eval_coeff")
		for node in nodes
			n_id = node.id
			#M is the coefficient matrix (form x1 = a2x2+...+anxn+b)
			#b is the constant term vector
			#variables are all pairs of n,s
			actions = keys(node.actionProb)
			for s_index in 1:n_states
				s = POMDPs.states(pomdp)[s_index]
				for a in actions
					@deb("action = $a", :eval)
					p_a_n = node.actionProb[a]
					b[n_id, s_index] += POMDPs.reward(pomdp, s, a) * p_a_n
					@deb("b($n_id, $s) = $(POMDPs.reward(pomdp, s, a)*p_a_n)", :eval)
					M[n_id, s_index, n_id, s_index] += 1
					@deb("M[$n_id, $s][$n_id, $s] = 1", :eval)
					possible_obs = keys(node.edges[a])  #only consider observations possible from current node/action combo
					for obs in possible_obs
						@deb("obs = $obs", :eval)
						for s_prime_index in 1:n_states
							s_prime = states[s_prime_index]
							@deb("s_prime = $s_prime", :eval)

							p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp,s,a), s_prime)
							@deb("p_s_prime = $p_s_prime", :eval)
							if p_s_prime == 0.0
								continue
							end
							p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp,s_prime, a), obs)
							@deb("p_z = $p_z", :eval)

							partial_mult = p_a_n * POMDPs.discount(pomdp) * p_s_prime * p_z
							for (next_id, prob) in node.edges[a][obs]
								if next_id > length(controller.nodes)
									error("Node $(next_id) not present in nodes")
								end
								M[n_id, s_index, next_id, s_prime_index]-= partial_mult * prob
							end
						end
					end
				end
			end
		end
		#stop_time(controller.stats, "eval_coeff")

		@deb("M = $M")
		@deb("b = $b")
		#start_time(controller.stats, "eval_solve")

		res = reshape(M, n_nodes * n_states, n_nodes * n_states) \ reshape(b, n_nodes * n_states)
		#copy respective value functions in nodes
		#stop_time(controller.stats, "eval_solve")

		res_2d = reshape(res, n_nodes, n_states)
		for node in nodes
			#create a new node identical to the old one but with updated value
			controller.nodes[node.id] = Node(node.id, node.actionProb, node.edges, copy( res_2d[node.id, :]))
			@deb("Value vector of node $n_id = $(nodes[node.id].value)")
			#set old node to nothing to make sure it's garbage collected
			node = nothing
		end
		#stop_time(controller.stats, "eval")
	end
	"""
	Tries to improve the controller by checking if each node can be replaced by a convex combination of the other nodes.
	Default behavior is to stop after a single node is modified.
	first return value is whether it has improved at least one node, second is the tangent belief point (see paper)
	# Return Bool, Vector{Float64}
	"""
	function partial_backup!(controller::Controller{A, W}; add_one = true, debug_node = 0) where {A, W}
		if typeof(controller.frame) <: randomTiger
			return false, []
		end
		#start_time(controller.stats, "partial")
		pomdp = controller.frame
		nodes = controller.nodes
		n_nodes = length(controller.nodes)
		states = POMDPs.states(pomdp)
		n_states = POMDPs.n_states(pomdp)
		actions = POMDPs.actions(pomdp)
		n_actions = POMDPs.n_actions(pomdp)
		observations = POMDPs.observations(pomdp)
		n_observations = POMDPs.n_observations(pomdp)
		#vector containing the tangent belief states for all modified nodes
		tangent_b = Dict{Int64, Array{Float64}}()
		constraints = Array{ConstraintRef}(undef, n_states)

		#if at least one node has been modified.
		changed = false
		#recompacting dict for controller
		# node_counter = 1
		# temp_id = Dict{Int64, Int64}()
		# for real_id in keys(nodes)
		# 		temp_id[real_id] = node_counter
		# 		@deb("Node $real_id becomes $node_counter", :data )
		# 		node_counter+=1
		# end
		#start of actual algorithm
		nodecounter= 0
		for node in nodes
			n_id = node.id
			@deb("Node to be improved: $n_id", :checkNodes)

			nodecounter += 1
			if nodecounter >= (length(nodes)/100)
				for i in 1:(nodecounter * 100 / length(nodes))
					print("|")
				end
				nodecounter = 0
			end
			lpmodel = JuMP.Model(with_optimizer(CPLEX.Optimizer; CPX_PARAM_SCRIND=0))

			#define variables for LP. c(a, n, z)
			@variable(lpmodel, canz[a=1:n_actions, z=1:n_observations, n=1:n_nodes] >= 0.0)
			@variable(lpmodel, ca[a=1:n_actions] >= 0.0)
			#e to maximize
			@variable(lpmodel, e)
			@objective(lpmodel, Max, e)
			#define coefficients for constraints
			#start_time(controller.stats, "partial_coeff")
			for s_index in 1:n_states
				s = states[s_index]
				#matrix of canz coefficients
				M = zeros(n_actions, n_observations, n_nodes)
				#matrix of ca coefficients
				M_a = zeros(n_actions)
				for a_index in 1:n_actions
					action = actions[a_index]
					M_a[a_index] = POMDPs.reward(pomdp, s, action)
					s_primes = POMDPs.transition(pomdp,s,action).vals
					for obs_index in 1:n_observations
						obs = observations[obs_index]
						#array of edges given observation
						for s_prime in s_primes
							p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp,s,action), s_prime)
							p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, s_prime, action), obs)
							if p_s_prime != 0.0 && p_z != 0.0
								for n_prime in nodes
									#iterate over all possible n_prime
									v_nz_sp = n_prime.value[POMDPs.stateindex(pomdp, s_prime)]
									M[a_index, obs_index, n_prime.id]+= p_s_prime*p_z*v_nz_sp
								end
							end
						end
					end
				end
				#set constraint for a state
				constraints[s_index] = @constraint(lpmodel,  e + node.value[s_index] <= sum( M_a[a]*ca[a]+POMDPs.discount(pomdp)*sum(sum( M[a, z, n] * canz[a, z, n] for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions))
			end
			#stop_time(controller.stats, "partial_coeff")

			@constraint(lpmodel, con_sum[a=1:n_actions, z=1:n_observations], sum(canz[a, z, n] for n in 1:n_nodes) == ca[a])
			@constraint(lpmodel, ca_sum, sum(ca[a] for a in 1:n_actions) == 1.0)

			@deb(lpmodel, :data)
			#start_time(controller.stats, "partial_optimize")

			optimize!(lpmodel)
			#stop_time(controller.stats, "partial_optimize")

			@deb("$(termination_status(lpmodel))")
			@deb("$(primal_status(lpmodel))")
			@deb("$(dual_status(lpmodel))")
			@deb("Obj = $(objective_value(lpmodel))", :data)
			delta = JuMP.objective_value(lpmodel)
			if delta > config.minval
				@deb("Node improved by $delta", :flow)
				#means that node can be improved!
				changed = true
				new_edges = Dict{A, Dict{W,Vector{Pair{Int64, Float64}}}}()
				new_actions = Dict{A, Float64}()
				#building the new node using the ca and canz obtained with LP
				for action_index in 1:n_actions
					ca_v = JuMP.value(ca[action_index])
					@deb("Action $(actions[action_index])")
					@deb("ca $(actions[action_index])= $ca_v")
					# if ca_v > 1.0-config.minval
					# 	ca_v = 1.0
					# end
					if ca_v > config.minval
						new_obs = Dict{W, Vector{Pair{Int64, Float64}}}()
						for obs_index in 1:n_observations
							#obs_total = 0.0
							#fill a temporary edge dict with unnormalized probs
							temp_edge_dict = Vector{Pair{Int64, Float64}}()
							for nz in nodes
								prob = JuMP.value(canz[action_index, obs_index, nz.id])/ca_v
								#@deb("canz $(observations[obs_index]) -> $nz_id = $prob")
								# if prob < 0.0
								# 	@deb("Set prob to 0 even though it was negative")
								# 	prob = 0.0
								# end
								# if prob > 1.0 && prob < 1.0+config.minval
								# 	@deb("Set prob slightly greater than 1 to 1")
								# 	prob = 1.0
								# end
								# if prob < 0.0 || prob > 1.0
								# 	#error("Probability outside of bounds: $prob")
								# end
								if prob > config.minval
									# obs_total+= prob
									#@deb("New edge: $(action_index), $(obs_index) -> $nz_id, $(prob)")
									push!(temp_edge_dict, (nz.id => prob))
								end
							end
							# if obs_total == 0.0
							# 	error("sum of prob for obs $(observations[obs_index]) == 0")
							# end
							new_edge_dict = Vector{Pair{Int64, Float64}}()
							for (next_id, prob) in temp_edge_dict
								#@deb("normalized prob: $normalized_prob")
								# if prob >= 1.0-config.minval
								# 	push!(new_edge_dict, (next_id => 1.0))
								# elseif prob > config.minval
									# push!(new_edge_dict, (next_id => prob/obs_total))
									push!(new_edge_dict, (next_id => prob))

								# end
								#do not add anything if prob < config.minval
							end
							#@deb("length of dict for obs $(observations[obs_index]) = $(length(new_edge_dict))")
							if length(new_edge_dict) != 0
								new_obs[observations[obs_index]] = new_edge_dict
							end
						end
						if length(keys(new_obs)) != 0
							new_edges[actions[action_index]] = new_obs
							new_actions[actions[action_index]] = ca_v
						end
					end
				end
				new_node = Node(node.id, new_actions, new_edges, Array{Float64, 2}(undef, 0, 0))
				checkNode(new_node, controller; normalize = config.normalize)
				controller.nodes[new_node.id] = new_node
				#make sure it's garbage collected!
				node = nothing
				if !add_one
					if :flow in debug
						println("Changed controller after eval")
						for node in controller.nodes
							println(node)
						end
					end
				end
				if add_one
					#no need to update tangent points because they wont be used!
					println()
					@deb("Changed node, value still to be recomputed", :flow)
					@deb(new_node, :flow)
					return true, Dict{Int64, Array{Float64}}()
				end
			end
			#set the tangent_b of a node with -1 * dual of the constraint of each state.
			tangent_b[n_id] = [-1*dual(constraints[s_index]) for s_index in 1:n_states]
		end
		println()
		#stop_time(controller.stats, "partial")
		return changed, tangent_b
	end
	"""
	Tries to escape local optima by adding new nodes to the controller.
	Default behavior is to add all nodes that improve reachable beliefs of a single node.
	"""
	function escape_optima_standard!(controller::Controller{A, W}, tangent_b::Dict{Int64, Array{Float64}}; add_one=true) where {A, W}
		#@deb("$tangent_b")
		if typeof(controller.frame) <: randomTiger
			return false, []
		end
		#start_time(controller.stats, "escape")

		pomdp = controller.frame
		nodes = controller.nodes
		n_nodes = length(controller.nodes)
		states = POMDPs.states(pomdp)
		n_states = POMDPs.n_states(pomdp)
		actions = POMDPs.actions(pomdp)
		n_actions = POMDPs.n_actions(pomdp)
		observations = POMDPs.observations(pomdp)
		n_observations = POMDPs.n_observations(pomdp)
		if length(tangent_b) == 0
			error("tangent_b was empty!")
		end

		# old_deb = debug
		# debug = false

		# new_nodes = full_backup_generate_nodes(controller, minval)
		#
		# debug = old_deb
		#if :data in debug
		#	println("new_nodes:")
		#	for node in new_nodes
		#		println(node)
		#	end
		#end


		escaped = false
		reachable_beliefs = Set{Vector{Float64}}()
		for (id, start_b) in tangent_b
			#id = collect(keys(tangent_b))[1]
			#start_b = tangent_b[id]
			@deb("$id - >$start_b")
			for a in keys(nodes[id].actionProb)
				for z in observations
					new_b = belief_update(start_b, a, z, pomdp)
					@deb("from belief $start_b action $a and obs $z -> $new_b")

					if add_one
						escaped = escaped || add_escape_node!(new_b, controller)
						@deb("Added node $(controller.nodes[end]) to improve node $id")
					else
						push!(reachable_beliefs, new_b)
					end
				end
			end
			if add_one && escaped
				break
			end
		end
		if !add_one
			for reachable_b in reachable_beliefs
				@deb("Trying to improve reachable belief $reachable_b", :escape)
				escaped_single = add_escape_node!(reachable_b, controller)
				if escaped_single
					@deb("Added node $(controller.nodes[end].id)", :escape)
				end
				escaped = escaped || escaped_single
			end
		end
		#@deb("$reachable_b")
		#stop_time(controller.stats, "escape")

		return escaped
	end

	function add_escape_node!(reachable_b::Array{Float64}, controller::Controller{A, W}) where {A, W}
		best_old_node, best_old_value = get_best_node(reachable_b, controller.nodes)
		#generate node directly
		best_new_node, best_new_value = generate_node_directly(controller, reachable_b)
		if best_new_value - best_old_value > config.min_improvement
			@deb("in $reachable_b node $(best_new_node.id) has $best_new_value > $best_old_value", :escape)
			@deb("best old node:", :generatenode)
			@deb(best_old_node, :generatenode)
			checkNode(best_new_node, controller; normalize = config.normalize)
			push!(controller.nodes, best_new_node)
			@deb("Added node $(best_new_node.id) to improve $reachable_b", :flow)
			@deb(best_new_node, :flow)
			return true
		end
		return false
	end

	function belief_update(start_b::Array{Float64}, a::A, z::W, pomdp::POMDP) where{A, W}
		states = POMDPs.states(pomdp)
		n_states = length(states)
		new_b = Vector{Float64}(undef, n_states)
		normalize = 0.0
		for s_prime_index in 1:n_states
			s_prime = states[s_prime_index]
			sum_s = 0.0
			p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, s_prime, a), z)
			for s_index in 1:n_states
				s = states[s_index]
				p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp, s, a), s_prime)
				sum_s+= p_s_prime*start_b[s_index]
			end
			new_b[s_prime_index] = p_z * sum_s
			normalize += p_z * sum_s
		end
		new_b = new_b ./ normalize
		return new_b
	end

	function get_best_node(belief::Array{Float64}, nodes::Vector{Node{A, W}}) where {A, W}
		@deb("started get_best_node", :bestnode)
		best_node = nothing
		best_value = 0.0
		for node in nodes
			if length(belief) != length(node.value)
				@deb(belief, :bestnode)
				@deb(node, :bestnode)
				error("Dimension mismatch between belief and value vector")
			end
			value =  sum(belief[i] * node.value[i]  for i in 1:length(belief))

			if best_node == nothing || best_value < value
				best_node = node
				best_value = value
			end
		end
		return best_node, best_value
	end

	# function get_best_node(belief::Array{Float64}, nodes::Vector{Node})
	# 	@deb("started get_best_node", :bestnode)
	#
	# 	best_node = nothing
	# 	best_value = 0.0
	# 	@assert length(belief) == length(nodes[1].value)
	# 	for node in nodes
	# 		@deb(" belief: $belief, value = $value")
	# 		value =  sum(belief[i] * node.value[i]  for i in 1:length(belief))
	# 		@deb(" new = $value, best =  $best_value", :bestnode)
	# 		if best_node == nothing || best_value < value
	# 			@deb("new best node $(node.id)", :bestnode)
	# 			best_node = node
	# 			best_value = value
	# 		end
	# 	end
	# 	return best_node, best_value
	# end
	#
	# function rework_node(controller::AbstractController, new_node::Node{A, W}) where {A, W}
	# 		id = controller.maxId+1
	# 		actionProb = copy(new_node.actionProb)
	# 		value = copy(new_node.value)
	# 		edges = Dict{A, Dict{W, Vector{Pair{Int64, Float64}}}}()
	# 		for (a, obs_dict) in new_node.edges
	# 			edges[a] = Dict{W, Vector{Pair{Int64, Float64}}}()
	# 			for (z, node_dict) in obs_dict
	# 				edges[a][z] = Dict{Int64,Float64}()
	# 				for (node, prob) in node_dict
	# 					current_controller_node = controller.nodes[node.id]
	# 					edges[a][z][current_controller_node] = prob
	# 				end
	# 			end
	# 		end
	# 		return Node(id, actionProb,edges, value, Dict{Int64, Vector{Vector{Pair{Int64, Float64}}}}())
	# end
	function generate_node_directly(controller::Controller{A, W}, start_b::Array{Float64}) where {A, W}
		actions = POMDPs.actions(controller.frame)
		observations = POMDPs.observations(controller.frame)
		n_observations = length(observations)

		best_node = nothing
		best_value = 0
		for a in actions
			#try all actions
			new_node = nothing
			for z_index in 1:n_observations
				#find the best edge (aka the best next node) for each observation
				z = observations[z_index]
				#compute the result belief of executing action a and receiving obs z starting from belief b.
				result_b = belief_update(start_b,a,z,controller.frame)
				#get the best node in the controller for the updated beief
				best_next_node, best_value_obs = get_best_node(result_b, controller.nodes)
				new_v = node_value(best_next_node, a, z, controller.frame)
				new_node_partial = build_node(length(controller.nodes)+1, a, z, best_next_node, new_v)
				#add that edge to the node and update the value vector
				if z_index ==1
					new_node = new_node_partial
				else
					new_node = mergeNode(new_node, new_node_partial, a,  length(controller.nodes)+1)
				end
			end
			#compute the best node (choose between actions)
			new_value = sum(start_b[i]*new_node.value[i] for i in 1:length(start_b))
			if best_node == nothing || best_value < new_value
				best_node = new_node
				best_value = new_value
			end
		end
		return best_node, best_value
	end


function bpi!(policy::BPIPolicy)
	controller = policy.controller
	evaluate!(controller)

	full_backup_stochastic!(controller)
	improved = false
	iteration = 0
	while iteration <= config.maxrep || config.maxrep < 0
		println("Iteration $iteration")
		evaluate!(controller)

		improved, tangent_b = partial_backup!(controller; add_one = true)


		if !improved
			escaped = escape_optima_standard!(controller, tangent_b; add_one = false)
			if !escaped
				println("Convergence!")
				return true
			end
		end
		iteration += 1
	end
	println("Maxrep exceeded")
	return !improved
end
