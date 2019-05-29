#=
IBPIPolicyUtils:
- Julia version: 1.1.0
- Author: Riccardo Ficarra
- Date: 2019-02-11
=#
	using LinearAlgebra
	using DataStructures
	using JuMP
	using GLPK
	using SparseArrays

	"""
	Structure used for policy nodes.
	ActionDist specifies the probability of executing the action at the corresponding index in actions
	Edges stores the possible edges given action and observations obtained after executing the action.
	outer key is the action -> inner key is the observation -> for each observation get the list of all possible edges
	Actions only contains actions with probability >0
	Right now the (any) operator for observation is not implemented, we just have entries for all possible values
	receives as parameters all possible actions and all possible observations
	Each node has an unique identifier, the ids of deleted nodes are reused and continuous (1:n_nodes)
	"""
	mutable struct Node{A, W}
		id::Int64
		actionProb::Dict{A, Float64}
		#action -> observation -> node -> prob
		edges::Dict{A, Dict{W, Dict{Node, Float64}}}
		value::Array{Float64}
		#needed to efficiently redirect edges during pruning
		#srcNode -> vectors of dictionaries that contains edge to this node
		incomingEdgeDicts::Dict{Node, Vector{Dict{Node, Float64}}}
	end

	#overload hash and isequal to use only id as keys in dicts
	Base.hash(n::Node) = hash(n.id)
	Base.isequal(n1::Node, n2::Node) = Base.isequal(hash(n1), hash(n2))
	#overload display function to avoid walls of text when printing nodes
	Base.display(n::Node) = println(n)

	function Node(id::Int64,actions::Vector{A}, observations::Vector{W}) where {A, W}
		actionProb = Dict{A, Float64}()
		for i in 1:length(actions)
			actionProb[actions[i]] = 1/length(actions)
		end
		return Node(id::Int64, actionProb::Dict{A, Float64}, Dict{A, Dict{W, Dict{Node, Float64}}}(), Vector{Float64}(), Dict{Node, Vector{Dict{Node, Float64}}}())
	end

	function Base.println(node::Node)
		for (a, a_prob) in node.actionProb
			obs = node.edges[a]
			for (obs, edges) in obs
				for (next, prob) in edges
					println("node_id=$(node.id), a=$a, $a_prob, $obs -> $(next.id), $prob")
				end
			end
		end
		#=
		if debug[] == true
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
		Receives vectors of all possible actions and observations, plus number of states
		Get a node with a random action chosen and with all observation edges
		pointing back to itself
		If force is defined the action is decided by the user (by index in 1:length(actions))
	"""
	function InitialNode(pomdp::POMDP{A, W},force::Int64) where {A, W}
			actions = POMDPs.actions(pomdp)
			observations = POMDPs.observations(pomdp)
			states = POMDPs.states(pomdp)
			n_states = POMDPs.n_states(pomdp)
			if force == 0
				actionindex = rand(1:length(actions))
			else
				if force > length(actions)
					error("forced action outside of action vector length")
				end
				actionindex = force
			end
			n = Node(1, [actions[actionindex]], observations)
			obsdict = Dict{W, Dict{Node, Float64}}()
			n.incomingEdgeDicts[n] = Vector{Dict{Node, Float64}}(undef, 0)
			for obs in observations
				edges = Dict{Node, Float64}(n => 1.0)
				obsdict[obs] = edges
				push!(n.incomingEdgeDicts[n], edges)
			end
			n.edges[actions[actionindex]] = obsdict
			n.value = [POMDPs.reward(pomdp, s, actions[actionindex]) for s in states]
			return n
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
		Get a vector of actions with probability != 0
		TODO: transform the two arrays in a dict, only keep possible actions
	"""
	function getPossibleActions(node::Node{A, W}) where {A, W}
		return keys(node.actionProb)
	end
	"""
		given node, action and observation returns the next node
		maps the whole array of edges to get edge prob (O(n)), then calls chooseWithProbability O(n)
	"""
	function getNextNode(node::Node{A, W}, action::A, observation::W) where {A, W}
		if !haskey(node.edges, action)
			error("Action has probability 0!")
		end
		edges = node.edges[action][observation]
		next = chooseWithProbability(edges).next
		@deb("Chosen $(next.id) as next node")
		return next
	end
	"""
	Given a dictionary in the form item => probability
	Pick a random item based on the probability.
	probability must sum to 1.
	O(n)
	"""
	function chooseWithProbability(items::Dict)
		randn = rand() #number in [0, 1)
		@deb(randn)
		for i in keys(items)
			if randn <= items[i]
				return i
			else
				randn-= items[i]
			end
		end
		error("Out of dict bounds while choosing items")
	end


	mutable struct Controller{A, W} <: AbstractController
		#level::Int64
		frame::POMDP{A, W}
		nodes::Dict{Int64, Node{A, W}}
		maxId::Int64
	end
	"""
	Initialize a controller with the initial node, start id counter from 2
	"""
	function Controller(pomdp::POMDP{A,W}, force::Int64) where {A, W}
		newNode = InitialNode(pomdp, force)
		Controller{A, W}(pomdp, Dict(1 => newNode), 1)
	end

	function optimal_tiger_controller(pomdp::POMDP{A, W}) where {A, W}
		controller = Controller(pomdp, 3)
		controller.nodes[1].id = 1
		#create the open left(2)- open right(3) nodes
		controller.nodes[3] = InitialNode(pomdp, 1)
		controller.nodes[3].id = 3
		controller.nodes[2] = InitialNode(pomdp, 2)
		controller.nodes[2].id = 2
		for i in 5:10
			controller.nodes[i] = InitialNode(pomdp, 3)
			controller.nodes[i].id = i
		end
		controller.nodes[1].edges[:L][:GL] = Dict(controller.nodes[5] => 1.0)
		controller.nodes[1].edges[:L][:GR] = Dict(controller.nodes[6] => 1.0)
		controller.nodes[2].edges[:OR][:GL] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[2].edges[:OR][:GR] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[3].edges[:OL][:GL] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[3].edges[:OL][:GR] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[5].edges[:L][:GL] = Dict(controller.nodes[2] => 1.0)
		controller.nodes[5].edges[:L][:GR] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[6].edges[:L][:GL] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[6].edges[:L][:GR] = Dict(controller.nodes[3] => 1.0)
		controller.nodes[7].edges[:L][:GL] = Dict(controller.nodes[10] => 1.0)
		controller.nodes[7].edges[:L][:GR] = Dict(controller.nodes[8] => 1.0)
		controller.nodes[8].edges[:L][:GL] = Dict(controller.nodes[7] => 1.0)
		controller.nodes[8].edges[:L][:GR] = Dict(controller.nodes[9] => 1.0)
		controller.nodes[9].edges[:L][:GL] = Dict(controller.nodes[8] => 1.0)
		controller.nodes[9].edges[:L][:GR] = Dict(controller.nodes[3] => 1.0)
		controller.nodes[10].edges[:L][:GL] = Dict(controller.nodes[2] => 1.0)
		controller.nodes[10].edges[:L][:GR] = Dict(controller.nodes[7] => 1.0)
		controller.maxId = 10
		old_deb = debug[]
		debug[] = false
		evaluate!(controller, pomdp)
		debug[] = old_deb
		if debug[] == true
			println("Optimal controller for tiger game:")
			for (node_id, node) in controller.nodes
			    println(node)
			end
		end
		return controller
	end

	function optimal_tiger_controller_stochastic(pomdp::POMDP{A, W}) where {A, W}
		controller = Controller(pomdp, 3)
		controller.nodes[1].id = 1
		#create the open left(2)- open right(3) nodes
		controller.nodes[3] = InitialNode(pomdp, 1)
		controller.nodes[3].id = 3
		controller.nodes[2] = InitialNode(pomdp, 2)
		controller.nodes[2].id = 2
		for i in 5:6
			controller.nodes[i] = InitialNode(pomdp, 3)
			controller.nodes[i].id = i
		end
		controller.nodes[1].edges[:L][:GL] = Dict(controller.nodes[5] => 0.73,controller.nodes[1] => 0.27)
		controller.nodes[1].edges[:L][:GR] = Dict(controller.nodes[6] => 1.0)
		controller.nodes[2].edges[:OR][:GL] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[2].edges[:OR][:GR] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[3].edges[:OL][:GL] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[3].edges[:OL][:GR] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[5].edges[:L][:GL] = Dict(controller.nodes[2] => 1.0)
		controller.nodes[5].edges[:L][:GR] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[6].edges[:L][:GL] = Dict(controller.nodes[1] => 1.0)
		controller.nodes[6].edges[:L][:GR] = Dict(controller.nodes[3] => 0.69, controller.nodes[6] => 0.31)
		controller.maxId = 6
		old_deb = debug[]
		debug[] = false
		evaluate!(controller, pomdp)
		debug[] = old_deb
		if debug[] == true
			println("Optimal controller for tiger game:")
			for (node_id, node) in controller.nodes
			    println(node)
			end
		end
		return controller
	end

	function example_controller(pomdp::POMDP{A, W}) where {A, W}
		controller = Controller(pomdp, 1)
		controller.nodes[1].id = 1
		#create the open left(2)- open right(3) nodes
		controller.nodes[2] = InitialNode(pomdp, 2)
		controller.nodes[2].id = 2
		controller.maxId = 2
		old_deb = debug[]
		debug[] = false
		evaluate!(controller, pomdp)
		debug[] = old_deb
		if debug[] == true
			println("Optimal controller for tiger game:")
			for (node_id, node) in controller.nodes
			    println(node)
			end
		end
		return controller
	end

	struct BPIPolicy{A, W}
        controller::Controller{A, W}
    end

    function BPIPolicy(pomdp::POMDP{A, W}, force::Int64) where {A, W}
        BPIPolicy(Controller(pomdp, force))
    end

	function build_node(node_id::Int64, actions::Vector{A}, actionProb::Vector{Float64}, observations::Vector{Vector{W}}, observation_prob::Vector{Vector{Float64}}, next_nodes::Vector{Vector{Node{A,W}}}, value::Array{Float64}) where {A, W}
		if length(actions) != length(observations) || length(actions) != length(actionProb) || length(actions) != length(observation_prob) || length(actions) != length(next_nodes)
			error("Length of action-level arrays are different")
		end
		edges = Dict{A, Dict{W, Dict{Node, Float64}}}()
		d_actionprob = Dict{A, Float64}()
		for a_i in 1:length(actions)
			action = actions[a_i]
			#fill actionprob dict
			d_actionprob[action] = actionProb[a_i]
			#vector of observations tied to action
			a_obs = observations[a_i]
			a_obs_prob = observation_prob[a_i]
			a_next_nodes = next_nodes[a_i]
			if length(a_obs) != length(a_obs_prob) || length(a_obs) != length(a_next_nodes)
				error("Length of observation-level arrays are different")
			end
			new_obs = Dict{W, Dict{Node, Float64}}()
			for obs_index in 1:length(a_obs)
				obs = a_obs[obs_index]
				new_obs[obs] = Dict{Node, Float64}(a_next_nodes[obs_index] => a_obs_prob[obs_index])
			end
			edges[action] = new_obs
		end
		return Node(node_id, d_actionprob, edges, value, Dict{Node, Vector{Dict{Node, Float64}}}())
	end
	function build_node(node_id::Int64, action::A, observation::W, next_node::Node{A, W}, value::Array{Float64}) where {A, W}
		actionprob = Dict{A, Float64}(action => 1.0)
		edges = Dict{A, Dict{W, Dict{Node, Float64}}}(action => Dict{W, Dict{Node, Float64}}(observation => Dict{Node, Float64}(next_node=> 1.0)))
		return Node(node_id, actionprob, edges, value, Dict{Node, Vector{Dict{Node, Float64}}}())
	end

	function full_backup_generate_nodes(controller::Controller{A, W}, pomdp::POMDP{A, W}, minval::Float64) where {A, W}
		minval = 1e-10
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
				if debug[] == true
					println("New nodes created:")
					for node in new_nodes_a_z
						println(node)
					end
				end
				new_nodes_z[obs_index] = filterNodes(new_nodes_a_z, minval)
			end
			#set that contains all nodes generated from action a after incremental pruning
			new_nodes_counter, new_nodes_a = incprune(new_nodes_z, new_nodes_counter, minval)
			union!(new_nodes, new_nodes_a)
		end
		#all new nodes, final filtering
		return filterNodes(new_nodes, minval)
	end
	"""
	Perform a full backup operation according to Pourpart and Boutilier's paper on Bounded finite state controllers
	"""
	function full_backup_stochastic!(controller::Controller{A, W}, pomdp::POMDP{A, W}) where {A, W}
		minval = 1e-10
		nodes = controller.nodes
		observations = POMDPs.observations(pomdp)
		#tentative from incpruning
		#prder of it -> actions, obs
		#for each a, z produce n new nodes (iterate on nodes)
		#for each node iterate on s and s' to produce a new node
		#new node is tied to old node?, action a and obs z
		#with stochastic pruning we get the cis needed
		new_nodes = full_backup_generate_nodes(controller, pomdp, minval)
		#before performing filtering with the old nodes update incomingEdge structure of old nodes
		#also assign permanent ids
		nodes_counter = controller.maxId+1
		for new_node in new_nodes
			@deb("Node $(new_node.id) becomes node $(nodes_counter)")
			new_node.id = nodes_counter
			if debug[] == true
				println(new_node)
			end
			nodes_counter+=1
			for (action, observation_map) in new_node.edges
				for (observation, edge_map) in observation_map
					#@deb("Obs $observation")
					for (next, prob) in edge_map
						#@deb("adding incoming edge from $(new_node.id) to $(next.id) ($action, $observation)")
						if haskey(next.incomingEdgeDicts, new_node)
							#@deb("it was the $(length(next.incomingEdgeDicts[new_node])+1)th")
							push!(next.incomingEdgeDicts[new_node], edge_map)
						else
							#@deb("it was the first edge for $(new_node.id)")
							next.incomingEdgeDicts[new_node] = [edge_map]
						end
					end
				end
			end
		end
		new_max_id = nodes_counter-1
		#add new nodes to controller
		#i want to have the new nodes first, so in case there are two nodes with identical value the newer node is pruned and we skip rewiring
		#no need to sort, just have new nodes examined before old nodes
		#the inner union is needed to have an orderedset as first parameter, as the result of union({ordered}, {not ordered}) = {ordered}
		orderedset = union(union(OrderedSet{Node}(), new_nodes), Set{Node}(oldnode for oldnode in values(nodes)))
		all_nodes = filterNodes(orderedset, minval)
		new_controller_nodes = Dict{Int64, Node}()
		for node in all_nodes
			#add nodes to the controller
			new_controller_nodes[node.id] = node
		end
		controller.nodes = new_controller_nodes
		controller.maxId = new_max_id
	end

	#using eq τ(n, a, z) from incremental pruning paper
	function node_value(node::Node{A, W}, action::A, observation::W, pomdp::POMDP) where {A, W}
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
				p_obs = POMDPModelTools.pdf(POMDPs.observation(pomdp, action, s_prime), observation)
				@deb("$action, $state, $observation, $s_prime")
				@deb("$(node.value[s_prime_index]) * $(p_obs) * $(p_s_prime)")
				sum+= node.value[s_prime_index] * p_obs * p_s_prime
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
		@deb("Called incprune, startId = $startId")
		id = startId
		id, xs = xsum(nodeVec[1], nodeVec[2], id)
		res = filterNodes(xs, minval)
		#=if debug[] == true
			for node in res
				println(node)
			end
		end
		=#
		for i = 3:length(nodeVec)
			id, xs = xsum(xs, nodeVec[i], id)
			res = filterNodes(xs, minval)
			#@deb("Length $i = $(length(nodeVec[i]))")
		end
		#@deb("returned ID = $id")
		return id, res
	end

	function xsum(A::Set{Node}, B::Set{Node}, startId::Int64)
		@deb("Called xsum, startId = $startId")
		X = Set{Node}()
		id = startId
	    for a in A, b in B
			#each of the newly generated nodes only has one action!
			@assert length(a.actionProb) == length(b.actionProb) == 1 "more than one action in freshly generated node"
			a_action = collect(keys(a.actionProb))[1]
			b_action = collect(keys(b.actionProb))[1]
			@assert a_action == b_action "action mismatch"
			c = mergeNode(a, b, a_action, id)
			if debug[] == true
				println("nodes merged: $id<- $(a.id), $(b.id)")
				println(c)
			end
			id-=1
			push!(X, c)
	    end
		#@deb("returned id = $id")
	    return id, X
	end

	function mergeNode(a::Node{A, W}, b::Node{A, W}, action::A, id::Int64) where {A, W}
		a_observation_map = a.edges[action]
		b_observation_map = b.edges[action]

		#There is no way we have repeated observation because set[obs]
		#only contains nodes with obs
		actionProb = Dict{A, Float64}(action => 1.0)
		edges = Dict{A, Dict{W, Dict{Node, Float64}}}(action => Dict{W, Dict{Node, Float64}}())
		for (obs, node_map) in a_observation_map
			edges[action][obs] = Dict{Node, Float64}()
			for (node, prob) in node_map
				edges[action][obs][node] = prob
			end
		end
		for (obs, node_map) in b_observation_map
			if haskey(edges[action], obs)
				error("Observation already present")
			end
			edges[action][obs] = Dict{Node, Float64}()
			for (node, prob) in node_map
				edges[action][obs][node] = prob
			end
		end
		return Node(id, actionProb, edges, a.value+b.value, Dict{Node, Vector{Dict{Node, Float64}}}())
	end

	"""
		Filtering function to remove dominated nodes.
		Minval is the minimum probability that an edge can have. values below minval are treated as zero, values above 1-minval are treated as 1
	"""
	function filterNodes(nodes::Set{Node}, minval::Float64)
		@deb("Called filterNodes, length(nodes) = $(length(nodes))")
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
	        lpmodel = JuMP.Model(with_optimizer(GLPK.Optimizer))
	        #define variables for LP. c(i)
	        @variable(lpmodel, c[i=keys(new_nodes)] >= 0)
	        #e to maximize
	        @variable(lpmodel, e)
	        @objective(lpmodel, Max, e)
	        @constraint(lpmodel, con[s_index=1:n_states], n.value[s_index] + e <= sum(c[ni_id]*ni.value[s_index] for (ni_id, ni) in new_nodes))
			#this constraint is used to avoid having edges with extremely low probability
			#@constraint(lpmodel, con_small[i=keys(new_nodes)], c[i]*(c[i]-minval) >= 0)
			#not supported by solver =(
	        @constraint(lpmodel, con_sum, sum(c[i] for i in keys(new_nodes)) == 1)
	        optimize!(lpmodel)
			if debug[] == true
				println("node $(n.id) -> eps = $(JuMP.value(e))")
			end
	        if JuMP.value(e) >= -1e-12
	            #rewiring function here!
				if debug[] == true
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
								@deb("length of dict coming from $(src_node.id) == 0 after deletion, removing")
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
				if debug[] == true
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
		@deb("Called filterNodes, length(nodes) = $(length(nodes))")
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
			@deb("temp = $node_counter, id = $(node.id)")
			@deb(node)
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
	        lpmodel = JuMP.Model(with_optimizer(GLPK.Optimizer))
	        #define variables for LP. c(i)
	        @variable(lpmodel, 0.0 <= c[i=keys(new_nodes)] <= 1.0)
	        #e to maximize
	        @variable(lpmodel, e)
	        @objective(lpmodel, Max, e)
	        @constraint(lpmodel, con[s_index=1:n_states], n.value[s_index] + e <= sum(c[ni_id]*ni.value[s_index] for (ni_id, ni) in new_nodes))
	        @constraint(lpmodel, con_sum, sum(c[i] for i in keys(new_nodes)) == 1)
	        optimize!(lpmodel)
			if debug[] == true
				println("node $(n.id) -> eps = $(JuMP.value(e))")
			end
	        if JuMP.value(e) >= -1e-10
	            #rewiring function here!
				if debug[] == true
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
										@deb("updated probability of edge from node $(src_node.id) to $(dst_node.id)")
										dict[dst_node]+= v*old_prob
										if dict[dst_node] > 1
											error("probability > 1 after redirection!")
										end
									else
										@deb("Added edge from node $(src_node.id) to $(dst_node.id)")
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
				if debug[] == true
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


	function evaluate!(controller::Controller{A,W}, pomdp::POMDP{A, W}) where {A, W}
			#solve V(n,s) = R(s, a(n)) + gamma*sumz(P(s'|s,a(n))Pr(z|s',a(n))V(beta(n,z), s'))
			#R(s,a(n)) is the reward function
			nodes = controller.nodes
			n_nodes = length(keys(controller.nodes))
			states = POMDPs.states(pomdp)
			n_states = POMDPs.n_states(pomdp)
			M = zeros(n_states*n_nodes, n_states*n_nodes)
			b = zeros(n_states*n_nodes)

			#dictionary used for recompacting ids
			temp_id = Dict{Int64, Int64}()
			for (node_id, node) in nodes
				temp_id[node_id] = length(temp_id)+1
			end

			#compute coefficients for sum(a)[R(s|a)*P(a|n)+gamma*sum(z, n', s')[P(s'|s,a)*P(z|s',a)*P(a|n)*P(n'|z)*V(nz, s')]]
			for (n_id, node) in nodes
				#M is the coefficient matrix (form x1 = a2x2+...+anxn+b)
				#b is the constant term vector
				#variables are all pairs of n,s
				actions = getPossibleActions(node)
				for s_index in 1:n_states
					s = POMDPs.states(pomdp)[s_index]
					for a in actions
						@deb("action = $a")
						p_a_n = node.actionProb[a]
						b[composite_index([temp_id[n_id], s_index],[n_nodes, n_states])] = POMDPs.reward(pomdp, s, a)*p_a_n
						@deb("b($n_id, $s) = $(POMDPs.reward(pomdp, s, a)*p_a_n)")
						M[composite_index([temp_id[n_id], s_index],[n_nodes, n_states]), composite_index([temp_id[n_id], s_index],[n_nodes, n_states])] += 1
						@deb("M[$n_id, $s][$n_id, $s] = 1")
						s_primes = POMDPs.transition(pomdp,s,a).vals
						possible_obs = keys(node.edges[a])  #only consider observations possible from current node/action combo
						for obs in possible_obs
							@deb("obs = $obs")
							for s_prime_index in 1:length(s_primes)
								s_prime = s_primes[s_prime_index]
								p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp,s,a), s_prime)
								if p_s_prime == 0.0
									continue
								end
								p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, a, s_prime), obs)
								@deb("p_z = $p_z")
								for (next, prob) in node.edges[a][obs]
									if !haskey(controller.nodes, next.id)
										error("Node $(next.id) not present in nodes")
									end
									M[composite_index([temp_id[n_id], s_index],[n_nodes, n_states]), composite_index([temp_id[next.id], s_prime_index],[n_nodes,n_states])]-= POMDPs.discount(pomdp)*p_s_prime*p_z*p_a_n*prob
									@deb("M[$n_id, $s][$(next.id), $s_prime] = gamma=$(POMDPs.discount(pomdp))*ps'=$p_s_prime*pz=$p_z*pa=$p_a_n*pn'=$prob = $(M[composite_index([temp_id[n_id], s_index],[n_nodes, n_states]), composite_index([temp_id[next.id], s_prime_index],[n_nodes,n_states])])")
								end
							end
						end
					end
				end
			end
			@deb("M = $M")
			@deb("b = $b")
			res = M \ b
			#copy respective value functions in nodes
			for (n_id, node) in nodes
				node.value = copy(res[(temp_id[n_id]-1)*n_states+1 : temp_id[n_id]*n_states])
				@deb("Value vector of node $n_id = $(nodes[n_id].value)")
			end
	end
	"""
	Given multiple indexes of a multidimensional matrix with dimension specified by lengths return the index in the corresponding 1D vector
	lengths[1] is actually never used, but it is there for consistency (can actually be set to any number)
	"""
function composite_index(dimension::Vector{Int64}, lengths::Vector{Int64})
	#return (primary-1)*secondary_len+secondary
	if length(dimension) != length(lengths)
		error("Dimension and lengths vector have different length!")
	end
	for d in 1:length(dimension)
		if dimension[d] > lengths[d]
			error("Dimension cannot be greater than dimension length")
		end
	end
	index = 0
	for i in 1:length(dimension)
		index= index*lengths[i]+(dimension[i]-1)
	end
	return index+1
end

function partial_backup!(controller::Controller{A, W}, pomdp::POMDP{A, W}; minval = 0.0, add_one = true, debug_node = 0) where {A, W}
	#this time the matrix form is a1x1+...+anxn = b1
	#sum(a,s)[sum(nz)[canz*[R(s,a)+gamma*sum(s')p(s'|s, a)p(z|s', a)v(nz,s')]] -eps = V(n,s)
	#number of variables is |A||Z||N|+1 (canz and eps)
	nodes = controller.nodes
	n_nodes = length(keys(controller.nodes))
	states = POMDPs.states(pomdp)
	n_states = POMDPs.n_states(pomdp)
	actions = POMDPs.actions(pomdp)
	n_actions = POMDPs.n_actions(pomdp)
	observations = POMDPs.observations(pomdp)
	n_observations = POMDPs.n_observations(pomdp)
	#vector containing the tangent belief states for all modified nodes
	tangent_b = Dict{Int64, Array{Float64}}()
	#dim = n_nodes*n_actions*n_observations
	changed = false
	node_counter = 1
	M_TR =  zeros(n_actions, n_observations, n_nodes)
	M_TL =  zeros(n_actions, n_observations, n_nodes)
	temp_id = Dict{Int64, Int64}()
	for real_id in keys(nodes)
			temp_id[real_id] = node_counter
			@deb("Node $real_id becomes $node_counter")
			node_counter+=1
	end
	for (n_id, node) in nodes
		@deb("Node to be improved: $n_id")
		lpmodel = JuMP.Model(with_optimizer(GLPK.Optimizer))
		#define variables for LP. c(a, n, z)
		@variable(lpmodel, canz[a=1:n_actions, z=1:n_observations, n=1:n_nodes] >= 0.0)
		@variable(lpmodel, ca[a=1:n_actions] >= 0.0)
		#e to maximize
		@variable(lpmodel, e)
		@objective(lpmodel, Max, e)
		#define constraints
		for s_index in 1:n_states
			s = states[s_index]
			M = zeros(n_actions, n_observations, n_nodes)
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
						p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, action, s_prime), obs)
						if p_s_prime != 0.0 && p_z != 0.0
							for (nz_id, nz) in nodes
								v_nz_sp = nz.value[POMDPs.stateindex(pomdp, s_prime)]
								#if n_id == 7 || n_id == 8
									#@deb("state = $s, action = $action, obs = $obs, nz = $(nz_id), s_prime = $s_prime")
									#@deb("$p_s_prime $p_z $v_nz_sp")
								#end
								M[a_index, obs_index, temp_id[nz_id]]+= p_s_prime*p_z*v_nz_sp
							end
						end
						#@deb("M[$a_index, $obs_index, $nz_id] = $(M[a_index, obs_index, nz_id])")
					end
				end
			end
			if debug[] == true
				if n_id == debug_node
					if s_index == 1
						M_TR = M
					else
						M_TL = M
					end
					#=
					L = 3
					GL = 1
					GR = 2
					TR = 1
					TL = 2
					println("correct vals")
					correct_val = -1 + POMDPs.discount(pomdp)*(M[L, GL, temp_id[10]] +  M[L, GR, temp_id[8]]) - node.value[s_index]
					println("e <= $correct_val")
					=#
				end
			end
			#=
			if debug[] == true
				for a in 1:n_actions
					println("Action $(actions[a])")
					for z in 1:n_observations
						println("Obs $(observations[z])")
						for nz in keys(nodes)
							println("Node $nz: $(M[a,z,temp_id[nz]])")
						end
					end
				end
			end
			=#
			#@deb("state $s: $M_a")
			#constraint on the big formula in table 2
			#@constraint(lpmodel,  e - M.*canz .<= -1*node.value[s_index])
			#n are actually temp_ids here
			#@deb(M)
			con = @constraint(lpmodel,  e + node.value[s_index] <= sum( M_a[a]*ca[a]+POMDPs.discount(pomdp)*sum(sum( M[a, z, n] * canz[a, z, n] for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions))
			set_name(con, "$(s)")
		end
		#sum canz over a,n,z = 1
		@constraint(lpmodel, con_sum[a=1:n_actions, z=1:n_observations], sum(canz[a, z, n] for n in 1:n_nodes) == ca[a])
		@constraint(lpmodel, ca_sum, sum(ca[a] for a in 1:n_actions) == 1.0)

		if debug[] == true
			print(lpmodel)
		end

		optimize!(lpmodel)
		#@deb("$(termination_status(lpmodel))")
		#@deb("$(primal_status(lpmodel))")
		#@deb("$(dual_status(lpmodel))")

		if debug[] == true && n_id == debug_node
			L = 3
			GL = 2
			GR = 1
			TR = 1
			TL = 2
			println("correct vals")
			correct_val = -1 + POMDPs.discount(pomdp)*(M_TR[L, GL, temp_id[5]] +  M_TR[L, GR, temp_id[8]]) - node.value[TR]
			println("e <= $correct_val")
			correct_val = -1 +POMDPs.discount(pomdp)*(M_TL[L, GL, temp_id[5]] +  M_TL[L, GR, temp_id[8]]) - node.value[TL]
			println("e <= $correct_val")

			#actual_val = -1 + POMDPs.discount(pomdp)*(M_7_TR[L, GL, temp_id[10]]* JuMP.value(canz[L, GL, temp_id[10]]) +  M_7_TR[L, GR, temp_id[8]] * JuMP.value(canz[L, GR, temp_id[8]])+ M_7_TR[L, GR, temp_id[1]] * JuMP.value(canz[L, GR,temp_id[1]])) - node.value[TR]
			actual_val = -1 + POMDPs.discount(pomdp)* sum(sum(sum( M_TR[a,z,n] * JuMP.value(canz[a,z,n])  for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions) - node.value[TR]
			println("actual val")
			println("e <= $actual_val")
			#actual_val = -1 + POMDPs.discount(pomdp)*(M_7_TL[L, GL, temp_id[10]]* JuMP.value(canz[L, GL, temp_id[10]]) +  M_7_TL[L, GR, temp_id[8]] * JuMP.value(canz[L, GR, temp_id[8]])+ M_7_TL[L, GR, temp_id[1]] * JuMP.value(canz[L, GR, temp_id[1]])) - node.value[TL]
			actual_val = -1 + POMDPs.discount(pomdp)* sum(sum(sum( M_TL[a,z,n]*JuMP.value(canz[a,z,n]) for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions) - node.value[TL]

			println("e <= $actual_val")
			for n_id in keys(nodes)

				println("canz $n_id =  $(JuMP.value(canz[L, GR, temp_id[n_id]]))")
				println("canz $n_id =  $(JuMP.value(canz[L, GL, temp_id[n_id]]))")

			end
		end
		#=if debug[] == true && n_id == 8 && haskey(controller.nodes, 9) && haskey(controller.nodes, 7)
			L = 3
			GL = 1
			GR = 2
			TR = 1
			TL = 2
			println("correct vals")
			correct_val = -1 + POMDPs.discount(pomdp)*(M_8_TR[L, GL, temp_id[7]] +  M_8_TR[L, GR, temp_id[9]]) - node.value[TR]
			println("e <= $correct_val")
			correct_val = -1 + POMDPs.discount(pomdp)*(M_8_TL[L, GL, temp_id[7]] +  M_8_TL[L, GR, temp_id[9]]) - node.value[TL]
			println("e <= $correct_val")

			actual_val = -1 + POMDPs.discount(pomdp)*(M_8_TR[L, GL, temp_id[7]]* JuMP.value(canz[L, GL, temp_id[7]]) +  M_8_TR[L, GR, temp_id[9]] * JuMP.value(canz[L, GR, temp_id[9]])+ M_8_TR[L, GL, temp_id[1]] * JuMP.value(canz[L, GL, temp_id[1]])) - node.value[TR]
			println("actual val")
			println("e <= $actual_val")
			actual_val = -1 + POMDPs.discount(pomdp)*(M_8_TL[L, GL, temp_id[7]]* JuMP.value(canz[L,GL,temp_id[7]]) +  M_8_TL[L, GR, temp_id[9]] * JuMP.value(canz[L, GR, temp_id[9]])+ M_8_TL[L, GL, temp_id[1]] * JuMP.value(canz[L, GL, temp_id[1]])) - node.value[TL]
			println("e <= $actual_val")


		end
		=#
		@deb("eps = $(JuMP.value(e))")
		@deb("Obj = $(objective_value(lpmodel))")
		if JuMP.objective_value(lpmodel) > minval
			changed = true
			#@deb("Good so far")
			new_edges = Dict{A, Dict{W,Dict{Node, Float64}}}()
			new_actions = Dict{A, Float64}()
			#@deb("New structures created")
			for action_index in 1:n_actions
				ca_v = JuMP.value(ca[action_index])
				@deb("Action $(actions[action_index])")
				@deb("ca $(actions[action_index])= $ca_v")
				if ca_v > 1.0-minval
					ca_v = 1.0
				end
				if ca_v > minval
					new_obs = Dict{W, Dict{Node, Float64}}()
					for obs_index in 1:n_observations
						obs_total = 0.0
						#fill a temporary edge dict with unnormalized probs
						temp_edge_dict = Dict{Node, Float64}()
						for (nz_id, nz) in nodes
							prob = JuMP.value(canz[action_index, obs_index, temp_id[nz_id]])/ca_v
							#@deb("canz $(observations[obs_index]) -> $nz_id = $prob")
							if prob < 0.0
								@deb("Set prob to 0 even though it was negative")
								prob = 0.0
							end
							if prob > 1.0 && prob < 1.0+minval
								@deb("Set prob slightly greater than 1 to 1")
								prob = 1.0
							end
							if prob < 0.0 || prob > 1.0
								error("Probability outside of bounds: $prob")
							end
							if prob > 0.0
								obs_total+= prob
								#@deb("New edge: $(action_index), $(obs_index) -> $nz_id, $(prob)")
								temp_edge_dict[nz] = prob
							end
						end
						if obs_total == 0.0
							error("sum of prob for obs $(observations[obs_index]) == 0")
						end
						new_edge_dict = Dict{Node, Float64}()
						for (next, prob) in temp_edge_dict
							#@deb("normalized prob: $normalized_prob")
							if prob >= 1.0-minval
								new_edge_dict[next] = 1.0
							elseif prob > minval
								new_edge_dict[next] = prob
							end
							#do not add anything if prob < minval
						end
						#@deb("length of dict for obs $(observations[obs_index]) = $(length(new_edge_dict))")
						if length(new_edge_dict) != 0
							new_obs[observations[obs_index]] = new_edge_dict
							#update incoming edge vector for other node
							for (next, prob) in new_edge_dict
								if haskey(next.incomingEdgeDicts, node)
									push!(next.incomingEdgeDicts[node], new_edge_dict)
								else
									next.incomingEdgeDicts[node] = [new_edge_dict]
								end
							end
						end
					end
					if length(keys(new_obs)) != 0
						new_edges[actions[action_index]] = new_obs
						new_actions[actions[action_index]] = ca_v
					end
				end
			end
			node.edges = new_edges
			node.actionProb = new_actions
			if !add_one
				old_deb = debug[]
				debug[] = false
				#evaluate!(controller, pomdp)
				debug[] = old_deb
				if debug[] == true
					println("Changed controller after eval")
					for (n_id, node) in controller.nodes
						println(node)
					end
				end
			end
			if add_one
				#no need to update tangent points because they wont be used!
				if debug[] == true
					println("Changed node after eval")
					println(node)
				end
				return true, Dict{Int64, Array{Float64}}()
			end
		end

		#constraint_list = JuMP.all_constraints(lpmodel, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
		# temp_tangent_b = zeros(n_states)
		# for s_index in 1:n_states
		# 	temp_tangent_b[s_index] = -1*dual(constraint_list[s_index])
		# 	s = states[s_index]
		# 	if temp_tangent_b[s_index] !=  -1*dual(constraint_by_name(lpmodel, "$(s)"))
		# 		println("Different value between two techniques")
		# 	end
		# end
		# tangent_b[n_id] = temp_tangent_b
		tangent_b[n_id] = [-1*dual(constraint_by_name(lpmodel, "$(s)")) for s in states]
	end
	return changed, tangent_b
end

function escape_optima_standard!(controller::Controller{A, W}, pomdp::POMDP{A, W}, tangent_b::Dict{Int64, Array{Float64}}; add_one=false, minval = 0.0) where {A, W}
	#@deb("$tangent_b")
	nodes = controller.nodes
	n_nodes = length(keys(controller.nodes))
	states = POMDPs.states(pomdp)
	n_states = POMDPs.n_states(pomdp)
	actions = POMDPs.actions(pomdp)
	n_actions = POMDPs.n_actions(pomdp)
	observations = POMDPs.observations(pomdp)
	n_observations = POMDPs.n_observations(pomdp)
	if length(tangent_b) == 0
		error("tangent_b was empty!")
	end

	old_deb = debug[]
	debug[] = false

	new_nodes = full_backup_generate_nodes(controller, pomdp, minval)
	#new_nodes = collect(values(backed_up_controller.nodes))

	debug[] = old_deb
	#if debug[] == true
	#	println("new_nodes:")
	#	for node in new_nodes
	#		println(node)
	#	end
	#end


	escaped = false
	reachable_b = Set{Vector{Float64}}()
	for (id, start_b) in tangent_b
		#id = collect(keys(tangent_b))[1]
		#start_b = tangent_b[id]
		@deb("$id - >$start_b")
		for a in keys(nodes[id].actionProb)
			for z in observations
				new_b = Vector{Float64}(undef, n_states)
				normalize = 0.0
				for s_prime_index in 1:n_states
					s_prime = states[s_prime_index]
					sum_s = 0.0
					p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, a, s_prime), z)
					for s_index in 1:n_states
						s = states[s_index]
						p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp, s, a), s_prime)
						sum_s+= p_s_prime*start_b[s_index]
					end
					new_b[s_prime_index] = p_z * sum_s
					normalize += p_z * sum_s
				end
				for i in 1:n_states
					new_b[i] = new_b[i] / normalize
				end
				@deb("from belief $start_b action $a and obs $z -> $new_b")
				push!(reachable_b, new_b)
				#find the value of the current controller in the reachable belief state
				best_old_node = nothing
				best_old_value = 0.0
				for (id,old_node) in controller.nodes
					temp_value = new_b' * old_node.value
					if best_old_node == nothing || best_old_value < temp_value
						best_old_node = old_node
						best_old_value = temp_value
					end
				end
				#find the value of the backed up controller in the reachable belief state
				best_new_node = nothing
				best_new_value = 0.0
				for new_node in new_nodes
					new_value = new_b' * new_node.value
					if best_new_node == nothing || best_new_value < new_value
						best_new_node = new_node
						best_new_value = new_value
					end
				end
				if best_new_value - best_old_value > minval
					@deb("in $new_b node $(best_new_node.id) has $best_new_value > $best_old_value")
					reworked_node = rework_node(controller, best_new_node)
					controller.nodes[reworked_node.id] = reworked_node
					controller.maxId+=1
					@deb("Added node $(reworked_node.id)")

					#experimental: redirect edge of chosen node to newly added nodes
					#=
					old_prob = 1.0 #nodes[id].edges[a][z][best_old_node]
					nodes[id].edges[a][z] = Dict(reworked_node => old_prob)
					@deb("edge redirected from $(best_old_node.id) to $(reworked_node.id)")
					=#

					if debug[] == true
						println(controller.nodes[reworked_node.id])
					end
					if add_one
						return true
					end
					escaped = true
				end
			end
		end
		if escaped && !add_one
			return true
		end
	end
	#@deb("$reachable_b")
	return escaped
end

function rework_node(controller::AbstractController, new_node::Node{A, W}) where {A, W}
		id = controller.maxId+1
		actionProb = copy(new_node.actionProb)
		value = copy(new_node.value)
		edges = Dict{A, Dict{W, Dict{Node, Float64}}}()
		for (a, obs_dict) in new_node.edges
			edges[a] = Dict{W, Dict{Node, Float64}}()
			for (z, node_dict) in obs_dict
				edges[a][z] = Dict{Node,Float64}()
				for (node, prob) in node_dict
					current_controller_node = controller.nodes[node.id]
					edges[a][z][current_controller_node] = prob
				end
			end
		end
		return Node(id, actionProb,edges, value, Dict{Node, Vector{Dict{Node, Float64}}}())
end


#include("bpigraph.jl")

function improveNode(controller::Controller{A,W}, old_node::Node{A, W}) where {A, W}
	new_node = unconstrained_node_impro()
	swap_nodes!(controller, old_node, new_node)
	evaluate(controller, controller.frame)
end
