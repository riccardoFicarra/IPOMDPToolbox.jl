#=
IBPIPolicyUtils:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
	using LinearAlgebra
	using DataStructures
	using JuMP
	using GLPK

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
		value::Vector{Float64}
		#needed to efficiently redirect edges during pruning
		#srcNode -> vectors of dictionaries that contains edge to this node
		incomingEdgeDicts::Dict{Node, Vector{Dict{Node, Float64}}}
	end

	#overload hash and isequal to use only id as keys in dicts
	Base.hash(n::Node) = hash(n.id)
	Base.isequal(n1::Node, n2::Node) = Base.isequal(hash(n1), hash(n2))
	#overload display function to avoid walls of text when printing nodes
	Base.display(n::IPOMDPToolbox.Node) = println(n)

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
		for (src_node, dict_vect) in node.incomingEdgeDicts
			for dict in dict_vect
				for (next, prob) in dict
					println("from node $(src_node.id) p=$(prob) to node $(next.id)")
				end
			end
		end
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
			for obs in observations
				edges = Dict{Node, Float64}(n => 1.0)
				obsdict[obs] = edges
				n.incomingEdgeDicts[n] = [edges]
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


	mutable struct Controller{A, W}
		nodes::Dict{Int64, Node{A, W}}
		maxId::Int64
	end
	"""
	Initialize a controller with the initial node, start id counter from 2
	"""
	function Controller(pomdp::POMDP{A,W}, force::Int64) where {A, W}
		newNode = InitialNode(pomdp, force)
		Controller{A, W}(Dict(1 => newNode), 1)

	end

	function build_node(node_id::Int64, actions::Vector{A}, actionProb::Vector{Float64}, observations::Vector{Vector{W}}, observation_prob::Vector{Vector{Float64}}, next_nodes::Vector{Vector{Node{A,W}}}, value::Vector{Float64}) where {A, W}
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
	function build_node(node_id::Int64, action::A, observation::W, next_node::Node{A, W}, value::Vector{Float64}) where {A, W}
		actionprob = Dict{A, Float64}(action => 1.0)
		edges = Dict{A, Dict{W, Dict{Node, Float64}}}(action => Dict{W, Dict{Node, Float64}}(observation => Dict{Node, Float64}(next_node=> 1.0)))
		return Node(node_id, actionprob, edges, value, Dict{Node, Vector{Dict{Node, Float64}}}())
	end
	"""
	Perform a full backup operation according to Pourpart and Boutilier's paper on Bounded finite state controllers
	TODO: make this thing actually do a backup
	"""
	function full_backup_stochastic!(controller::Controller{A, W}, pomdpmodel::pomdpModel) where {A, W}
		pomdp = pomdpmodel.frame
		belief = pomdpmodel.history.b
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
				if IPOMDPToolbox.debug[] == true
					println("New nodes created:")
					for node in new_nodes_a_z
						println(node)
					end
				end
				new_nodes_z[obs_index] = filterNodes(new_nodes_a_z)
			end
			#set that contains all nodes generated from action a after incremental pruning
			new_nodes_counter, new_nodes_a = incprune(new_nodes_z, new_nodes_counter)
			union!(new_nodes, new_nodes_a)
		end
		#all new nodes, final filtering
		new_nodes = filterNodes(new_nodes)
		#before performing filtering with the old nodes update incomingEdge structure of old nodes
		#also assign permanent ids
		nodes_counter = controller.maxId+1
		for new_node in new_nodes
			@deb("Node $(new_node.id) becomes node $(nodes_counter)")
			new_node.id = nodes_counter
			nodes_counter+=1
			for (action, observation_map) in new_node.edges
				for (observation, edge_map) in observation_map
					#@deb("Obs $observation")
					for (next, prob) in edge_map
						@deb("adding incoming edge from $(new_node.id) to $(next.id) ($action, $observation)")
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
		all_nodes = filterNodes(orderedset)
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
	function incprune(nodeVec::Vector{Set{Node}}, startId::Int64)
		@deb("Called incprune, startId = $startId")
		id = startId
		id, xs = xsum(nodeVec[1], nodeVec[2], id)
		res = filterNodes(xs)
		#=if debug[] == true
			for node in res
				println(node)
			end
		end
		=#
		for i = 3:length(nodeVec)
			id, xs = xsum(nodeVec[1], nodeVec[2], id)
			res = filterNodes(xs)
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
			if IPOMDPToolbox.debug[] == true
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
		Filtering function to remove dominated nodes
	"""
	function filterNodes(nodes::Set{IPOMDPToolbox.Node})
	    @deb("Called filterNodes, length(nodes) = $(length(nodes))")
		if length(nodes) == 0
			error("called FilterNodes on empty set")
		end
		if length(nodes) == 1
			#if there is only one node it is useless to try and prune anything
			return nodes
		end
	    new_nodes = Dict{Int64, IPOMDPToolbox.Node}()
	    #careful, here dict key != node.id!!!!
	    node_counter = 1
	    for node in nodes
	        new_nodes[node_counter] = node
	        node_counter+=1
	    end
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
	        @constraint(lpmodel, con_sum, sum(c[i] for i in keys(new_nodes)) == 1)
	        optimize!(lpmodel)
			if debug[] == true
				println("node $(n.id) -> eps = $(JuMP.value(e))")
			end
	        if JuMP.value(e) >= -1e-14
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
								if v > 0.0
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
								@deb("removed incoming edge from $(n.id) to $(next.id) ($action, $observation)")
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

	    return Set{IPOMDPToolbox.Node}(node for node in values(new_nodes))
	end

	"""
		Filtering function to remove dominated nodes
	"""
	function filterNodes(nodes::OrderedSet{IPOMDPToolbox.Node})
	    @deb("Called filterNodes, length(nodes) = $(length(nodes))")
		if length(nodes) == 0
			error("called FilterNodes on empty set")
		end
		if length(nodes) == 1
			#if there is only one node it is useless to try and prune anything
			return nodes
		end
	    new_nodes = OrderedDict{Int64, IPOMDPToolbox.Node}()
	    #careful, here dict key != node.id!!!!
	    node_counter = 1
	    for node in nodes
	        new_nodes[node_counter] = node
			@deb("$(node.id)")
	        node_counter+=1
	    end
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
	        @constraint(lpmodel, con_sum, sum(c[i] for i in keys(new_nodes)) == 1)
	        optimize!(lpmodel)
			if debug[] == true
				println("node $(n.id) -> eps = $(JuMP.value(e))")
			end
	        if JuMP.value(e) >= -1e-14
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
								if v > 0.0
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
								@deb("removed incoming edge from $(n.id) to $(next.id) ($action, $observation)")
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

	    return Set{IPOMDPToolbox.Node}(node for node in values(new_nodes))
	end


	function evaluate!(controller::Controller{A,W}, pomdpmodel::pomdpModel) where {A, W}
			#solve V(n,s) = R(s, a(n)) + gamma*sumz(P(s'|s,a(n))Pr(z|s',a(n))V(beta(n,z), s'))
			#R(s,a(n)) is the reward function
			pomdp = pomdpmodel.frame
			nodes = controller.nodes
			n_nodes = length(keys(controller.nodes))
			states = POMDPs.states(pomdp)
			n_states = POMDPs.n_states(pomdp)
			M = spzeros(n_states*n_nodes, n_states*n_nodes)
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
						b[composite_index([temp_id[n_id], s_index],[n_nodes, n_states])] += POMDPs.reward(pomdp, s, a)*node.actionProb[a]
						s_primes = POMDPs.transition(pomdp,s,a).vals
						possible_obs = keys(node.edges[a])  #only consider observations possible from current node/action combo
						for obs in possible_obs
							for s_prime_index in 1:length(s_primes)
								s_prime = s_primes[s_prime_index]
								p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp,s,a), s_prime)
								p_a_n = node.actionProb[a]
								p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, s_prime, a), obs)
								for (next, prob) in node.edges[a][obs]
									if !haskey(controller.nodes, next.id)
										error("Node $(next.id) not present in nodes")
									end
									nz_index = temp_id[next.id]
									c_a_nz = prob*node.actionProb[a] #CHECK THAT THIS IS THE RIGHT VALUE (page 5 of BPI paper)
									M[composite_index([temp_id[n_id], s_index],[n_nodes, n_states]), composite_index([nz_index, s_prime_index],[n_nodes,n_states])]+= POMDPs.discount(pomdp)*p_s_prime*p_z*p_a_n*c_a_nz
								end
							end
						end
					end
				end
			end
			@deb("M = $M")
			@deb("b = $b")
			#maybe put this in coefficient computation instead of doing matrix operations for faster comp?
			I = Diagonal(ones(Float64,size(M,1), size(M,2) ))
			res = (I- M) \ b
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

function partial_backup!(controller::Controller{A, W}, pomdpmodel::pomdpModel) where {A, W}
	#this time the matrix form is a1x1+...+anxn = b1
	#sum(a,s)[sum(nz)[canz*[R(s,a)+gamma*sum(s')p(s'|s, a)p(z|s', a)v(nz,s')]] -eps = V(n,s)
	#number of variables is |A||Z||N|+1 (canz and eps)
	pomdp = pomdpmodel.frame
	nodes = controller.nodes
	n_nodes = length(keys(controller.nodes))
	states = POMDPs.states(pomdp)
	n_states = POMDPs.n_states(pomdp)
	actions = POMDPs.actions(pomdp)
	n_actions = POMDPs.n_actions(pomdp)
	observations = POMDPs.observations(pomdp)
	n_observations = POMDPs.n_observations(pomdp)
	#dim = n_nodes*n_actions*n_observations
	changed = false
	for (n_id, node) in nodes
		lpmodel = JuMP.Model(with_optimizer(GLPK.Optimizer))
		#define variables for LP. c(a, n, z)
		@variable(lpmodel, c[a=1:n_actions, z=1:n_observations, n=keys(nodes)] >= 0)
		#e to maximize
		@variable(lpmodel, e)
		@objective(lpmodel, Max, e)
		#define constraints
		for s_index in 1:n_states
			s = states[s_index]
			M = zeros(n_actions, n_observations, n_nodes)
			for a_index in 1:n_actions
				action = actions[a_index]
				r_s_a = POMDPs.reward(pomdp, s, action)
				s_primes = POMDPs.transition(pomdp,s,action).vals
				for obs_index in 1:n_observations
					obs = observations[obs_index]
					#array of edges given observation
					for s_prime in s_primes
						for (nz_id, nz) in nodes
							#comp_eq_index = composite_index([a_index, obs_index, n_id], [n_actions,n_observations, n_nodes])
							#comp_var_index = composite_index([a_index, obs_index, nz_id], [n_actions,n_observations, n_nodes])
							p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp,s,action), s_prime)
							p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, s_prime, action), obs)
							v_nz_sp = nz.value[POMDPs.stateindex(pomdp, s_prime)]
							#@deb("obs = $obs, nz = $(nz_id), action = $action, , state = $s, s_prime = $s_prime")
							M[a_index, obs_index, nz_id] += r_s_a+POMDPs.discount(pomdp)*p_s_prime*p_z*v_nz_sp
						end
					end
				end
			end
			@constraint(lpmodel, [s_index],  e - M.*c .<= -1*node.value[s_index])
		end
		@expression(lpmodel, sumc, sum(sum(sum(c[a,z,n] for n in keys(nodes)) for z in 1:n_observations) for a in 1:n_actions))
		@constraint(lpmodel, con_sum,  sumc == 1)
		#@constraint(lpmodel, canz_prob[a=1:n_actions, z=1:n_observations, n=1:n_nodes], 0 <= c[a,z,n] <= 1)
		#print(lpmodel)
		optimize!(lpmodel)


		if JuMP.value(e) >= 0
			changed = true
			#@deb("Good so far")
			new_edges = Dict{A, Dict{W,Dict{Int64, Float64}}}()
			new_actions = Dict{A, Float64}()
			#@deb("New structures created")
			for action_index in 1:n_actions
				ca = 0
				new_obs = Dict{W, Dict{Int64, Float64}}()
				for obs_index in 1:n_observations
					new_edge_dict = Dict{Int64, Float64}()
					for (nz_id, nz) in nodes
						prob = JuMP.value(c[action_index, obs_index, nz_id])
						if abs(prob) < 1e-15
							@deb("Set prob to 0 even though it was negative")
							prob = 0
						end
						if prob < 0 || prob > 1
							error("Probability outside of bounds: $prob")
						end
						if prob != 0
							@deb("New edge: $(action_index), $(obs_index) -> $nz_id, $prob")
							new_edge_dict[nz] = prob
							ca+=prob
						end
					end
					if length(new_edge_dict) != 0
						new_obs[observations[obs_index]] = new_edge_dict
						#update incoming edge vector for other node
						#set should handle duplicates
						for (next, prob) in new_edge_dict
							next.incomingEdgeDicts[node] = new_edge_dict
						end
					end
				end
				if length(keys(new_obs)) != 0
					#re-normalize c(a,n,z)
					for (obs, dict) in new_obs
						for (next, prob) in dict
							#FIXME *n_obs is a quick fix to have sum of prob for each obs = 1
							dict[next] = prob/ca
							@deb("renormalized: $dict[next]), ca = $ca")
						end
					end
					new_edges[actions[action_index]] = new_obs
					new_actions[actions[action_index]] = ca
				end
			end
			node.edges = new_edges
			node.actionProb = new_actions
		end
	end
	return changed
end

include("bpigraph.jl")
