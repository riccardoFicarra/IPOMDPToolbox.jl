#=
IBPIPolicyUtils:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
	using LinearAlgebra
	using JuMP
	using GLPK

	abstract type AbstractEdge
	#used to implement reciprocally nested structs until it gets fixed
	end
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
	mutable struct Node{A, W, E <: AbstractEdge}
		id::Int64
		actionProb::Dict{A, Float64}
		edges::Dict{A, Dict{W, Vector{E}}}
		value::Vector{Float64}
		#needed to efficiently redirect edges during pruning
		incomingEdgeVector::Set{Vector{E}}
	end

	"""
	Each edge structure contains the node to which it brings to, and the probability of taking that edge.
	"""
	struct Edge{A, W} <: AbstractEdge
		next::Node{A, W, Edge}
		prob::Float64
	end

	function Node(id::Int64,actions::Vector{A}, observations::Vector{W}) where {A, W}
		actionProb = Dict{A, Float64}()
		for i in 1:length(actions)
			actionProb[actions[i]] = 1/length(actions)
		end
		return Node(id::Int64, actionProb::Dict{A, Float64}, Dict{A, Dict{W, Vector{Edge}}}(), Vector{Float64}(), Set{Vector{Edge}}())
	end

	function Base.println(node::Node)
		for (a, prob) in node.actionProb
			obs = node.edges[a]
			for (obs, edges) in obs
				for edge in edges
					println("node_id=$(node.id), a=$a, $prob, $obs -> $(edge.next.id), $(edge.prob)")
				end
			end
		end
		println("Value vector = $(node.value)")
	end
	"""
		Receives vectors of all possible actions and observations, plus number of states
		Get a node with a random action chosen and with all observation edges
		pointing back to itself
	"""
	function InitialNode(pomdp::POMDP{A, W}) where {A, W}
			actions = POMDPs.actions(pomdp)
			observations = POMDPs.observations(pomdp)
			states = POMDPs.states(pomdp)
			n_states = POMDPs.n_states(pomdp)
			randindex = rand(1:length(actions))
			n = Node(1, [actions[randindex]], observations)
			obsdict = Dict{W, Vector{Edge}}()
			for obs in observations
				edge = Edge(n, 1.0)
				obsdict[obs] = [edge]
				push!(n.incomingEdgeVector, [edge])
			end
			n.edges[actions[randindex]] = obsdict
			n.value = [POMDPs.reward(pomdp, s, actions[randindex]) for s in states]
			return n
	end
	"""
		Randomly choose an action based on action probability given a node
		returns action::A
	"""
	function getAction(node::Node{A, W, Edge}) where {A, W}
		action = chooseWithProbability(node.actionProb)
		@deb("Chosen action $action")
		return action
	end
	"""
		Get a vector of actions with probability != 0
		TODO: transform the two arrays in a dict, only keep possible actions
	"""
	function getPossibleActions(node::Node{A, W, Edge}) where {A, W}
		return keys(node.actionProb)
	end
	"""
		given node, action and observation returns the next node
		maps the whole array of edges to get edge prob (O(n)), then calls chooseWithProbability O(n)
	"""
	function getNextNode(node::Node{A, W, Edge}, action::A, observation::W) where {A, W}
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
	"""
	Given an array of edges
	Pick a random edge based on the prob.
	prob must sum to 1.
	O(n)
	"""
	function chooseWithProbability(edges::Vector{Edge})
		randn = rand() #number in [0, 1)
		@deb(randn)
		for edge in edges
			if randn <= edge.prob
				return edge
			else
				randn-= edge.prob
			end
		end
		error("Out of dict bounds while choosing items")
	end
	#no need for an ID counter, just use length(nodes)
	#Todo add a hashmap of id -> node index to have O(1) on access from id
	struct Controller{A, W}
		nodes::Dict{Int64, Node{A, W, Edge}}
	end
	"""
	Initialize a controller with the initial node, start id counter from 2
	"""
	function Controller(pomdp::POMDP{A,W}) where {A, W}
		newNode = InitialNode(pomdp)
		Controller{A, W}(Dict(1 => newNode))
	end

	function build_node(node_id::Int64, actions::Vector{A}, actionProb::Vector{Float64}, observations::Vector{Vector{W}}, observation_prob::Vector{Vector{Float64}}, next_nodes::Vector{Vector{Node{A,W,Edge}}}, value::Vector{Float64}) where {A, W}
		if length(actions) != length(observations) || length(actions) != length(actionProb) || length(actions) != length(observation_prob) || length(actions) != length(next_nodes)
			error("Length of action-level arrays are different")
		end
		edges = Dict{A, Dict{W, Vector{Edge}}}()
		d_actionprob = Dict{A, Float64}()
		for a_i in 1:length(actions)
			action = actions[a_i]
			d_actionprob[action] = actionProb[a_i]
			a_obs = observations[a_i]
			a_obs_prob = observation_prob[a_i]
			a_next_nodes = next_nodes[a_i]
			if length(a_obs) != length(a_obs_prob) || length(a_obs) != length(a_next_nodes)
				error("Length of observation-level arrays are different")
			end
			new_obs = Dict{W, Vector{Edge}}()
			for obs_index in 1:length(a_obs)
				obs = a_obs[obs_index]
				new_obs[obs] = [Edge(a_next_nodes[obs_index], a_obs_prob[obs_index])]
			end
			edges[action] = new_obs
		end
		return Node(node_id, d_actionprob, edges, value, Set{Vector{Edge}}())
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
				for (n_id, node) in nodes
					new_v = node_value(node, a, obs, pomdp)
					#do not set node id for now
					new_node = build_node(new_nodes_counter, [a], [1.0], [[obs]], [[1.0]], [[node]], new_v)
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
			new_nodes_a = incprune(new_nodes_z)
			union!(new_nodes, new_nodes_a)
		end
		#all new nodes, final filtering
		new_nodes = filterNodes(new_nodes)
		#add new nodes to controller
		#all_nodes = union(new_nodes, Set{Node}(oldnode for oldnode in values(nodes)));
		all_nodes = filterNodes(union(new_nodes, Set{Node}(oldnode for oldnode in values(nodes))))
		nodes_counter = 1
		for node in all_nodes
			#set id and add nodes to controller
			#compact node ids so they are contiguous
			node.id = nodes_counter
			nodes[nodes_counter] = node
			nodes_counter+=1
		end
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
	    for (n_id, n) in new_nodes
			#remove the node we're testing from the node set (else we always get that a node dominates itself!)
			pop!(new_nodes, n_id)
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
	        if JuMP.value(e) >= 0
	            #rewiring function here!
			if debug[] == true
				for i in keys(new_nodes)
					print("c$(new_nodes[i].id) = $(JuMP.value(c[i])) ")
				end
				println("")
			end
	            for edgeV in n.incomingEdgeVector
	                #FIXME change data strucure to avoid linear search!
					#constant for normalization
					tot_prob = 0.0
	                for e_i in 1:length(edgeV)
	                    old_edge = edgeV[e_i]
						#search for edge
	                    if old_edge.next == n
							#update probabilities
	                        for i in keys(new_nodes)
	                            v = JuMP.value(c[i])
	                            if v != 0.0
									new_prob = old_edge.prob*v
									tot_prob+= new_prob
	                                push!(edgeV, IPOMDPToolbox.Edge(new_nodes[i], new_prob))
	                            end
	                        end
							#if tot_prob != 0.5
							@deb("tot_prob = $tot_prob")
							#end
							#for i in 1:length(new_nodes)
							#	edgeV[i].prob/= tot_prob
							#end
							#remove old edge from edge vector
	                        deleteat!(edgeV, e_i)
	                        break
	                    end
	                end
	                if length(edgeV) == 0
	                    #remove vector if empty
	                    pop!(n.incomingEdgeVector, edgeV)
	                end
	            end
	            #end of rewiring, remove dominated node
				if debug[] == true
					println("Deleting node $(n.id): obj value = $(JuMP.value(e))")
					println(n)
				end
	        else
				#if node is not dominated readd it to the dict!
				new_nodes[n_id] = n
			end
	    end
	    return Set{IPOMDPToolbox.Node}(node for node in values(new_nodes))
	end
	#=

	function filterNodes(nodeSet::Set{Node})
		return nodeSet
	end
=#
	"""
	Perform incremental pruning on a set of nodes by computing the cross sum and filtering every time
	Follows Cassandra et al paper
	"""
	function incprune(nodeVec::Vector{Set{Node}})
		@deb("Called incprune, length(nodevec) = $(length(nodeVec))")
		res = filterNodes(xsum(nodeVec[1], nodeVec[2]))
		#=if debug[] == true
			for node in res
				println(node)
			end
		end
		=#
		for i = 3:length(nodeVec)
			res = filterNodes(xsum(res, nodeVec[i]))
			@deb("Length $i = $(length(nodeVec[i]))")
		end
		return res
	end

	function node_value(node::Node{A, W, Edge}, action::A, observation::W, pomdp::POMDP) where {A, W}
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


	function xsum(A::Set{Node}, B::Set{Node})
		@deb("Called xsum")
		X = Set{Node}()
	    for a in A, b in B
			#each of the newly generated nodes only has one action!
			@assert length(a.actionProb) == length(b.actionProb) == 1 "more than one action in freshly generated node"
			a_action = collect(keys(a.actionProb))[1]
			b_action = collect(keys(b.actionProb))[1]
			@assert a_action == b_action "action mismatch"
			c = mergeNode(a, b, a_action)
			if IPOMDPToolbox.debug[] == true
				println("nodes merged:$(a.id) <- $(b.id)")
				println(c)
			end
			push!(X, c)
	    end
	    return X
	end

	function mergeNode(a::Node, b::Node, action::A) where {A}
		b_obs = b.edges[action]
		res = deepcopy(a)
		#There is no way we have repeated observation because set[obs]
		#only contains nodes with obs
		for (obs, edges) in b_obs
			if haskey(res.edges[action], obs)
				@deb("Obs already present")
			end
			res.edges[action][obs] = edges
		end
		res.value = res.value + b.value
		return res
	end

	function evaluate!(controller::Controller{A,W}, pomdpmodel::pomdpModel) where {A, W}
			#solve V(n,s) = R(s, a(n)) + gamma*sumz(P(s'|s,a(n))Pr(z|s',a(n))V(beta(n,z), s'))
			#R(s,a(n)) is the reward function
			pomdp = pomdpmodel.frame
			nodes = controller.nodes
			n_nodes = length(keys(controller.nodes))
			states = POMDPs.states(pomdp)
			n_states = POMDPs.n_states(pomdp)
			M = zeros(n_states*n_nodes, n_states*n_nodes)
			b = zeros(n_states*n_nodes)

			#compute coefficients for sum(a)[R(s|a)*P(a|n)+gamma*sum(z, n')[P(s'|s,a)*P(z|s',a)*P(a|n)*P(n'|z)*V(nz, s')]]
			for (n_id, node) in nodes
				#M is the coefficient matrix (form x1 = a2x2+...+anxn+b)
				#b is the constant term vector
				#variables are all pairs of n,s
				actions = getPossibleActions(node)
				for s_index in 1:n_states
					s = POMDPs.states(pomdp)[s_index]
					for a in actions
						b[composite_index([n_id, s_index],[n_nodes, n_states])] += POMDPs.reward(pomdp, s, a)*node.actionProb[a]
						s_primes = POMDPs.transition(pomdp,s,a).vals
						possible_obs = keys(node.edges[a])  #only consider observations possible from current node/action combo
						for obs in possible_obs
							for s_prime_index in 1:length(s_primes)
								s_prime = s_primes[s_prime_index]
								p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp,s,a), s_prime)
								p_a_n = node.actionProb[a]
								p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, s_prime, a), obs)
								for edge in node.edges[a][obs]
									if !haskey(controller.nodes,edge.next.id)
										error("Node not present in nodes")
									end
									nz_index = edge.next.id
									c_a_nz = edge.prob*node.actionProb[a] #CHECK THAT THIS IS THE RIGHT VALUE (page 5 of BPI paper)
									M[composite_index([n_id, s_index],[n_nodes, n_states]), composite_index([nz_index, s_prime_index],[n_nodes,n_states])]+= POMDPs.discount(pomdp)*p_s_prime*p_z*p_a_n*c_a_nz
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
				node.value = copy(res[(n_id-1)*n_states+1 : n_id*n_states])
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
		@variable(lpmodel, c[a=1:n_actions, z=1:n_observations, n=1:n_nodes] >= 0)
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
		@expression(lpmodel, sumc, sum(sum(sum(c[a,z,n] for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions))
		@constraint(lpmodel, con_sum,  sumc == 1)
		#@constraint(lpmodel, canz_prob[a=1:n_actions, z=1:n_observations, n=1:n_nodes], 0 <= c[a,z,n] <= 1)
		#print(lpmodel)
		optimize!(lpmodel)


		if JuMP.value(e) >= 0
			changed = true
			#@deb("Good so far")
			new_edges = Dict{A, Dict{W, Vector{Edge}}}()
			new_actions = Dict{A, Float64}()
			#@deb("New structures created")
			for action_index in 1:n_actions
				ca = 0
				new_obs = Dict{W, Vector{Edge}}()
				for obs_index in 1:n_observations
					new_edge_vec = Vector{Edge}()
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
							push!(new_edge_vec, Edge(nz, prob))
							ca+=prob
						end
					end
					if length(new_edge_vec) != 0
						new_obs[observations[obs_index]] = new_edge_vec
						#update incoming edge vector for other node
						#set should handle duplicates
						for edge in new_edge_vec
							push!(edge.next.incomingEdgeVector, new_edge_vec)
						end
					end
				end
				if length(keys(new_obs)) != 0
					#re-normalize c(a,n,z)
					for (obs, vec) in new_obs
						for i in 1:length(vec)
							#FIXME *n_obs is a quick fix to have sum of prob for each obs = 1
							vec[i] = Edge(vec[i].next, vec[i].prob/ca)
							@deb("renormalized: $(vec[i].prob), ca = $ca")
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
