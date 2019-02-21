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
	Actions only contains actions with probability >
	Right now the (any) operator for observation is not implemented, we just have entries for all possible values
	receives as parameters all possible actions and all possible observations
	Each node has an unique identifier, the ids of deleted nodes are reused and continuous (1:n_nodes)
	"""
	mutable struct Node{A, W, E <: AbstractEdge}
		id::Int64
		actionProb::Dict{A, Float64}
		edges::Dict{A, Dict{W, Vector{E}}}
		value::Vector{Float64}
		incomingEdges::Vector{E}
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
		return Node(id::Int64, actionProb::Dict{A, Float64}, Dict{A, Dict{W, Vector{Edge}}}(), Vector{Float64}(), Vector{Edge}())
	end

	function printNode(node::Node)
		for (a, prob) in node.actionProb
			obs = node.edges[a]
			for (obs, edges) in obs
				for edge in edges
					println("a=$a, $prob, $obs -> $(edge.next.id), $(edge.prob)")
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
	function InitialNode(actions::Vector{A}, observations::Vector{W}, value_len::Int64) where {A, W}
			randindex = rand(1:length(actions))
			n = Node(1, [actions[randindex]], observations)
			obsdict = Dict{W, Vector{Edge}}()
			for obs in observations
				edge = Edge(n, 1.0)
				push!(n.incomingEdges, edge)
				obsdict[obs] = [edge]
			end
			n.edges[actions[randindex]] = obsdict
			#FIXME what do i initialize this at
			n.value = ones(Float64, value_len)
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
	function Controller(actions, observations, value_len)
		newNode = InitialNode(actions, observations, value_len)
		Controller( Dict(1 => newNode))
	end
	#="""
	Perform a full backup operation according to Pourpart and Boutilier's paper on Bounded finite state controllers
	TODO: make this thing actually do a backup
	"""
	function full_backup!(controller::Controller, pomdpmodel::pomdpModel)
		max_value_n_index = 1
		max_value = 0
		for ni in 1:length(controller.nodes)
			node = controller.nodes[ni]
			#Value given node and belief state
			vnb = 0
			for s in 1:length(node.value)
				vnb+= node.value[s]*pomdpmodel.history.b[s]
			end
			if vnb > max_value
				@deb("Max value node: $max_value_n_index with value $vnb")
				max_value = vnb
				max_value_n_index = ni
			end
		end
		@deb("Max value node: $max_value_n_index")
	end
=#
	function evaluate!(controller::Controller{A,W}, pomdpmodel::pomdpModel) where {A, W}
			#solve V(n,s) = R(s, a(n)) + gamma*sumz(P(s'|s,a(n))Pr(z|s',a(n))V(beta(n,z), s'))
			#R(s,a(n)) is the reward function
			pomdp = pomdpmodel.frame
			nodes = controller.nodes
			n_nodes = length(keys(controller.nodes))
			states = POMDPs.states(pomdp)
			n_states = POMDPs.n_states(pomdp)
			A = zeros(n_states*n_nodes, n_states*n_nodes)
			b = zeros(n_states*n_nodes)

			#compute coefficients for sum(a)[R(s|a)*P(a|n)+gamma*sum(z, n')[P(s'|s,a)*P(z|s',a)*P(a|n)*P(n'|z)*V(nz, s')]]
			for (n_id, node) in nodes
				#A is the coefficient matrix (form x1 = a2x2+...+anxn+b)
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
									A[composite_index([n_id, s_index],[n_nodes, n_states]), composite_index([nz_index, s_prime_index],[n_nodes,n_states])]+= POMDPs.discount(pomdp)*p_s_prime*p_z*p_a_n*c_a_nz
								end
							end
						end
					end
				end
			end
			@deb("A = $A")
			@deb("b = $b")
			#maybe put this in coefficient computation instead of doing matrix operations for faster comp?
			I = Diagonal(ones(Float64,size(A,1), size(A,2) ))
			res = (I- A) \ b
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
						if prob < 0 || prob > 1
							error("Probability outside of bounds")
						end
						if prob != 0
							@deb("New edge: $(action_index), $(obs_index) -> $nz_id, $prob")
							push!(new_edge_vec, Edge(nz, prob))
							ca+=prob
						end
					end
					if length(new_edge_vec) != 0
						new_obs[observations[obs_index]] = new_edge_vec
					end
				end
				if length(keys(new_obs)) != 0
					#re-normalize c(a,n,z)
					for (obs, vec) in new_obs
						for i in 1:length(vec)
							vec[i] = Edge(vec[i].next, vec[i].prob/ca)
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
end
