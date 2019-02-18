#=
IBPIPolicyUtils:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
	using LinearAlgebra

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
		probability::Float64
	end

	function Node(id::Int64,actions::Vector{A}, observations::Vector{W}) where {A, W}
		actionProb = Dict{A, Float64}()
		for i in 1:length(actions)
			actionProb[actions[i]] = 1/length(actions)
		end
		return Node(id::Int64, actionProb::Dict{A, Float64}, Dict{A, Dict{W, Vector{Edge}}}(), Vector{Float64}(), Vector{Edge}())
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
	Pick a random edge based on the probability.
	probability must sum to 1.
	O(n)
	"""
	function chooseWithProbability(edges::Vector{Edge})
		randn = rand() #number in [0, 1)
		@deb(randn)
		for edge in edges
			if randn <= edge.probability
				return edge
			else
				randn-= edge.probability
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
	function evaluate!(controller::Controller, pomdpmodel::pomdpModel)
			#solve V(n,s) = R(s, a(n)) + gamma*sumz(P(s'|s,a(n))Pr(z|s',a(n))V(beta(n,z), s'))
			#R(s,a(n)) is the reward function
			pomdp = pomdpmodel.frame
			nodes = controller.nodes
			n_nodes = length(keys(controller.nodes))
			states = POMDPs.states(pomdp)
			n_states = POMDPs.n_states(pomdp)
			#this system has to be solved for each node, each is size n_states*n_nodes
			A = zeros(n_states*n_nodes, n_states*n_nodes)
			b = zeros(n_states*n_nodes)

			#compute coefficients for sum(a)[R(s|a)*P(a|n)+gamma*sum(z, n')[P(s'|s,a)*P(z|s',a)*P(a|n)*P(n'|z)*V(nz, s')]]
			for (n_id, node) in nodes
				#A is the coefficient matrix
				#b is the constant term vector
				actions = getPossibleActions(node)
				for s_index in 1:n_states
					s = POMDPs.states(pomdp)[s_index]
					for a in actions
						b[composite_index(n_id,n_states, s_index)] += POMDPs.reward(pomdp, s, a)*node.actionProb[a]
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
									c_a_nz = edge.probability*node.actionProb[a] #CHECK THAT THIS IS THE RIGHT VALUE (page 5 of BPI paper)
									A[composite_index(n_id,n_states, s_index), composite_index(nz_index,n_states, s_prime_index)]+= POMDPs.discount(pomdp)*p_s_prime*p_z*p_a_n*c_a_nz
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
			for (node_id, node) in nodes
				node.value = copy(res[(n_id-1)*n_states+1 : n_id*n_states])
				@deb("Value of node $n_id[1] = $(nodes[n_id].value[1])")
			end
	end
function composite_index(primary::Int64, secondary_len::Int64, secondary::Int64)
	return (primary-1)*secondary_len+secondary
end
