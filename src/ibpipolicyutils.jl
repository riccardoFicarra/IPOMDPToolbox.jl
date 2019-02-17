#=
IBPIPolicyUtils:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
	"""
	Snippet to have debug utility. Use @deb(String) to print debug info
	Modulename.debug[] = true to enable, or just debug[] = true if you are in the module
	"""
	global debug = [false]
	macro deb(str)
	    :( debug[] && println($(esc(str))) )
	end

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
	Each node has an unique identifier, the ids of deleted nodes are not reused: possible cause for overflow?
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
			n.edges[n.actions[randindex]] = obsdict
			#FIXME what do i initialize this at
			n.value = ones(Float64, value_len)
			return n
	end
	"""
		Randomly choose an action based on action probability given a node
		returns action::A
	"""
	function getAction(node::Node{A, W, Edge}) where {A, W}
		action = chooseWithProbability(node.actionsProb)
		@deb("Chosen action $action")
		return action
	end
	"""
		Get a vector of actions with probability != 0
		TODO: transform the two arrays in a dict, only keep possible actions
	"""
	function getPossibleActions(node::Node{A, W, Edge}) where {A, W}
		return keys(node.actions)
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
		edgeProbability = map(edge -> edge.probability, edges)
		next = chooseWithProbability(edges, edgeProbability).next
		@deb("Chosen $(next.id) as next node")
		return next

	end
	"""
	Given an item vector and a probability vector (item probability at the same index as the item)
	Pick a random item based on the probability.
	probability must sum to 1. Items and probability must have the same length
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
	#no need for an ID counter, just use length(nodes)
	struct Controller{A, W}
		nodes::Vector{Node{A, W, Edge}}
	end
	"""
	Initialize a controller with the initial node, start id counter from 2
	"""
	Controller(actions, observations, value_len) = Controller([InitialNode(actions, observations, value_len)])
	"""
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

	function evaluation(controller::Controller, pomdpmodel::pomdpModel)
			#solve V(n,s) = R(s, a(n)) + gamma*sumz(P(s'|s,a(n))Pr(z|s',a(n))V(beta(n,z), s'))
			#R(s,a(n)) is the reward function
			pomdp = pomdpmodel.frame
			nodes = controller.nodes
			nodes_len = length(controller.nodes)
			states = POMDPs.states(pomdp)
			n_states = POMDPs.n_states(pomdp)
			v = Matrix{Float64}(undef, 0, nodes_len)
			#this system has to be solved for each node, each is size n_states
			for i in 1:nodes_len
				#A is the coefficient matrix
				#b is the constant term vector
				node = nodes[i]
				A = zeros(n_states, n_states)
				b = zeros(1,n_states)
				actions = getPossibleActions(nodes[i])
				for s in 1:n_states
					for a in actions
						b[s] += POMDPs.reward(pomdp, s, a)*node.actionProb[a]
						s_primes = POMDPs.transition(pomdp,s,a).vals
						possible_obs = keys(node.edges[a])  #only consider observations possible from current node/action combo
						for obs in possible_obs
							for s_prime in s_primes
								A[s, s_prime]+= POMDPs.discount(pomdp)*POMDPModelTools.pdf(POMDPs.transition(pomdp,s,a), s_prime)*POMDPModelTools.pdf(POMDPs.observation(pomdp, s_prime, a), obs) * nodes.edges[a][obs].prob*node.actionProb[a] #CHECK THAT THIS IS THE RIGHT VALUE (page 5 of BPI paper)
							end
						end
					end
				end
				v = cat(dims = 2,v, a \ transpose(b))
			end
			for i in 1:size(v, 2)
				nodes[i].value = copy(v[:, i])
			end
	end
