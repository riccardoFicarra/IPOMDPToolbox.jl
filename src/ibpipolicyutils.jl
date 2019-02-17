#=
IBPIPolicyUtils:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
#module IBPIPolicyUtils
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
		actions::Vector{A}
		actionDist::Vector{Float64}
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
	    return Node(id::Int64, actions::Vector{A}, zeros(Float64, length(actions)), Dict{A, Dict{W, Vector{Edge}}}(), Vector{Float64}(), Vector{Edge}())
	end
	"""
		Get a node with a random action chosen and with all observation edges
		pointing back to itself
	"""
	function InitialNode(actions::Vector{A}, observations::Vector{W}, value_len::Int64) where {A, W}
			n = Node(1, actions, observations)
			randindex = rand(1:length(actions))
			n.actionDist[randindex] = 1.0
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
		action = chooseWithProbability(node.actions, node.actionDist)
		@deb("Chosen action $action")
		return action
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
	function chooseWithProbability(items::Vector, probability::Vector{Float64})
		randn = rand() #number in [0, 1)
		@deb(randn)
		if length(items) != length(probability)
			error("Length of item vector is different from length of probability vector")
		end
		for i in 1:length(items)
			if randn <= probability[i]
				return items[i]
			else
				randn-= probability[i]
			end
		end
		error("Out of bounds in item array while choosing items")
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

#end
