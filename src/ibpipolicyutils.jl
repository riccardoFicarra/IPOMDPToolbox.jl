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
"""
mutable struct Node{A, W, E <: AbstractEdge}
	actions::Vector{A}
	actionDist::Vector{Float64}
	edges::Dict{A, Dict{W, Vector{E}}}
	alpha::Vector{Float64}

end

"""
Each edge structure contains the node to which it brings to, and the probability of taking that edge.
"""
struct Edge{A, W} <: AbstractEdge
	next::Node{A, W, Edge}
	probability::Float64
end

function Node(actions::Vector{A}, observations::Vector{W}) where {A, W}
    return Node(actions::Vector{A}, zeros(Float64, length(actions)), Dict{A, Dict{W, Vector{Edge}}}(), Vector{Float64}())
end
"""
	Get a node with a random action chosen and with all observation edges
	pointing back to itself
"""
function InitialNode(actions::Vector{A}, observations::Vector{W}) where {A, W}
		n = Node(actions, observations)
		randindex = rand(1:length(actions))
		n.actionDist[randindex] = 1.0
		obsdict = Dict{W, Vector{Edge}}()
		for obs in observations
			obsdict[obs] = [Edge(n, 1.0)]
		end
		n.edges[n.actions[randindex]] = obsdict
		return n
end
"""
	Randomly choose an action based on action probability given a node
	returns action::A
"""
function getAction(node::Node{A, W, Edge}) where {A, W}
	return chooseWithProbability(node.actions, node.actionDist)
end
"""
	given node, action and observation returns the next node
"""
function getNextNode(node::Node{A, W, Edge}, action::A, observation::W) where {A, W}
	edges = node.edges[action][observation]
	edgeProbability = edges.map(edge -> edge.probability)
	return chooseWithProbability(edges, edgeProbability).next
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

struct Controller{A, W}
	nodes::Vector{Node{A, W, Edge}}
end

Controller(actions, observations) = Controller([InitialNode(actions, observations)])

struct IBPIPolicy{A, W}
	#temporary, find a way to store multiple controllers for frames and other agents
	controllers::Vector{Controller{A, W}}
end
