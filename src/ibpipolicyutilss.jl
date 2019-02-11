#=
IBPIPolicyUtils:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
"""
Structure used for policy nodes.
ActionDist specifies the probability of executing the action at the corresponding index in actions
Edges stores the possible edges given action and observations obtained after executing the action.
Right now the (any) operator for observation is not implemented, we just have entries for all possible values
"""
mutable struct Node{A, W}
	actions::Vector{A}
	actionDist::Array{Float64}
	edges::Dict{A, Dict{W, Array{Edge}}}
	#constructor to get a node that points to himself for all actions and obs
	Node{actions::Vector{A}, observations::Vector{W}} = (
									n = new();
									n.actions = actions
									n.actionDist = ones(Float64, size(actions))
									randact = actions[rand(1:end)]

								)
end

"""
Each edge structure contains the node to which it brings to, and the probability of taking that edge.
"""
struct Edge
	next::Node
	probability:Float64
end


struct IBPIPolicy
	nodes::Array{Node}
end