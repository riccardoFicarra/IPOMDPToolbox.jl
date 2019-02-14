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
receives as parameters all possible actions and all possible observations
"""
abstract type AbstractEdge
#used to implement reciprocally nested structs until it gets fixed
end
struct Node{A, W, E <: AbstractEdge}
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
    return actions::Vector{A}, zeros(Float64, length(actions)), Dict{A, Dict{W, Vector{Edge}}}(), Vector{Float64}()
end

#Node(actions::Vector{A}, observations::Vector{W})= Node(actions::Vector{A}, zeros(Float64, length(actions)), Dict{A, Dict{W, Vector{Edge}}}(), Vector{Float64}())


function InitialNode(actions::Vector{A}, observations::Vector{W}) where {A, W}
		n = Node(actions, observations)
		randindex = rand(1:length(actions))
		n.actionDist[randindex] = 1.0
		obsdict = Dict{W, Vector{Edge}}()
		for obs in observations
			obsdict[obs] = [Edge(n, 1.0)]
		end
		n.edges[n.actions[randindex]] = obsdict
end

struct Controller{A, W}
	nodes::Array{Node{A, W, Edge}}
end

struct IBPIPolicy{A, W}
	controllers:Array{Controller{A, W}}
end