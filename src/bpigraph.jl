using Plots
using GraphRecipes
using SparseArrays


function adjacency_matrix(controller::IPOMDPToolbox.Controller)
    src = Vector{Int64}(undef, 0)
    dst = Vector{Int64}(undef, 0)
    weight = Vector{Float64}(undef, 0)
    for (src_id, src_node) in controller.nodes
        for a in keys(src_node.edges)
            for obs in keys(src_node.edges[a])
                for edge in src_node.edges[a][obs]
                    push!(src, src_id)
                    push!(dst, edge.next.id)
                    push!(weight, edge.prob)
                end
            end
        end
    end
    return sparse(src, dst, weight)
end
"""
Function to draw value vectors. Only works with problems with 2 states.
"""
function valueVectors2D(controller::Controller)
    @deb("valueVectors2D called")
    pyplot()
    n_nodes = length(controller.nodes)
    alphas = Matrix{Float64}(undef,2,n_nodes)
    for (n_id, node) in controller.nodes
        alphas[:, n_id] = node.value
    end
    #@deb("$alphas")
    plot(0:1, alphas, label=reshape(1:n_nodes, 1, n_nodes))
end
