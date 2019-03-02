using Plots
using GraphRecipes
using SparseArrays


function adjacency_matrix(controller::Controller)
    return 0
end
"""
Function to draw value vectors. Only works with problems with 2 states.
"""
function valueVectors2D(controller::Controller)
    pyplot()
    n_nodes = length(controller.nodes)
    alphas = Matrix{Float64}(undef,2,n_nodes)
    for (n_id, node) in controller.nodes
        alphas[:, n_id] = node.value
    end
    #@deb("$alphas")
    plot(0:1, alphas, label=reshape(1:n_nodes, 1, n_nodes))
end
