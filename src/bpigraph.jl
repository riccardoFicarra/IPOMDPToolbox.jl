#=
using Plots
#using GraphRecipes
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
    col_counter = 1
    for (n_id, node) in controller.nodes
        alphas[:, col_counter] = node.value
        col_counter+=1
    end
    #@deb("$alphas")
    plot(0:1, alphas, label=reshape(collect(keys(controller.nodes)), 1, n_nodes))
end

using LightGraphs, GraphPlot
function draw_controller(controller::Controller{A, W}) where {A, W}
    nodes = controller.nodes
    g = SimpleDiGraph(length(nodes))
    edgelabel = Vector{String}()
    temp_id = Dict{Int64, Int64}()
    for i in keys(controller.nodes)
        temp_id[i] = length(temp_id)+1
    end
    @deb("$temp_id")
    for (id, node) in nodes
        for (action, action_dict) in node.edges
            for (obs, obs_dict) in action_dict
                for (next, prob) in obs_dict
                     ok = LightGraphs.add_edge!(g, temp_id[id], temp_id[next.id])
                     #set_props!(mg, Edge(id, next.id), Dict(:action => action, :obs => obs, :prob => prob))
                     if !ok
                         @deb("failed $id -> $(next.id) $action $obs")
                     else
                         @deb("$id -> $(next.id)")
                         push!(edgelabel, "$action $obs $prob")
                     end
                end
            end
        end
    end

    nodelabel = collect(keys(nodes))
    #=
    nodelabel = zeros(Int64, controller.maxId)
    for i in 1:controller.maxId
        if haskey(nodes, i)
            nodelabel[i] = i
        end
    end
    =#
    gplot(g, nodelabel = nodelabel, edgelabel = edgelabel)
end

=#

function print_graph_latex(controller::AbstractController, name::String)
    open("../pictures/controllers/$string.tex", "w") do file
    # do stuff with the open file
end


    close(f)
end
