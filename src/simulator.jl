

#include("bpigraph.jl")

mutable struct IBPIAgent
    controller::AbstractController
    current_node::Node
    value::Float64
    stats::agent_stats
    visited::Array{Int64}
end
function IBPIAgent(controller::AbstractController, initial_belief::Array{Float64})
    best_node = nothing
    best_value = nothing
    for node in controller.nodes
        new_value = sum(initial_belief[i]*node.value[i] for i in 1:length(initial_belief))
        if best_node == nothing || new_value > best_value
            best_node = node
            best_value = new_value
        end
    end
    return IBPIAgent(controller, best_node, 0.0, agent_stats(), zeros(Int64, length(controller.nodes)))
end
function best_action(agent::IBPIAgent)
    return chooseWithProbability(agent.current_node.actionProb)
end
function update_agent!(agent::IBPIAgent, action::A, observation::W) where {A, W}
    agent.current_node = chooseWithProbability(agent.current_node.edges[action][observation])
    agent.visited[agent.current_node.id]+=1
end
function compute_s_prime(state::S, ai::A, aj::A, frame::IPOMDP) where {S, A}
    dist = IPOMDPs.transition(frame, state, ai, aj)
    items = dist.probs
    randn = rand() #number in [0, 1)
    for i in 1:length(dist.vals)
        if randn <= items[i]
            return dist.vals[i]
        else
            randn-= items[i]
        end
    end
    error("Out of dict bounds while choosing items")
end

function compute_s_prime(state::S, ai::A, aj::A, frame::POMDP) where {S, A}
    dist = POMDPs.transition(frame, state, ai)
    items = dist.probs
    randn = rand() #number in [0, 1)
    for i in 1:length(dist.vals)
        if randn <= items[i]
            return dist.vals[i]
        else
            randn-= items[i]
        end
    end
    error("Out of dict bounds while choosing items")
end

function compute_observation(s_prime::S, ai::A, aj::A, frame::IPOMDP) where {S, A}
    dist = IPOMDPs.observation(frame, s_prime, ai, aj)
    items = dist.probs
    randn = rand() #number in [0, 1)
    for i in 1:length(dist.vals)
        if randn <= items[i]
            return dist.vals[i]
        else
            randn-= items[i]
        end
    end
    error("Out of dict bounds while choosing items")
end

function compute_observation(s_prime::S, ai::A, aj::A, frame::POMDP) where {S, A}
    dist = POMDPs.observation(frame, s_prime, ai)
    items = dist.probs
    randn = rand() #number in [0, 1)
    for i in 1:length(dist.vals)
        if randn <= items[i]
            return dist.vals[i]
        else
            randn-= items[i]
        end
    end
    error("Out of dict bounds while choosing items")
end

function IBPIsimulate(controller_i::InteractiveController{S, A, W}, controller_j::AbstractController, maxsteps::Int64; trace=false) where {S, A, W}
    frame_i = controller_i.frame
    anynode = controller_i.nodes[1]
    initial = ones(length(anynode.value))
    initial = initial ./ length(initial)
    agent_i = IBPIAgent(controller_i, initial)

    frame_j = controller_j.frame
    anynode_j = controller_j.nodes[1]
    initial_j = ones(length(anynode_j.value))
    initial_j = initial_j ./ length(initial_j)
    agent_j = IBPIAgent(controller_j, initial_j)
    state = randn() > 0.5 ? :TL : :TR
    value = 0.0
    if !trace
        for i in 1:95
            print(" ")
        end
        println("end v")
    end
    for i in 1:maxsteps
        if i % (maxsteps/100) == 0 && !trace
            print("|")
        end
        ai = best_action(agent_i)
        aj = best_action(agent_j)
        if trace
            println("state: $state -> ai: $ai, aj: $aj", :sim)
        end
        value =  IPOMDPs.discount(frame_i) * value + IPOMDPs.reward(frame_i, state, ai, aj)
        if trace
            println("value this step: $(IPOMDPs.reward(frame_i, state, ai, aj))", :sim)
        end
        s_prime = compute_s_prime(state, ai, aj, frame_i)

        zi = compute_observation(s_prime, ai, aj, frame_i)
        zj = compute_observation(s_prime, aj, ai, frame_j)
        if trace
            println("zi -> $zi, zj -> $zj")
        end
        update_agent!(agent_i, ai, zi)
        @deb("new current node for I:", :sim)
        @deb(agent_i.current_node, :sim)
        update_agent!(agent_j, aj, zj)
        computestats!(agent_i.stats, ai, aj, state, s_prime, zi, zj)
        computestats!(agent_j.stats, aj, ai, state, s_prime, zj, zi)

        state = s_prime
    end
    println()
    return value, agent_i, agent_j
end
