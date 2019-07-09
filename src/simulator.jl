

#include("bpigraph.jl")

mutable struct IBPIAgent
    controller::AbstractController
    current_node::Node
    value::Float64
    stats::agent_stats
end
function IBPIAgent(controller::AbstractController, initial_belief::Array{Float64})
    best_node = nothing
    best_value = nothing
    for (id, node) in controller.nodes
        new_value = sum(initial_belief[i]*node.value[i] for i in 1:length(initial_belief))
        if best_node == nothing || new_value > best_value
            best_node = node
            best_value = new_value
        end
    end
    return IBPIAgent(controller, best_node, 0.0, agent_stats())
end
function best_action(agent::IBPIAgent)
    return chooseWithProbability(agent.current_node.actionProb)
end
function update_agent!(agent::IBPIAgent, action::A, observation::W) where {A, W}
    agent.current_node = chooseWithProbability(agent.current_node.edges[action][observation])
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

function IBPIsimulate(policy::IBPIPolicy, maxsteps::Int64) where {S, A, W}
    maxlevel = length(policy.controllers)
    controller_i = policy.controllers[maxlevel][1]
    controller_j = policy.controllers[maxlevel-1][1]
    frame_i = controller_i.frame
    anynode = first(controller_i.nodes)[2]
    initial = ones(size(anynode.value))
    initial = initial ./ length(initial)
    agent_i = IBPIAgent(controller_i, initial)

    frame_j = controller_j.frame
    anynode_j = first(controller_j.nodes)[2]
    initial_j = ones(size(anynode_j.value))
    initial_j = initial_j ./ length(initial_j)
    if maxlevel - 1 == 0
        agent_j = IBPIAgent(controller_j, initial_j)

    else
        agent_j = IBPIAgent(controller_j, initial_j)
    end
    state = randn() > 0.5 ? :TL : :TR
    value = 0.0
    for i in 1:95
        print(" ")
    end
    println("end v")
    for i in 1:maxsteps
        if i % (maxsteps/100) == 0
            print("|")
        end
        ai = best_action(agent_i)
        aj = best_action(agent_j)
        @deb("state: $state -> ai: $ai, aj: $aj", :sim)

        value +=  IPOMDPs.reward(frame_i, state, ai, aj)
        @deb("value this step: $(IPOMDPs.reward(frame_i, state, ai, aj))", :sim)

        s_prime = compute_s_prime(state, ai, aj, frame_i)

        zi = compute_observation(s_prime, ai, aj, frame_i)
        zj = compute_observation(s_prime, aj, ai, frame_j)
        @deb("zi -> $zi, zj -> $zj", :sim)
        update_agent!(agent_i, ai, zi)
        update_agent!(agent_j, aj, zj)
        computestats!(agent_i.stats, ai, aj, state, s_prime, zi, zj)
        computestats!(agent_j.stats, aj, ai, state, s_prime, zj, zi)

        state = s_prime
    end
    println()
    return value/maxsteps, agent_i.stats, agent_j.stats
end
