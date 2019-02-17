# ] add POMDPs IPOMDPs BeliefUpdaters
using POMDPs
using POMDPModelTools
using IPOMDPs
using BeliefUpdaters

struct normalTiger <: POMDP{Symbol, Symbol, Symbol}
end



POMDPs.states(pomdp::normalTiger) = [:TR, :TL]
POMDPs.n_states(pomdp::normalTiger) = 2
function POMDPs.stateindex(pomdp::normalTiger, s::Symbol)
    if s == :TR
        return 1
    end
    if s == :TL
        return 2
    end
    error("no index for state $s")
end

POMDPs.observations(pomdp::normalTiger) = [:GL, :GR]
POMDPs.n_observations(pomdp::normalTiger) = 2
function POMDPs.obsindex(pomdp::normalTiger, o::Symbol)
    if o == :GL
        return 1
    elseif o == :GR
        return 2
    else
        error("No index for observation $o")
    end
    return 0
end

POMDPs.actions(pomdp::normalTiger) = [:OL, :OR, :L]
POMDPs.n_actions(pomdp::normalTiger) = 3
function POMDPs.actionindex(pomdp::normalTiger, a::Symbol)
    if a == :OL
        return 1
    elseif a == :OR
        return 2
    elseif a == :L
        return 3
    else
        error("No inxed for action $a")
    end
    return 0
end

function POMDPs.transition(pomdp::normalTiger, s::Symbol, a::Symbol)
    if a == :L
        if s == :TR
            return SparseCat([:TR, :TL],[1.0, 0])
        else
            return SparseCat([:TR, :TL],[0, 1.0])
        end
    elseif (a == :OL) || (a == :OR)
        return SparseCat([:TR, :TL], [0.5, 0.5])
    else
        error("No transition defined for taking action $a in state $s")
    end
end

function POMDPs.observation(pomdp::normalTiger, a::Symbol, s::Symbol)
    observations = [:GL, :GR]
    probs = []
    if a == :L
        if s == :TL
            probs = [0.85, 0.15]
        else
            probs = [0.15, 0.85]
        end
    else
        probs = [0.5, 0.5]
    end
    return SparseCat(observations, probs)

end
function POMDPs.reward(pomdp::normalTiger, s::Symbol, a::Symbol)
    if a == :L
        return -1.0
    elseif a == :OR
        if s == :TR
            return -100.0
        elseif s == :TL
            return 10.0
        end
    elseif a == :OL
        if s == :TL
            return -100.0
        elseif s == :TR
            return 10.0
        end
    else
        error("No reward defined for action $a and state $s")
    end
end

POMDPs.initialstate_distribution(pomdp::normalTiger) = SparseCat([:TR, :TL],[0.5, 0.5])
POMDPs.discount(pomdp::normalTiger) = 0.99




include("ibpisolver.jl")
Main.debug[]=true
#=actions = [:a1, :a2, :a3]
observations = [:o1, :o2, :o3]

node = IBPIPolicyUtils.InitialNode(actions, observations)
next = IBPIPolicyUtils.getNextNode(node, IBPIPolicyUtils.getAction(node), :o1)
=#
pomdp = normalTiger()
p = BPIPolicy(pomdp)

struct pomdpModel{S,A,W,P} <: IPOMDPs.Model{A,W,P}
    history::DiscreteBelief

    # Immutable part of the structure! This is commo to all the models of the same frame!
    frame::POMDP{S,A,W}

    # Data
    updater::DiscreteUpdater
    policy::P
    depth::Int64
end

pm = IPOMDPs.Model(pomdp, depth=0, solvertype=:IBPI)
full_backup!(p.controller, pm)
