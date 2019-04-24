function init_controllers(ipomdp::IPOMDP{S,A,W}, maxlevel::Int64, force::Int64) where {S, A, W}
    #hardcoded for now
    agents = [agent(ipomdp), agent(ipomdp)]
    controllers = Dict{Int64, Controller{A, W}}()
    for i in 0:maxlevel
        l = maxlevel-i
        #alternate between agent I and J starting with I
        agent = agents[i%2+1]
        controller = Controller(l, agent, force)
        controllers[l] = controller
    end
    return controllers
end

function Controller(level::Int64, agent::Agent{S, A, W}, force::Int64) where {S, A, W}
    newNode = InitialNode(actions_agent(agent), observations_agent(agent), force)
    Controller{A, W}(level, agent, Dict(1 => newNode), 1)
end
#no need to init value vectors here, they will be set by evaluate.
function InitialNode(actions::Vector{A}, observations::Vector{W}, force::Int64) where {S, A, W}
        if force == 0
            actionindex = rand(1:length(actions))
        else
            if force > length(actions)
                error("forced action outside of action vector length")
            end
            actionindex = force
        end
        n = Node(1, [actions[actionindex]], observations)
        obsdict = Dict{W, Dict{Node, Float64}}()
        n.incomingEdgeDicts[n] = Vector{Dict{Node, Float64}}(undef, 0)
        for obs in observations
            edges = Dict{Node, Float64}(n => 1.0)
            obsdict[obs] = edges
            push!(n.incomingEdgeDicts[n], edges)
        end
        n.edges[actions[actionindex]] = obsdict
        n.value = []
        return n
end

function evaluate!(controller::Controller{A,W},  controller_j::Controller{A, W}, ipomdp::IPOMDP{S, A, W},) where {S, A, W}

    nodes = controller.nodes
    n_nodes = length(controller.nodes)
    nodes_j = controller_j.nodes
    n_nodes_j = length(nodes_j)
    states = IPOMDPs.states(ipomdp)
    n_states = length(states)
    #M[s, nj, ni, s', nj', ni']
    M = zeros(n_states, n_nodes_j, n_nodes, n_states, n_nodes_j, n_nodes)
    b = zeros(n_states, n_nodes_j, n_nodes)

    #dictionary used for recompacting ids
    temp_id = Dict{Int64, Int64}()
    for (node_id, node) in nodes
        temp_id[node_id] = length(temp_id)+1
    end

    #dictionary used for recompacting ids -> they are sorted!
    #quick fix to have the values in some order
    temp_id_j = Dict{Int64, Int64}()
    for node_id in sort(collect(keys(nodes_j)))
        temp_id_j[node_id] = length(temp_id_j)+1
    end

    #compute coefficients for sum(a)[R(s|a)*P(a|n)+gamma*sum(z, n', s')[P(s'|s,a)*P(z|s',a)*P(a|n)*P(n'|z)*V(nz, s')]]
    for (ni_id, ni) in nodes
        #M is the coefficient matrix (form x1 = a2x2+...+anxn+b)
        #b is the constant term vector
        #variables are all pairs of n,s
        for s_index in 1:n_states
            s = states[s_index]
            for (nj_id, nj) in nodes
                M[s_index, temp_id_j[nj_id], temp_id[ni_id], s_index, temp_id_j[nj_id], temp_id[ni_id]] +=1
                for (ai, p_ai) in ni.actionProb
                    #@deb("ai = $ai")
                    @deb("ai = $ai")
                    for (aj, p_aj) in nj.actionProb
                        #@deb("aj = $aj")
                        @deb("aj = $aj")
                        action_dict = Dict{Agent, Any}(IPOMDPs.agent(ipomdp) => ai, IPOMDPs.agent(emulated_frames(ipomdp)[1]) => aj)
                        r = IPOMDPs.reward(ipomdp, IPOMDPs.IS(s, Vector{Model}(undef, 0)), action_dict)
                        #@deb("r = $s")
                        @deb("r = $r")
                        b[s_index, temp_id_j[nj_id], temp_id[ni_id]] = p_ai * p_aj * r
                        for (zi, obs_dict_i) in ni.edges[ai]
                            @deb("zi = $zi")
                            for s_prime_index in 1:n_states
                                s_prime = states[s_prime_index]
                                @deb("s_prime = $s_prime")
                                transition_i = POMDPModelTools.pdf(IPOMDPs.transition(ipomdp, s, action_dict), s_prime)
                                observation_i = POMDPModelTools.pdf(IPOMDPs.observation(ipomdp, s_prime, action_dict), zi)
                                @deb(transition_i)
                                @deb(observation_i)
                                for (zj, obs_dict_j) in nj.edges[aj]
                                    @deb("zj = $zj")
                                    observation_j = POMDPModelTools.pdf(IPOMDPs.observation(ipomdp, s_prime, action_dict), zj)
                                    for (n_prime_j, prob_j) in nj.edges[aj][zj]
                                        for (n_prime_i, prob_i) in ni.edges[ai][zi]
                                            M[s_index, temp_id_j[nj_id], temp_id[ni_id], s_prime_index, temp_id_j[n_prime_j.id], temp_id[n_prime_i.id]] -= p_ai * p_aj * IPOMDPs.discount(ipomdp) * transition_i * observation_i * observation_j * prob_j * prob_i
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    M_2d = reshape(M,n_states* n_nodes_j* n_nodes, n_states* n_nodes_j* n_nodes)
    b_1d = reshape(b, n_states* n_nodes_j* n_nodes)
    res_1d = M_2d \ b_1d
    res = reshape(res_1d, n_states, n_nodes_j, n_nodes)
    #copy respective value functions in nodes
    for (n_id, node) in nodes
        node.value = copy(res[:, :, temp_id[n_id]])
        #@deb("Value vector of node $n_id = $(nodes[n_id].value)")
    end
end
