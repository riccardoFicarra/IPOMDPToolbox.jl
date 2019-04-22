function init_controllers(ipomdp::IPOMDP{S,A,W}, maxlevel::Int64, force::Int64) where {S, A, W}
    #hardcoded for now
    agents = [agent(ipomdp), agent(ipomdp)]
    controllers = Dict{Int64, Controller{A, W}}()
    for l in 0:maxlevel
        #alternate between agent I and J
        controller = Controller(l, agents[l%2+1], force)
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

function evaluate!(controller::Controller{A,W}, ipomdp::IPOMDP{S, A, W}, l_controller::Controller(A, W)) where {S, A, W}
        #solve V(n,s) = R(s, a(n)) + gamma*sumz(P(s'|s,a(n))Pr(z|s',a(n))V(beta(n,z), s'))
        #R(s,a(n)) is the reward function
        nodes = controller.nodes
        n_nodes = length(controller.nodes)
        nodes_l = l_controller.nodes
        n_nodes_l = length(l_controller.nodes)
        states = IPOMDPs.states(ipomdp)
        n_states = length(states)
        M = spzeros(n_states*n_nodes*n_nodes_l, n_states*n_nodes*n_nodes_l)
        b = zeros(n_states*n_nodes*n_nodes_l)

        #dictionary used for recompacting ids
        temp_id = Dict{Int64, Int64}()
        for (node_id, node) in nodes
            temp_id[node_id] = length(temp_id)+1
        end

        #dictionary used for recompacting ids
        temp_id_l = Dict{Int64, Int64}()
        for (node_id, node) in nodes_l
            temp_id_l[node_id] = length(temp_id_l)+1
        end

        #compute coefficients for sum(a)[R(s|a)*P(a|n)+gamma*sum(z, n', s')[P(s'|s,a)*P(z|s',a)*P(a|n)*P(n'|z)*V(nz, s')]]
        for (n_id, node) in nodes
            #M is the coefficient matrix (form x1 = a2x2+...+anxn+b)
            #b is the constant term vector
            #variables are all pairs of n,s
            actions = getPossibleActions(node)
            for s_index in 1:n_states
                s = POMDPs.states(pomdp)[s_index]
                for a in actions
                    @deb("action = $a")
                    p_a_n = node.actionProb[a]
                    b[composite_index([temp_id[n_id], s_index],[n_nodes, n_states])] = POMDPs.reward(pomdp, s, a)*p_a_n
                    @deb("b($n_id, $s) = $(POMDPs.reward(pomdp, s, a)*p_a_n)")
                    M[composite_index([temp_id[n_id], s_index],[n_nodes, n_states]), composite_index([temp_id[n_id], s_index],[n_nodes, n_states])] = 1
                    @deb("M[$n_id, $s][$n_id, $s] = 1")
                    s_primes = POMDPs.transition(pomdp,s,a).vals
                    possible_obs = keys(node.edges[a])  #only consider observations possible from current node/action combo
                    for obs in possible_obs
                        @deb("obs = $obs")
                        for s_prime_index in 1:length(s_primes)
                            s_prime = s_primes[s_prime_index]
                            p_s_prime =POMDPModelTools.pdf(POMDPs.transition(pomdp,s,a), s_prime)
                            if p_s_prime == 0.0
                                continue
                            end
                            p_z = POMDPModelTools.pdf(POMDPs.observation(pomdp, a, s_prime), obs)
                            @deb("p_z = $p_z")
                            for (next, prob) in node.edges[a][obs]
                                if !haskey(controller.nodes, next.id)
                                    error("Node $(next.id) not present in nodes")
                                end
                                M[composite_index([temp_id[n_id], s_index],[n_nodes, n_states]), composite_index([temp_id[next.id], s_prime_index],[n_nodes,n_states])]-= POMDPs.discount(pomdp)*p_s_prime*p_z*p_a_n*prob
                                @deb("M[$n_id, $s][$(next.id), $s_prime] = gamma=$(POMDPs.discount(pomdp))*ps'=$p_s_prime*pz=$p_z*pa=$p_a_n*pn'=$prob = $(M[composite_index([temp_id[n_id], s_index],[n_nodes, n_states]), composite_index([temp_id[next.id], s_prime_index],[n_nodes,n_states])])")
                            end
                        end
                    end
                end
            end
        end
        @deb("M = $M")
        @deb("b = $b")
        res = M \ b
        #copy respective value functions in nodes
        for (n_id, node) in nodes
            node.value = copy(res[(temp_id[n_id]-1)*n_states+1 : temp_id[n_id]*n_states])
            @deb("Value vector of node $n_id = $(nodes[n_id].value)")
        end
end
