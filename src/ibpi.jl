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

function partial_backup!(controller::IPOMDPToolbox.Controller{A, W}, controller_j::IPOMDPToolbox.Controller{A, W}, ipomdp::IPOMDP{S, A, W}; minval = 0.0, add_one = false, debug_node = 0) where {S, A, W}
	#this time the matrix form is a1x1+...+anxn = b1
	#sum(a,s)[sum(nz)[canz*[R(s,a)+gamma*sum(s')p(s'|s, a)p(z|s', a)v(nz,s')]] -eps = V(n,s)
	#number of variables is |A||Z||N|+1 (canz and eps)
	nodes = controller.nodes
	nodes_j = controller_j.nodes
	n_nodes = length(nodes)
	#@deb(n_nodes)
	n_nodes_j = length(nodes_j)
	agent_i = IPOMDPs.agent(ipomdp)
	agent_j = IPOMDPs.agent(emulated_frames(ipomdp)[1])
	states = IPOMDPs.states(ipomdp)
	n_states = length(states)
	actions_i = actions_agent(agent_i)
	n_actions = length(actions_i)
	actions_j = actions_agent(agent_j)
	observations_i = observations_agent(agent_i)
	observations_j = observations_agent(agent_j)
	n_observations = length(observations_i)
	#vector containing the tangent belief states for all modified nodes
	tangent_b = Dict{Int64, Vector{Float64}}()
	#dim = n_nodes*n_actions*n_observations
	changed = false
	#M_TR =  zeros(n_actions, n_observations, n_nodes)
	#M_TL =  zeros(n_actions, n_observations, n_nodes)
	temp_id = Dict{Int64, Int64}()
	for real_id in keys(nodes)
			temp_id[real_id] = length(temp_id)+1
			#@deb("Node $real_id becomes $node_counter")
	end
	temp_id_j = Dict{Int64, Int64}()
	for real_id in sort(collect(keys(nodes_j)))
			temp_id_j[real_id] = length(temp_id_j)+1
			#@deb("Node $real_id becomes $node_counter")
	end
	for (n_id, node) in nodes
		@deb("Node to be improved: $n_id")
		lpmodel = JuMP.Model(with_optimizer(GLPK.Optimizer))
		#define variables for LP. c(a, n, z)
		@variable(lpmodel, canz[a=1:n_actions, z=1:n_observations, n=1:n_nodes] >= 0.0)
		@variable(lpmodel, ca[a=1:n_actions] >= 0.0)
		#e to maximize
		@variable(lpmodel, e)
		@objective(lpmodel, Max, e)
		#define constraints
		for s_index in 1:n_states
			s = states[s_index]
			for (nj_id, nj) in nodes_j
				M = zeros(n_actions, n_observations, n_nodes)
				M_a = zeros(n_actions)
				for ai_index in 1:n_actions
					ai = actions_i[ai_index]
					for (aj, p_aj) in nj.actionProb
						action_dict = Dict{Agent, Any}(agent_i => ai, agent_j => aj)
						r = IPOMDPs.reward(ipomdp, IPOMDPs.IS(s, Vector{IPOMDPToolbox.Model}(undef, 0)), action_dict)
						M_a[ai_index] += r * p_aj
						for zi_index in 1:n_observations
							zi = observations_i[zi_index]
							#array of edges given observation
							for s_prime_index in 1:length(states)
								s_prime = states[s_prime_index]
								transition_i =POMDPModelTools.pdf(IPOMDPs.transition(ipomdp,s,action_dict), s_prime)
								observation_i = POMDPModelTools.pdf(IPOMDPs.observation(ipomdp, s_prime, action_dict), zi)
								if transition_i != 0.0 && observation_i != 0.0
									for (zj, obs_dict_j) in nj.edges[aj]
										observation_j = POMDPModelTools.pdf(IPOMDPs.observation(ipomdp, s_prime, action_dict), zj)
										if observation_j != 0.0
											for (n_prime_j, prob_j) in obs_dict_j
												for (n_prime_i_index, n_prime_i) in nodes
													v_nz_sp = n_prime_i.value[s_prime_index,temp_id_j[n_prime_j.id]]
													#if n_id == 7 || n_id == 8
													@deb("state = $s, action_i = $ai, action_j = $aj, obs_i = $zi, obs_j = $zj n_prime_i = $(n_prime_i_index), s_prime = $s_prime")
													@deb("$transition_i * $observation_i * $observation_j * $prob_j * $v_nz_sp")
													#end
													M[ai_index, zi_index, temp_id[n_prime_i_index]]+= transition_i * observation_i * observation_j * prob_j * v_nz_sp
												end
											end
										end
									end
								end
								#@deb("M[$a_index, $obs_index, $nz_id] = $(M[a_index, obs_index, nz_id])")
							end
						end
					end
					#@deb(M)

				end
				@deb(M)
				@constraint(lpmodel,  e + node.value[s_index, temp_id_j[nj_id]] <= sum( M_a[a]*ca[a]+IPOMDPs.discount(ipomdp)*sum(sum( M[a, z, n] * canz[a, z, n] for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions))
			end
		end
		#sum canz over a,n,z = 1
		@constraint(lpmodel, con_sum[a=1:n_actions, z=1:n_observations], sum(canz[a, z, n] for n in 1:n_nodes) == ca[a])
		@constraint(lpmodel, ca_sum, sum(ca[a] for a in 1:n_actions) == 1.0)

		#if debug[] == true
		#	print(lpmodel)
		#end

		optimize!(lpmodel)
		@deb("$(termination_status(lpmodel))")
		@deb("$(primal_status(lpmodel))")
		@deb("$(dual_status(lpmodel))")
		#=
		if debug[] == true && n_id == debug_node
			L = 3
			GL = 2
			GR = 1
			TR = 1
			TL = 2
			@deb("correct vals")
			correct_val = -1 + POMDPs.discount(pomdp)*(M_TR[L, GL, temp_id[5]] +  M_TR[L, GR, temp_id[8]]) - node.value[TR]
			@deb("e <= $correct_val")
			correct_val = -1 +POMDPs.discount(pomdp)*(M_TL[L, GL, temp_id[5]] +  M_TL[L, GR, temp_id[8]]) - node.value[TL]
			@deb("e <= $correct_val")

			#actual_val = -1 + POMDPs.discount(pomdp)*(M_7_TR[L, GL, temp_id[10]]* JuMP.value(canz[L, GL, temp_id[10]]) +  M_7_TR[L, GR, temp_id[8]] * JuMP.value(canz[L, GR, temp_id[8]])+ M_7_TR[L, GR, temp_id[1]] * JuMP.value(canz[L, GR,temp_id[1]])) - node.value[TR]
			actual_val = -1 + POMDPs.discount(pomdp)* sum(sum(sum( M_TR[a,z,n] * JuMP.value(canz[a,z,n])  for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions) - node.value[TR]
			println("actual val")
			println("e <= $actual_val")
			#actual_val = -1 + POMDPs.discount(pomdp)*(M_7_TL[L, GL, temp_id[10]]* JuMP.value(canz[L, GL, temp_id[10]]) +  M_7_TL[L, GR, temp_id[8]] * JuMP.value(canz[L, GR, temp_id[8]])+ M_7_TL[L, GR, temp_id[1]] * JuMP.value(canz[L, GR, temp_id[1]])) - node.value[TL]
			actual_val = -1 + POMDPs.discount(pomdp)* sum(sum(sum( M_TL[a,z,n]*JuMP.value(canz[a,z,n]) for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions) - node.value[TL]

			println("e <= $actual_val")
			for n_id in keys(nodes)

				println("canz $n_id =  $(JuMP.value(canz[L, GR, temp_id[n_id]]))")
				println("canz $n_id =  $(JuMP.value(canz[L, GL, temp_id[n_id]]))")

			end
		end
		=#
		#@deb("eps = $(JuMP.value(e))")
		@deb("Obj = $(objective_value(lpmodel))")
		if JuMP.objective_value(lpmodel) > minval
			changed = true
			#@deb("Good so far")
			new_edges = Dict{A, Dict{W,Dict{Node, Float64}}}()
			new_actions = Dict{A, Float64}()
			#@deb("New structures created")
			for action_index in 1:n_actions
				ca_v = JuMP.value(ca[action_index])
				#@deb("Action $(actions[action_index])")
				#@deb("ca $(actions[action_index])= $ca_v")
				if ca_v > 1.0-minval
					ca_v = 1.0
				end
				if ca_v > minval
					new_obs = Dict{W, Dict{Node, Float64}}()
					for obs_index in 1:n_observations
						obs_total = 0.0
						#fill a temporary edge dict with unnormalized probs
						temp_edge_dict = Dict{Node, Float64}()
						for (nz_id, nz) in nodes
							prob = JuMP.value(canz[action_index, obs_index, temp_id[nz_id]])/ca_v
							#@deb("canz $(observations[obs_index]) -> $nz_id = $prob")
							if prob < 0.0
								#@deb("Set prob to 0 even though it was negative")
								prob = 0.0
							end
							if prob > 1.0 && prob < 1.0+minval
								#@deb("Set prob slightly greater than 1 to 1")
								prob = 1.0
							end
							if prob < 0.0 || prob > 1.0
								error("Probability outside of bounds: $prob")
							end
							if prob > 0.0
								obs_total+= prob
								#@deb("New edge: $(action_index), $(obs_index) -> $nz_id, $(prob)")
								temp_edge_dict[nz] = prob
							end
						end
						if obs_total == 0.0
							error("sum of prob for obs $(observations[obs_index]) == 0")
						end
						new_edge_dict = Dict{Node, Float64}()
						for (next, prob) in temp_edge_dict
							#@deb("normalized prob: $normalized_prob")
							if prob >= 1.0-minval
								new_edge_dict[next] = 1.0
							elseif prob > minval
								new_edge_dict[next] = prob
							end
							#do not add anything if prob < minval
						end
						#@deb("length of dict for obs $(observations[obs_index]) = $(length(new_edge_dict))")
						if length(new_edge_dict) != 0
							new_obs[observations[obs_index]] = new_edge_dict
							#update incoming edge vector for other node
							for (next, prob) in new_edge_dict
								if haskey(next.incomingEdgeDicts, node)
									push!(next.incomingEdgeDicts[node], new_edge_dict)
								else
									next.incomingEdgeDicts[node] = [new_edge_dict]
								end
							end
						end
					end
					if length(keys(new_obs)) != 0
						new_edges[actions[action_index]] = new_obs
						new_actions[actions[action_index]] = ca_v
					end
				end
			end
			node.edges = new_edges
			node.actionProb = new_actions
			if !add_one
				old_deb = debug[]
				debug[] = false
				#evaluate!(controller, pomdp)
				debug[] = old_deb
				if debug[] == true
					@deb("Changed controller after eval")
					for (n_id, node) in controller.nodes
						@deb(node)
					end
				end
			end
			if add_one
				#no need to update tangent points because they wont be used!
				if debug[] == true
					@deb("Changed node after eval")
					@deb(node)
				end
				break
			end
		end
		constraint_list = JuMP.all_constraints(lpmodel, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
		tangent_b[n_id] =  [-1*dual(constraint_list[s]) for s in 1:n_states]
	end
	return changed, tangent_b
end
