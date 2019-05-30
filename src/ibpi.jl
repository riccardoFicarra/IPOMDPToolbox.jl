
mutable struct InteractiveController{S, A, W} <: AbstractController
	level::Int64
	frame::IPOMDP{S, A, W}
	nodes::Dict{Int64, Node{A, W}}
	maxId::Int64
end

function init_controllers(ipomdp::IPOMDP{S,A,W}, pomdp::POMDP{A, W},maxlevel::Int64; force = 0) where {S, A, W}
    #for now i assume i modeling another agent same as him.
    controllers = Dict{Int64, AbstractController}()
    for l in maxlevel:-1:1
		if force == 0
			controller = InteractiveController(l, ipomdp)
		else
			controller = InteractiveController(l, ipomdp; force=force)
		end
        controllers[l] = controller
    end
	if force == 0
		controllers[0] = Controller(pomdp)
	else
		controllers[0] = Controller(pomdp; force = force)
	end
    return controllers
end

function InteractiveController(level::Int64, ipomdp::IPOMDP{S, A, W}; force=0) where {S, A, W}
	if force == 0
		newNode = InitialNode(actions_agent(ipomdp), observations_agent(ipomdp))
	else
		newNode = InitialNode(actions_agent(ipomdp), observations_agent(ipomdp); force=force)
	end
    return InteractiveController{S, A, W}(level, ipomdp, Dict(1 => newNode), 1)
end


function observation(frame::Any, s_prime::S, ai::A, aj::A) where {S, A}
	if typeof(frame) <: POMDP
		return POMDPs.observation(frame, aj, s_prime)
	elseif typeof(frame) <: IPOMDP
		return IPOMDPs.observation(frame, s_prime, ai, aj)
	else
		error("Wrong frame type in observation function call")
	end
end


# interactive -> interactive version
function evaluate!(controller::InteractiveController{A,W},  controller_j::AbstractController) where {S, A, W}
	ipomdp_i = controller.frame
	frame_j = controller_j.frame
    nodes = controller.nodes
    n_nodes = length(controller.nodes)
    nodes_j = controller_j.nodes
    n_nodes_j = length(nodes_j)
    states = IPOMDPs.states(ipomdp_i)
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
            for (nj_id, nj) in nodes_j
                M[s_index, temp_id_j[nj_id], temp_id[ni_id], s_index, temp_id_j[nj_id], temp_id[ni_id]] +=1
                for (ai, p_ai) in ni.actionProb
                    #@deb("ai = $ai")
                    @deb("ai = $ai")
                    for (aj, p_aj) in nj.actionProb
                        #@deb("aj = $aj")
                        @deb("aj = $aj")
                        r = IPOMDPs.reward(ipomdp_i, s, ai, aj)
                        #@deb("r = $s")
                        @deb("r = $r")
                        b[s_index, temp_id_j[nj_id], temp_id[ni_id]] = p_ai * p_aj * r
                        for (zi, obs_dict_i) in ni.edges[ai]
                            @deb("zi = $zi")
                            for s_prime_index in 1:n_states
                                s_prime = states[s_prime_index]
                                @deb("s_prime = $s_prime")
                                transition_i = POMDPModelTools.pdf(IPOMDPs.transition(ipomdp_i, s, ai, aj), s_prime)
                                observation_i = POMDPModelTools.pdf(IPOMDPs.observation(ipomdp_i, s_prime, ai, aj), zi)
                                @deb(transition_i)
                                @deb(observation_i)
                                for (zj, obs_dict_j) in nj.edges[aj]
                                    @deb("zj = $zj")
                                    observation_j = POMDPModelTools.pdf(observation(frame_j, s_prime, ai, aj), zj)
                                    for (n_prime_j, prob_j) in nj.edges[aj][zj]
                                        for (n_prime_i, prob_i) in ni.edges[ai][zi]
                                            M[s_index, temp_id_j[nj_id], temp_id[ni_id], s_prime_index, temp_id_j[n_prime_j.id], temp_id[n_prime_i.id]] -= p_ai * p_aj * IPOMDPs.discount(ipomdp_i) * transition_i * observation_i * observation_j * prob_j * prob_i
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
        @deb("Value vector of node $n_id = $(nodes[n_id].value)")
    end
end

function observations(frame::Any)
	if typeof(frame) <: POMDP
		return POMDPs.observations(frame)
	elseif typeof(frame) <: IPOMDP
		return IPOMDPs.observations_agent(frame)
	else
		error("Wrong frame type in observation function call")
	end
end

function actions(frame::Any)
	if typeof(frame) <: POMDP
		return POMDPs.actions(frame)
	elseif typeof(frame) <: IPOMDP
		return IPOMDPs.actions_agent(frame)
	else
		error("Wrong frame type in observation function call")
	end
end
function partial_backup!(controller::InteractiveController{A, W}, controller_j::AbstractController; minval = 0.0, add_one = true, debug_node = 0) where {S, A, W}
	#this time the matrix form is a1x1+...+anxn = b1
	#sum(a,s)[sum(nz)[canz*[R(s,a)+gamma*sum(s')p(s'|s, a)p(z|s', a)v(nz,s')]] -eps = V(n,s)
	#number of variables is |A||Z||N|+1 (canz and eps)
	frame = controller.frame
	frame_j = controller_j.frame
	nodes = controller.nodes
	nodes_j = controller_j.nodes
	n_nodes = length(nodes)
	#@deb(n_nodes)
	n_nodes_j = length(nodes_j)
	states = IPOMDPs.states(frame)
	n_states = length(states)
	actions_i = actions(frame)
	n_actions = length(actions_i)
	actions_j = actions(frame_j)
	observations_i = observations(frame)
	observations_j = observations(frame)
	n_observations = length(observations_i)
	#vector containing the tangent belief states for all modified nodes
	tangent_b = Dict{Int64, Array{Float64}}()
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
	for real_id_j in sort(collect(keys(nodes_j)))
			temp_id_j[real_id_j] = length(temp_id_j)+1
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
			#@deb("state $s")
			for (nj_id, nj) in nodes_j
				M = zeros(n_actions, n_observations, n_nodes)
				M_a = zeros(n_actions)
			#	@deb("\tnode_j $nj_id ")
				for ai_index in 1:n_actions
					ai = actions_i[ai_index]
			#		@deb("\t\tai = $ai ")
					for (aj, p_aj) in nj.actionProb
						r = IPOMDPs.reward(frame, s, ai, aj)
				#		@deb("\t\t\taj = $aj p_aj = $p_aj, r = $r")
						M_a[ai_index] += r * p_aj
						for zi_index in 1:n_observations
							zi = observations_i[zi_index]
							#array of edges given observation
							for s_prime_index in 1:length(states)
								s_prime = states[s_prime_index]
								transition_i =POMDPModelTools.pdf(IPOMDPs.transition(frame,s,ai, aj), s_prime)
								observation_i = POMDPModelTools.pdf(observation(frame, s_prime, ai, aj), zi)
								if transition_i != 0.0 && observation_i != 0.0
									for (zj, obs_dict_j) in nj.edges[aj]
										observation_j = POMDPModelTools.pdf(observation(frame_j, s_prime, ai, aj), zj)
										if observation_j != 0.0
											for (n_prime_j, prob_j) in obs_dict_j
												for (n_prime_i_index, n_prime_i) in nodes
													#n_prime_j is always n1 when you have two nodes!
													v_nz_sp = n_prime_i.value[s_prime_index,temp_id_j[n_prime_j.id]]
													#if n_id == 7 || n_id == 8
													#@deb("state = $s, action_i = $ai, action_j = $aj, obs_i = $zi, obs_j = $zj n_prime_i = $(n_prime_i_index), s_prime = $s_prime")
													#@deb("$transition_i * $observation_i * $observation_j * $prob_j * $v_nz_sp = $(p_aj* transition_i * observation_i * observation_j * prob_j * v_nz_sp)")
													#end
													M[ai_index, zi_index, temp_id[n_prime_i_index]]+= p_aj* transition_i * observation_i * observation_j * prob_j * v_nz_sp
												end
											end
										end
									end
								end
								#@deb("M[$a_index, $obs_index, $nz_id] = $(M[a_index, obs_index, nz_id])")
							end
						end
					end
				end
				@deb("state $s, node $nj_id")
				@deb(M)

				#@deb(M_a)
				#node.value[s_index, temp_id_j[nj_id]] is V(n, is)
				con = @constraint(lpmodel, e + node.value[s_index, temp_id_j[nj_id]] <= sum( M_a[a]*ca[a]+ IPOMDPs.discount(frame)*sum(sum( M[a, z, n] * canz[a, z, n] for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions))
				set_name(con, "$(s_index)_$(temp_id_j[nj_id])")
			end
		end
		#sum canz over a,n,z = 1

		@constraint(lpmodel, con_sum[a=1:n_actions, z=1:n_observations], sum(canz[a, z, n] for n in 1:n_nodes) == ca[a])
		@constraint(lpmodel, ca_sum, sum(ca[a] for a in 1:n_actions) == 1.0)

		#if :data in debug
		#	print(lpmodel)
		#end

		optimize!(lpmodel)
		@deb("$(termination_status(lpmodel))")
		@deb("$(primal_status(lpmodel))")
		@deb("$(dual_status(lpmodel))")

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
							prob = JuMP.value(canz[action_index, obs_index, temp_id[nz_id]])
							#@deb("canz $(observations[obs_index]) -> $nz_id = $prob")
							if prob < 0.0
								@deb("Set prob to 0 even though it was negative")
								prob = 0.0
							end
							if prob > 1.0 && prob < 1.0+minval
								@deb("Set prob slightly greater than 1 to 1")
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
							new_obs[observations_i[obs_index]] = new_edge_dict
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
						new_edges[actions_i[action_index]] = new_obs
						new_actions[actions_i[action_index]] = ca_v
					end
				end
			end
			node.edges = new_edges
			node.actionProb = new_actions
			if add_one
				#no need to update tangent points because they wont be used!
				if :flow in debug
					@deb("Changed node after eval")
					@deb(node)
				end
				return true, Dict{Int64, Array{Float64}}()
			end
		end
		#constraint_list = JuMP.all_constraints(lpmodel, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
		tangent_belief = Array{Float64}(undef, n_states, n_nodes_j)
		for s_index in 1:n_states
			for nj_id in keys(nodes_j)
				@deb("nj = $nj_id")
				@deb("nj_temp = $(temp_id_j[nj_id])")
				con = constraint_by_name(lpmodel, "$(s_index)_$(temp_id_j[nj_id])")
				tangent_belief[s_index, temp_id_j[nj_id]] =  -1*dual(con)

				#=
				if tangent_belief[s, nj] <= 0.0
					error("belief == $(tangent_belief[s, nj])")
				end
				=#
				@deb("state $s_index node $nj_id")
				if :data in debug
					println(con)
				end
				@deb(-1*dual(con))
			end
		end
		tangent_b[n_id] = tangent_belief
	end
	return changed, tangent_b
end

function full_backup_generate_nodes(controller::InteractiveController{A, W}, controller_j::AbstractController, minval::Float64) where {A, W}
	minval = 1e-10
	frame = controller.frame
	frame_j = controller_j.frame
	nodes = controller.nodes
	nodes_j = controller_j.nodes
	n_nodes = length(nodes)
	#@deb(n_nodes)
	n_nodes_j = length(nodes_j)
	states = IPOMDPs.states(frame)
	n_states = length(states)
	actions_i = actions(frame)
	n_actions = length(actions_i)
	observations_i = observations(frame)
	n_observations = length(observations_i)
	#tentative from incpruning
	#prder of it -> actions, obs
	#for each a, z produce n new nodes (iterate on nodes)
	#for each node iterate on s and s' to produce a new node
	#new node is tied to old node?, action a and obs z
	#with stochastic pruning we get the cis needed

	temp_id_j = Dict{Int64, Int64}()
	for real_id in sort(collect(keys(nodes_j)))
			temp_id_j[real_id] = length(temp_id_j)+1
			#@deb("Node $real_id becomes $node_counter")
	end

	new_nodes = Set{Node}()
	#new nodes counter used mainly for debugging, counts backwards (gets overwritten eventually)
	new_nodes_counter = -1
	for ai in actions_i
		#this data structure has the set of nodes for each observation (new_nodes_z[obs] = Set{Nodes} generated from obs)
		new_nodes_z = Vector{Set{Node}}(undef, length(observations_i))
		for zi_index in 1:length(observations_i)
			zi = observations_i[zi_index]
			#this set contains all new nodes for action, obs for all nodes
			new_nodes_a_z = Set{Node}()
			for (ni_id, ni) in nodes
				new_v = node_value(ni, ai, zi, controller_j, frame, temp_id_j)
				#do not set node id for now
				#new_node = build_node(new_nodes_counter, [a], [1.0], [[obs]], [[1.0]], [[node]], new_v)
				new_node = build_node(new_nodes_counter, ai, zi, ni, new_v)
				push!(new_nodes_a_z, new_node)
				new_nodes_counter -=1
			end
			if :flow in debug
				println("New nodes created:")
				for node in new_nodes_a_z
					println(node)
				end
			end
			new_nodes_z[zi_index] = filterNodes(new_nodes_a_z, minval)
		end
		#set that contains all nodes generated from action a after incremental pruning
		new_nodes_counter, new_nodes_a = incprune(new_nodes_z, new_nodes_counter, minval)
		union!(new_nodes, new_nodes_a)
	end
	#all new nodes, final filtering
	filtered_nodes = filterNodes(new_nodes, minval)
	for new_node in new_nodes
		for (a, a_dict) in new_node.edges
			if length(a_dict) != n_observations
				error("Invalid node produced!")
			end
		end
	end
	return filtered_nodes
end
#using eq τ(n, a, z) from incremental pruning paper
function node_value(ni::Node{A, W}, ai::A, zi::W, controller_j::AbstractController, ipomdp::IPOMDP, temp_id_j::Dict{Int64, Int64}) where {A, W}
	states = IPOMDPs.states(ipomdp)
	frame_j = controller_j.frame
	nodes_j = controller_j.nodes
	n_states = length(states)
	n_observations = length(observations_agent(ipomdp))
	γ = IPOMDPs.discount(ipomdp)
	new_V = zeros(Float64,n_states, length(nodes_j))
	for s_index in 1:n_states
		s = states[s_index]
		for (nj_id, nj) in nodes_j
			for (aj, aj_prob) in nj.actionProb
				r = IPOMDPs.reward(ipomdp, s, ai, aj)
				sum = 0.0
				for s_prime_index in 1:n_states
					s_prime = states[s_prime_index]
					transition_i =POMDPModelTools.pdf(IPOMDPs.transition(ipomdp,s,ai, aj), s_prime)
					observation_i = POMDPModelTools.pdf(IPOMDPs.observation(ipomdp, s_prime, ai, aj), zi)
					if transition_i != 0.0 && observation_i != 0.0
						for (zj, obs_dict_j) in nj.edges[aj]
							observation_j = POMDPModelTools.pdf(observation(frame_j, s_prime, ai, aj), zj)
							for (nj_prime, prob_nj_prime) in obs_dict_j
								#@deb("$action, $state, $observation, $s_prime")
								#@deb("$(node.value[s_prime_index]) * $(p_obs) * $(p_s_prime)")
								sum+= ni.value[s_prime_index, temp_id_j[nj_prime.id]] *  aj_prob * transition_i * observation_i * observation_j * prob_nj_prime
							end
						end
					end
				end
				new_V[s_index, temp_id_j[nj_id]] = (1/n_observations) * IPOMDPs.reward(ipomdp, s, ai, aj) + γ*sum
			end


		end
	end
	return new_V
end


function escape_optima_standard!(controller::InteractiveController{A, W}, controller_j::AbstractController, tangent_b::Dict{Int64, Array{Float64}}; add_all=false, minval = 0.0) where {A, W}
	#@deb("$tangent_b")
	frame_i = controller.frame
	frame_j = controller_j.frame
	nodes = controller.nodes
	nodes_j = controller_j.nodes
	n_nodes_j = length(nodes_j)
	n_nodes = length(nodes)
	#@deb(n_nodes)
	n_nodes_j = length(nodes_j)
	states = IPOMDPs.states(frame_i)
	n_states = length(states)
	actions_i = actions(frame_i)
	n_actions = length(actions_i)
	actions_j = actions(frame_j)
	observations_i = observations(frame_i)
	observations_j = observations(frame_j)
	n_observations = length(observations_i)

	if length(tangent_b) == 0
		error("tangent_b was empty!")
	end


	temp_id_j = Dict{Int64, Int64}()
	for real_id in sort(collect(keys(nodes_j)))
			temp_id_j[real_id] = length(temp_id_j)+1
			#@deb("Node $real_id becomes $node_counter")
	end

	old_deb = debug[]
	debug[] = false

	new_nodes = full_backup_generate_nodes(controller, controller_j, minval)

	debug[] = old_deb
	if :data in debug
		println("new_nodes:")
		for node in new_nodes
			println(node)
		end
	end



	escaped = false
	reachable_b = Set{Array{Float64}}()
	for (id, start_b) in tangent_b
		#id = collect(keys(tangent_b))[1]
		#start_b = tangent_b[id]
		@deb(start_b)
		@deb("$id - >$start_b")
		for ai in keys(nodes[id].actionProb)
			for zi in observations_i
				new_b = zeros(n_states, n_nodes_j)
				normalize = 0.0
				for s_prime_index in 1:n_states
					s_prime = states[s_prime_index]
					for s_index in 1:n_states
						s = states[s_index]
						for (nj_id, nj) in nodes_j
							@deb("$(start_b[s_index, temp_id_j[nj_id]])")
							if start_b[s_index, temp_id_j[nj_id]] == 0.0
								continue
							end
							for (aj, aj_prob) in nj.actionProb
								transition_i = POMDPModelTools.pdf(IPOMDPs.transition(frame_i, s, ai, aj), s_prime)
								observation_i = POMDPModelTools.pdf(IPOMDPs.observation(frame_i, s_prime, ai, aj), zi)
								if transition_i == 0.0 || observation_i == 0.0 || aj_prob == 0.0
									continue
								end
								@deb("\t $aj_prob $transition_i $observation_i")
								for (zj, obs_dict) in nj.edges[aj]
									observation_j = POMDPModelTools.pdf(observation(frame_j, s_prime, ai, aj), zj)
									if observation_j == 0.0
										continue
									end
									@deb("\t\t $observation_j")
									for (n_prime_j, prob_j) in obs_dict
										#FIXME is this the right prob to use? last element of 1.8
										@deb("start_b = $(start_b[s_index, temp_id_j[nj_id]])")
										@deb("adding $(start_b[s_index, temp_id_j[nj_id]]) * $aj_prob * $transition_i* $observation_i * $observation_j * $prob_j")
										@deb("adding $(start_b[s_index, temp_id_j[nj_id]] * aj_prob * transition_i* observation_i * observation_j * prob_j)")
										new_b[s_prime_index, temp_id_j[n_prime_j.id]] += start_b[s_index, temp_id_j[nj_id]] * aj_prob * transition_i* observation_i * observation_j * prob_j
										normalize += start_b[s_index, temp_id_j[nj_id]] * aj_prob * transition_i* observation_i * observation_j * prob_j
									end
								end
							end
						end
					end
				end
				if normalize == 0.0
					error("normalization constant is $normalize !")
				end
				new_b = new_b  ./ normalize
				@deb("from belief $start_b action $ai and obs $zi -> $new_b")
				push!(reachable_b, new_b)
				#find the value of the current controller in the reachable belief state
				best_old_node = nothing
				best_old_value = 0.0
				for (id,old_node) in controller.nodes
					@deb("new_b = $new_b")
					@deb("old value = $(old_node.value)")
					temp_value = sum(sum(new_b[s, n] * old_node.value[s, n] for s in 1:n_states) for n in 1:n_nodes_j)
					@deb("temp value = $temp_value")
					if best_old_node == nothing || best_old_value < temp_value
						best_old_node = old_node
						best_old_value = temp_value
					end
				end
				#find the value of the backed up controller in the reachable belief state
				best_new_node = nothing
				best_new_value = 0.0
				for new_node in new_nodes
					@deb("new_b = $new_b")
					@deb("new value = $(new_node.value)")
					new_value =  sum(sum(new_b[s, n] * new_node.value[s, n] for s in 1:n_states) for n in 1:n_nodes_j)
					@deb("new value tot= $new_value")
					if best_new_node == nothing || best_new_value < new_value
						best_new_node = new_node
						best_new_value = new_value
					end
				end
				if best_new_value - best_old_value > minval
					@deb("in $new_b node $(best_new_node.id) has $best_new_value > $best_old_value")
					reworked_node = rework_node(controller, best_new_node)
					controller.nodes[reworked_node.id] = reworked_node
					controller.maxId+=1
					@deb("Added node $(reworked_node.id)")

					@deb(controller.nodes[reworked_node.id])
					escaped = true
				end
			end
		end
		if escaped && !add_all
			return true
		end
	end
	#@deb("$reachable_b")
	return escaped
end

function rework_node(controller::AbstractController, new_node::Node{A, W}) where {A, W}
		id = controller.maxId+1
		actionProb = copy(new_node.actionProb)
		value = copy(new_node.value)
		edges = Dict{A, Dict{W, Dict{Node, Float64}}}()
		for (a, obs_dict) in new_node.edges
			edges[a] = Dict{W, Dict{Node, Float64}}()
			for (z, node_dict) in obs_dict
				edges[a][z] = Dict{Node,Float64}()
				for (node, prob) in node_dict
					current_controller_node = controller.nodes[node.id]
					edges[a][z][current_controller_node] = prob
				end
			end
		end
		return Node(id, actionProb,edges, value, Dict{Node, Vector{Dict{Node, Float64}}}())
end


function full_backup_stochastic!(controller::InteractiveController{A, W}, controller_j::AbstractController; minval = 0.0) where {A, W}
	nodes = controller.nodes
	# observations = observations(controller_i.frame)
	#tentative from incpruning
	#prder of it -> actions, obs
	#for each a, z produce n new nodes (iterate on nodes)
	#for each node iterate on s and s' to produce a new node
	#new node is tied to old node?, action a and obs z
	#with stochastic pruning we get the cis needed
	new_nodes = full_backup_generate_nodes(controller, controller_j, minval)
	#before performing filtering with the old nodes update incomingEdge structure of old nodes
	#also assign permanent ids
	nodes_counter = controller.maxId+1
	for new_node in new_nodes
		@deb("Node $(new_node.id) becomes node $(nodes_counter)")
		new_node.id = nodes_counter
		if :flow in debug
			println(new_node)
		end
		nodes_counter+=1
		for (action, observation_map) in new_node.edges
			for (observation, edge_map) in observation_map
				#@deb("Obs $observation")
				for (next, prob) in edge_map
					#@deb("adding incoming edge from $(new_node.id) to $(next.id) ($action, $observation)")
					if haskey(next.incomingEdgeDicts, new_node)
						#@deb("it was the $(length(next.incomingEdgeDicts[new_node])+1)th")
						push!(next.incomingEdgeDicts[new_node], edge_map)
					else
						#@deb("it was the first edge for $(new_node.id)")
						next.incomingEdgeDicts[new_node] = [edge_map]
					end
				end
			end
		end
	end
	new_max_id = nodes_counter-1
	#add new nodes to controller
	#i want to have the new nodes first, so in case there are two nodes with identical value the newer node is pruned and we skip rewiring
	#no need to sort, just have new nodes examined before old nodes
	#the inner union is needed to have an orderedset as first parameter, as the result of union({ordered}, {not ordered}) = {ordered}
	orderedset = union(union(OrderedSet{Node}(), new_nodes), Set{Node}(oldnode for oldnode in values(nodes)))
	all_nodes = filterNodes(orderedset, minval)
	new_controller_nodes = Dict{Int64, Node}()
	for node in all_nodes
		#add nodes to the controller
		new_controller_nodes[node.id] = node
	end
	controller.nodes = new_controller_nodes
	controller.maxId = new_max_id
end
