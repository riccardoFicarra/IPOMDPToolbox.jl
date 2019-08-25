
mutable struct InteractiveController{S, A, W} <: AbstractController
	level::Int64
	frame::IPOMDP{S, A, W}
	nodes::Vector{Node{A, W}}
	stats::solver_statistics
	converged::Bool
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

function InteractiveController(level::Int64, ipomdp::IPOMDP{S, A, W}; force = 0) where {S, A, W}
	if force == 0
		#random action
		newNode = InitialNode([rand(actions_agent(ipomdp))], observations_agent(ipomdp))
	else
		if force > length(actions_agent(ipomdp))
			error("Forced action outisde of bounds")
		else
			newNode = InitialNode([actions_agent(ipomdp)[force]], observations_agent(ipomdp))
		end
	end
	return InteractiveController{S, A, W}(level, ipomdp, [newNode], solver_statistics(), false)
end


function observation(frame::Any, s_prime::S, ai::A, aj::A) where {S, A}
	if typeof(frame) <: POMDP
		return POMDPs.observation(frame, s_prime, ai)
	elseif typeof(frame) <: IPOMDP
		return IPOMDPs.observation(frame, s_prime, ai, aj)
	else
		error("Wrong frame type in observation function call")
	end
end



function evaluate!(controller::InteractiveController{A,W}, controllers_j::Array{AbstractController, 1}) where {S, A, W}
	#log_n_nodes(controller.stats, length(controller.nodes))
	#start_time(controller.stats,  "eval")
	ipomdp_i = controller.frame
	nodes = controller.nodes
	n_nodes = length(controller.nodes)
	states = IPOMDPs.states(ipomdp_i)
	n_states = length(states)

	n_controllers_j = length(controllers_j)
	@deb("length of controllers_j = $n_controllers_j", :multiple)

	# #dictionary used for recompacting ids
	# temp_id = Dict{Int64, Int64}()
	#
	# for (node_id, node) in nodes
	#     temp_id[node_id] = length(temp_id)+1
	# end

	#dictionary used for recompacting ids -> they are sorted!
	#concatenate all nodes from all lower level controllers
	temp_id_j = Array{Array{Int64, 1}, 1}(undef, n_controllers_j)
	node_counter = 1
	for controller_index in 1:n_controllers_j
		controller_j = controllers_j[controller_index]
		nodes_j = controller_j.nodes
		#initialize inner array
		temp_id_j[controller_index] = Array{Int64, 1}(undef, length(controller_j.nodes))
		#quick fix to have the values in some order
		for node_id in sort(collect(keys(nodes_j)))
			temp_id_j[controller_index][node_id] = node_counter
			node_counter += 1
		end
	end
	@deb("$temp_id_j", :multiple)
	n_nodes_j = node_counter-1
	@deb("total nodes in j: $n_nodes_j", :multiple)
	#M[s, nj, ni, s', nj', ni']
	M = zeros(n_states, n_nodes_j, n_nodes, n_states, n_nodes_j, n_nodes)
	b = zeros(n_states, n_nodes_j, n_nodes)
	#compute coefficients for sum(a)[R(s|a)*P(a|n)+gamma*sum(z, n', s')[P(s'|s,a)*P(z|s',a)*P(a|n)*P(n'|z)*V(nz, s')]]
	#start_time(controller.stats, "eval_coeff")
	for ni in nodes
		ni_id = ni.id
		#M is the coefficient matrix (form x1 = a2x2+...+anxn+b)
		#b is the constant term vector
		#variables are all pairs of n,s
		for s_index in 1:n_states
			s = states[s_index]
			for controller_index in 1:n_controllers_j
				controller_j = controllers_j[controller_index]
				nodes_j = controller_j.nodes
				frame_j = controller_j.frame
				for nj in nodes_j
					nj_id = nj.id
					M[s_index, temp_id_j[controller_index][nj_id], ni_id, s_index, temp_id_j[controller_index][nj_id], ni_id] +=1
					for (ai, p_ai) in ni.actionProb
						#@deb("ai = $ai")
						@deb("ai = $ai")
						for (aj, p_aj) in nj.actionProb
							#@deb("aj = $aj")
							@deb("aj = $aj")
							r = IPOMDPs.reward(ipomdp_i, s, ai, aj)
							#@deb("r = $s")
							@deb("r = $r")
							b[s_index, temp_id_j[controller_index][nj_id], ni_id] += p_ai * p_aj * r
							for (zi, obs_dict_i) in ni.edges[ai]
								@deb("zi = $zi")
								for s_prime_index in 1:n_states
									s_prime = states[s_prime_index]
									@deb("s_prime = $s_prime")
									transition_i = POMDPModelTools.pdf(IPOMDPs.transition(ipomdp_i, s, ai, aj), s_prime)
									observation_i = POMDPModelTools.pdf(IPOMDPs.observation(ipomdp_i, s_prime, ai, aj), zi)
									partial_mult_start = p_ai * p_aj * IPOMDPs.discount(ipomdp_i) * transition_i * observation_i
									@deb(transition_i)
									@deb(observation_i)
									for (zj, obs_dict_j) in nj.edges[aj]
										@deb("zj = $zj")
										observation_j = POMDPModelTools.pdf(observation(frame_j, s_prime, aj, ai), zj)
										partial_mult_zj = partial_mult_start * observation_j
										for (n_prime_j_id, prob_j) in nj.edges[aj][zj]
											n_prime = controller_j.nodes[n_prime_j_id]
											partial_mult_pj = partial_mult_zj * prob_j
											for (n_prime_i_id, prob_i) in ni.edges[ai][zi]
												n_prime_i = controller.nodes[n_prime_i_id]
												M[s_index, temp_id_j[controller_index][nj_id], ni_id, s_prime_index, temp_id_j[controller_index][n_prime_j_id], n_prime_i.id] -= partial_mult_pj * prob_i
												#M[s_index, temp_id_j[controller_index][nj_id], ni_id, s_prime_index, temp_id_j[controller_index][n_prime_j.id], temp_id[n_prime_i.id]] -= p_ai * p_aj * IPOMDPs.discount(ipomdp_i) * transition_i * observation_i * observation_j * prob_j * prob_i
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
	end
	#TL, all njs, only ni = 1, goes to TL
	@deb(M[1, :, 1, 1, :, 1], :multiple)
	M_2d = reshape(M,n_states* n_nodes_j* n_nodes, n_states* n_nodes_j* n_nodes)
	b_1d = reshape(b, n_states* n_nodes_j* n_nodes)
	#stop_time(controller.stats, "eval_coeff")
	#start_time(controller.stats, "eval_solve")
	res_1d = M_2d \ b_1d
	#stop_time(controller.stats, "eval_solve")
	res = reshape(res_1d, n_states, n_nodes_j, n_nodes)
	#copy respective value functions in nodes
	for node in nodes
		controller.nodes[node.id] = Node(node.id, node.actionProb, node.edges, copy(res[:, :, node.id]))
		@deb("Value vector of node $node.id = $(nodes[node.id].value)")
		node = nothing
	end
	#stop_time(controller.stats, "eval")
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

function partial_backup!(controller::InteractiveController{A, W}, controllers_j::Array{AbstractController, 1}; add_one = true) where {S, A, W}
	#debug = Set([:flow])
	#this time the matrix form is a1x1+...+anxn = b1
	#sum(a,s)[sum(nz)[canz*[R(s,a)+gamma*sum(s')p(s'|s, a)p(z|s', a)v(nz,s')]] -eps = V(n,s)
	#number of variables is |A||Z||N|+1 (canz and eps)
	#start_time(controller.stats, "partial")
	frame = controller.frame
	nodes = controller.nodes
	n_controllers_j = length(controllers_j)
	n_nodes = length(nodes)
	states = IPOMDPs.states(frame)
	n_states = length(states)
	actions_i = actions(frame)
	n_actions = length(actions_i)
	observations_i = observations(frame)
	n_observations = length(observations_i)
	#vector containing the tangent belief states for all modified nodes
	tangent_b = Dict{Int64, Array{Float64}}()
	#dim = n_nodes*n_actions*n_observations
	changed = false
	#M_TR =  zeros(n_actions, n_observations, n_nodes)
	#M_TL =  zeros(n_actions, n_observations, n_nodes)
	# temp_id = Dict{Int64, Int64}()
	# for real_id in keys(nodes)
	# 		temp_id[real_id] = length(temp_id)+1
	# 		#@deb("Node $real_id becomes $node_counter")
	# end

	temp_id_j = Array{Array{Int64, 1}, 1}(undef, n_controllers_j)
	node_counter = 1
	for controller_index in 1:n_controllers_j
		controller_j = controllers_j[controller_index]
		nodes_j = controller_j.nodes
		#initialize inner array
		temp_id_j[controller_index] = Array{Int64, 1}(undef, length(controller_j.nodes))
		#quick fix to have the values in some order
		for node_id in sort(collect(keys(nodes_j)))
			temp_id_j[controller_index][node_id] = node_counter
			node_counter += 1
		end
	end
	n_nodes_j = node_counter-1
	@deb("total nodes in j: $n_nodes_j", :multiple)
	constraints = Array{ConstraintRef}(undef, n_states, n_nodes_j)

	nodecounter= 0
	for node in nodes
		n_id = node.id
		@deb("Node to be improved: $n_id", :checkNodes)
		nodecounter += 1
		if nodecounter >= (length(nodes)/100)
			for i in 1:(nodecounter * 100 / length(nodes))
				print("|")
			end
			nodecounter = 0
		end

		lpmodel = JuMP.Model(with_optimizer(CPLEX.Optimizer; CPX_PARAM_SCRIND=0))

		#define variables for LP. c(a, n, z)
		@variable(lpmodel, canz[a=1:n_actions, z=1:n_observations, n=1:n_nodes] >= 0.0)
		@variable(lpmodel, ca[a=1:n_actions] >= 0.0)
		#e to maximize
		@variable(lpmodel, e)
		@objective(lpmodel, Max, e)
		#define constraints
		#start_time(controller.stats, "partial_coeff")
		@deb("Started computing coeff", :multiple)
		for s_index in 1:n_states
			s = states[s_index]
			@deb("state $s", :lpdual)
			for controller_index in 1:n_controllers_j
				controller_j = controllers_j[controller_index]
				frame_j = controller_j.frame
				nodes_j = controller_j.nodes
				for nj in nodes_j
					nj_id = nj.id
					M = zeros(n_actions, n_observations, n_nodes)
					M_a = zeros(n_actions)
					@deb("\tnode_j $nj_id ")
					for ai_index in 1:n_actions
						ai = actions_i[ai_index]
						@deb("\t\tai = $ai ")
						for (aj, p_aj) in nj.actionProb
							r = IPOMDPs.reward(frame, s, ai, aj)
							# @deb("\t\t\taj = $aj p_aj = $p_aj, r = $r", :lpdual)
							M_a[ai_index] += r * p_aj
							for zi_index in 1:n_observations
								zi = observations_i[zi_index]
								#array of edges given observation
								for s_prime_index in 1:length(states)
									s_prime = states[s_prime_index]
									transition_i =POMDPModelTools.pdf(IPOMDPs.transition(frame,s,ai, aj), s_prime)
									observation_i = POMDPModelTools.pdf(observation(frame, s_prime, ai, aj), zi)
									partial_mult_s_prime = p_aj * transition_i * observation_i
									if transition_i != 0.0 && observation_i != 0.0
										for (zj, obs_dict_j) in nj.edges[aj]
											observation_j = POMDPModelTools.pdf(observation(frame_j, s_prime, aj, ai), zj)
											partial_mult_zj = partial_mult_s_prime * observation_j
											if observation_j != 0.0
												for (n_prime_j_id, prob_j) in obs_dict_j
													n_prime_j = controller_j.nodes[n_prime_j_id]
													partial_mult_pj = partial_mult_zj * prob_j
													for n_prime_i in nodes
														#n_prime_j is always n1 when you have two nodes!
														n_prime_i_index = n_prime_i.id
														v_nz_sp = n_prime_i.value[s_prime_index, temp_id_j[controller_index][n_prime_j.id]]
														#M[ai_index, zi_index, temp_id[n_prime_i_index]]+= p_aj* transition_i * observation_i * observation_j * prob_j * v_nz_sp
														M[ai_index, zi_index, n_prime_i_index]+= partial_mult_pj * v_nz_sp
														# if p_aj* transition_i * observation_i * observation_j * prob_j * v_nz_sp - partial_mult_pj * v_nz_sp > 1e-10
														# 	error("Wrong partial mult")
														# end
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
					# @deb("state $s, node $nj_id", :lpdual)
					@deb(M, :data)

					#@deb(M_a)
					#node.value[s_index, temp_id_j[controller_index][nj_id]] is V(n, is)
					@deb("$(node.value)", :multiple)
					@deb("$(temp_id_j[controller_index][nj_id])", :multiple)
					constraints[s_index, temp_id_j[controller_index][nj_id]] = @constraint(lpmodel, e + node.value[s_index, temp_id_j[controller_index][nj_id] ] <= sum( M_a[a]*ca[a]+ IPOMDPs.discount(frame)*sum(sum( M[a, z, n] * canz[a, z, n] for n in 1:n_nodes) for z in 1:n_observations) for a in 1:n_actions))
					#set_name(con, "$(s_index)_$(temp_id_j[controller_index][nj_id])")
					@deb("constraint set", :multiple)

				end
			end
		end
		#stop_time(controller.stats, "partial_coeff")

		#sum canz over a,n,z = 1
		@constraint(lpmodel, con_sum[a=1:n_actions, z=1:n_observations], sum(canz[a, z, n] for n in 1:n_nodes) == ca[a])
		@constraint(lpmodel, ca_sum, sum(ca[a] for a in 1:n_actions) == 1.0)

		# if :lpdual in debug
		# 	print(lpmodel)
		# end
		#start_time(controller.stats, "partial_optimize")
		optimize!(lpmodel)
		#stop_time(controller.stats, "partial_optimize")

		@deb("has duals=$(has_duals(lpmodel))", :lpdual)

		@deb("$(termination_status(lpmodel))" , :lpdual)
		@deb("$(primal_status(lpmodel))", :lpdual)
		@deb("$(dual_status(lpmodel))", :lpdual)

		@deb("Obj = $(objective_value(lpmodel))", :lpdual)
		delta = JuMP.objective_value(lpmodel)
		if delta > config.minval
			#just to go on a newline after the progress bar
			@deb("", :flow)
			@deb("Improvement $delta", :flow)
			changed = true
			# @deb("Node $n_id can be improved", :flow)
			new_edges = Dict{A, Dict{W,Vector{Pair{Int64, Float64}}}}()
			new_actions = Dict{A, Float64}()
			#@deb("New structures created")
			for action_index in 1:n_actions
				ca_v = JuMP.value(ca[action_index])
				#@deb("Action $(actions[action_index])")
				#@deb("ca $(actions[action_index])= $ca_v")
				# if ca_v > 1.0-config.minval
				# 	ca_v = 1.0
				# end
				if ca_v > config.min_improvement
					new_obs = Dict{W, Vector{Pair{Int64, Float64}}}()
					for obs_index in 1:n_observations
						# obs_total = 0.0
						#fill a temporary edge dict with unnormalized probs
						temp_edge_dict = Vector{Pair{Int64, Float64}}(undef, 0)
						for nz in nodes
							prob = JuMP.value(canz[action_index, obs_index,nz.id])
							#@deb("canz $(observations[obs_index]) -> $nz_id = $prob")
							# if prob < 0.0
							# 	@deb("Set prob to 0 even though it was negative", :data)
							# 	prob = 0.0
							# end
							# if prob > 1.0 && prob < 1.0+config.minval
							# 	@deb("Set prob slightly greater than 1 to 1", :data)
							# 	prob = 1.0
							# end
							# if prob < 0.0 || prob > 1.0
							# 	#error("Probability outside of bounds: $prob")
							# end
							if prob > config.minval
								# obs_total+= prob
								#@deb("New edge: $(action_index), $(obs_index) -> $nz_id, $(prob)")
								push!(temp_edge_dict, (nz.id => prob))
							end
						end
						# if obs_total == 0.0
						# 	for nz_id in 1:length(nodes)
						# 		println("$(JuMP.value(canz[action_index, obs_index, nz_id]))")
						# 	end
						# 	error("sum of prob for obs $(observations_i[obs_index]) == 0")
						# end
						new_edge_dict = Vector{Pair{Int64, Float64}}()
						for (next_id, prob) in temp_edge_dict
							#@deb("normalized prob: $normalized_prob")
							# if prob >= 1.0-config.minval
							# 	push!(new_edge_dict, (next_id => 1.0))
							# elseif prob > config.minval
								# push!(new_edge_dict, (next_id => prob/obs_total))
								push!(new_edge_dict, (next_id => prob))


							# end
							#do not add anything if prob < minval
						end
						#@deb("length of dict for obs $(observations[obs_index]) = $(length(new_edge_dict))")
						if length(new_edge_dict) != 0
							new_obs[observations_i[obs_index]] = new_edge_dict
						end
					end
					if length(keys(new_obs)) != 0
						new_edges[actions_i[action_index]] = new_obs
						new_actions[actions_i[action_index]] = ca_v
					end
				end
			end
			new_node = Node(node.id, new_actions, new_edges, Array{Float64}(undef, 0,0))
			checkNode(new_node, controller; normalize = config.normalize)
			controller.nodes[new_node.id] = new_node
			node = nothing
			if add_one
				#no need to update tangent points because they wont be used!
				println()
				@deb("Changed node after eval", :partial)
				@deb(new_node, :partial)
				return true, Dict{Int64, Array{Float64}}()
			end
		end
		@deb("Node changed", :multiple)
		#constraint_list = JuMP.all_constraints(lpmodel, GenericAffExpr{Float64,VariableRef}, MOI.LessThan{Float64})
		#@deb("No node improved", :flow)
		tangent_belief = Array{Float64}(undef, n_states, n_nodes_j)
		for s_index in 1:n_states
			for controller_index in 1:n_controllers_j
				controller_j = controllers_j[controller_index]
				frame_j = controller_j.frame
				nodes_j = controller_j.nodes
				for nj_id in keys(nodes_j)
					# con = constraint_by_name(lpmodel, "$(s_index)_$(temp_id_j[controller_index][nj_id])")
					@deb("state $s_index node $nj_id", :lpdual)
					@deb(shadow_price(constraints[s_index, temp_id_j[controller_index][nj_id]]), :lpdual)
					tangent_belief[s_index, temp_id_j[controller_index][nj_id]] =  shadow_price(constraints[s_index, temp_id_j[controller_index][nj_id]])
					# @deb(dual(constraints[s_index, temp_id_j[controller_index][nj_id]]), :lpdual)
					# tangent_belief[s_index, temp_id_j[controller_index][nj_id]] =  -1*dual(constraints[s_index, temp_id_j[controller_index][nj_id]])



					# if tangent_belief[s_index, temp_id_j[controller_index][nj_id]] <= 0.0
					# 	@warn "belief == $(tangent_belief[s_index, temp_id_j[controller_index][nj_id]])"
					# end

					# if :data in debug
					# 	println(constraints[s_index, temp_id_j[controller_index][nj_id]])
					# end
					#@deb(-1*dual(constraints[s_index, temp_id_j[controller_index][nj_id]]), :data)
				end
			end
		end
		@deb(tangent_belief, :lpdual)
		tangent_b[n_id] = tangent_belief
	end
	println()
	#stop_time(controller.stats, "partial")
	return changed, tangent_b
end


# function full_backup_stochastic!(controller::InteractiveController{A, W}, controllers_j::Array{AbstractController, 1}; minval = 1e-10) where {A, W}
# 	debug = Set{Symbol}([])
# 	nodes = controller.nodes
# 	# observations = observations(controller_i.frame)
# 	#tentative from incpruning
# 	#prder of it -> actions, obs
# 	#for each a, z produce n new nodes (iterate on nodes)
# 	#for each node iterate on s and s' to produce a new node
# 	#new node is tied to old node?, action a and obs z
# 	#with stochastic pruning we get the cis needed
# 	new_nodes = full_backup_generate_nodes(controller, controllers_j)
# 	#before performing filtering with the old nodes update incomingEdge structure of old nodes
# 	#also assign permanent ids
# 	nodes_counter = controller.maxId+1
# 	for new_node in new_nodes
# 		@deb("Node $(new_node.id) becomes node $(nodes_counter)", :data)
# 		new_node.id = nodes_counter
# 		if :data in debug
# 			println(new_node)
# 		end
# 		nodes_counter+=1
# 		for (action, observation_map) in new_node.edges
# 			for (observation, edge_map) in observation_map
# 				#@deb("Obs $observation")
# 				for (next, prob) in edge_map
# 					#@deb("adding incoming edge from $(new_node.id) to $(next.id) ($action, $observation)")
# 					if haskey(next.incomingEdgeDicts, new_node)
# 						#@deb("it was the $(length(next.incomingEdgeDicts[new_node])+1)th")
# 						push!(next.incomingEdgeDicts[new_node], edge_map)
# 					else
# 						#@deb("it was the first edge for $(new_node.id)")
# 						next.incomingEdgeDicts[new_node] = [edge_map]
# 					end
# 				end
# 			end
# 		end
# 	end
# 	new_max_id = nodes_counter-1
# 	#add new nodes to controller
# 	#i want to have the new nodes first, so in case there are two nodes with identical value the newer node is pruned and we skip rewiring
# 	#no need to sort, just have new nodes examined before old nodes
# 	#the inner union is needed to have an orderedset as first parameter, as the result of union({ordered}, {not ordered}) = {ordered}
# 	orderedset = union(union(OrderedSet{Node}(), new_nodes), Set{Node}(oldnode for oldnode in values(nodes)))
# 	all_nodes = filterNodes(orderedset, minval)
# 	new_controller_nodes = Dict{Int64, Node{A, W}}()
# 	for node in all_nodes
# 		#add nodes to the controller
# 		checkNode(node, controller, minval; normalize = true, checkDistinct = false)
# 		new_controller_nodes[node.id] = node
# 	end
# 	controller.nodes = new_controller_nodes
# 	controller.maxId = new_max_id
# end

function full_backup_stochastic!(controller::InteractiveController{A, W}, controllers_j::Array{AbstractController, 1}) where {A, W}
	initial_node = controller.nodes[1]
	already_present_action = first(controller.nodes[1].actionProb)[1]
	@deb("Already present action: $already_present_action", :full)
	for a in actions(controller.frame)
		if a != already_present_action
			actionProb = Dict{A, Float64}(a => 1)
			edges = Dict{A, Dict{W, Vector{Pair{Int64, Float64}}}}( a => Dict{W, Vector{Pair{Int64, Float64}}}() )
			for z in observations(controller.frame)
				edges[a][z] = [(initial_node.id => 1)]
			end
			new_node = Node(length(controller.nodes)+1, actionProb, edges, Array{Float64, 2}(undef, 0, 0))
			@deb("Adding node", :full)
			@deb(new_node, :full)
			push!(controller.nodes, new_node)
		end
	end
end

#
# function full_backup_generate_nodes(controller::InteractiveController{A, W}, controllers_j::Array{AbstractController, 1}) where {A, W}
# 	@deb("Generating nodes", :full)
# 	minval = config.minval
# 	frame = controller.frame
# 	nodes = controller.nodes
# 	n_nodes = length(nodes)
# 	#@deb(n_nodes)
# 	n_controllers_j = length(controllers_j)
# 	states = IPOMDPs.states(frame)
# 	n_states = length(states)
# 	actions_i = actions(frame)
# 	n_actions = length(actions_i)
# 	observations_i = observations(frame)
# 	n_observations = length(observations_i)
# 	#tentative from incpruning
# 	#prder of it -> actions, obs
# 	#for each a, z produce n new nodes (iterate on nodes)
# 	#for each node iterate on s and s' to produce a new node
# 	#new node is tied to old node?, action a and obs z
# 	#with stochastic pruning we get the cis needed
# 	temp_id_j = Array{Array{Int64, 1}, 1}(undef, n_controllers_j)
# 	node_counter = 1
# 	for controller_index in 1:n_controllers_j
# 		controller_j = controllers_j[controller_index]
# 		nodes_j = controller_j.nodes
# 		#initialize inner array
# 		temp_id_j[controller_index] = Array{Int64, 1}(undef, length(controller_j.nodes))
# 		#quick fix to have the values in some order
# 	    for node_id in sort(collect(keys(nodes_j)))
# 	        temp_id_j[controller_index][node_id] = node_counter
# 			node_counter += 1
# 	    end
# 	end
# 	n_nodes_j = node_counter-1
# 	@deb("total nodes in j: $n_nodes_j", :multiple)
#
# 	new_nodes = Set{Node}()
# 	#new nodes counter used mainly for debugging, counts backwards (gets overwritten eventually)
# 	new_nodes_counter = -1
# 	for ai in actions_i
# 		#this data structure has the set of nodes for each observation (new_nodes_z[obs] = Set{Nodes} generated from obs)
# 		new_nodes_z = Vector{Set{Node}}(undef, length(observations_i))
# 		for zi_index in 1:length(observations_i)
# 			zi = observations_i[zi_index]
# 			#this set contains all new nodes for action, obs for all nodes
# 			new_nodes_a_z = Set{Node}()
# 			for (ni_id, ni) in nodes
# 				new_v = node_value(ni, ai, zi, controllers_j, frame, temp_id_j)
# 				#do not set node id for now
# 				#new_node = build_node(new_nodes_counter, [a], [1.0], [[obs]], [[1.0]], [[node]], new_v)
# 				new_node = build_node(new_nodes_counter, ai, zi, ni, new_v)
# 				push!(new_nodes_a_z, new_node)
# 				new_nodes_counter -=1
# 			end
# 			# if :escape in debug
# 			# 	println("New nodes created:")
# 			# 	for node in new_nodes_a_z
# 			# 		println(node)
# 			# 	end
# 			# end
# 			@deb("$(length(new_nodes_a_z)) nodes created for observation $(observations_i[zi_index])", :full)
# 			new_nodes_z[zi_index] = filterNodes(new_nodes_a_z, minval)
# 			@deb("$(length(new_nodes_z[zi_index])) nodes added after filtering for observation $(observations_i[zi_index])", :full)
# 		end
# 		#set that contains all nodes generated from action a after incremental pruning
# 		if :full in debug
# 			println("new nodes counter = $new_nodes_counter")
# 			println("calling incprune for action $ai")
# 		end
# 		new_nodes_counter, new_nodes_a = incprune(new_nodes_z, new_nodes_counter, minval)
# 		union!(new_nodes, new_nodes_a)
# 	end
# 	#all new nodes, final filtering
# 	filtered_nodes = filterNodes(new_nodes, minval)
# 	for new_node in new_nodes
# 		for (a, a_dict) in new_node.edges
# 			if length(a_dict) != n_observations
# 				error("Invalid node produced!")
# 			end
# 		end
# 	end
# 	return filtered_nodes
# end
#using eq τ(n, a, z) from incremental pruning paper
function node_value(ni::Node{A, W}, ai::A, zi::W, controllers_j::Array{AbstractController, 1}, ipomdp::IPOMDP{S, A, W}, temp_id_j::Array{Array{Int64, 1}, 1}) where {S, A, W}
	n_controllers_j = length(controllers_j)
	states = IPOMDPs.states(ipomdp)
	n_states = length(states)
	n_observations = length(observations_agent(ipomdp))
	γ = IPOMDPs.discount(ipomdp)
	tot_nj = 0
	for inner in temp_id_j
		tot_nj+= length(inner)
	end
	new_V = zeros(Float64, n_states, tot_nj )
	for s_index in 1:n_states
		s = states[s_index]
		for controller_index in 1:n_controllers_j
			controller_j = controllers_j[controller_index]
			frame_j = controller_j.frame
			nodes_j = controller_j.nodes
			for nj in nodes_j
				nj_id = nj.id
				immediate_reward = 0.0
				future_reward = 0.0
				for (aj, aj_prob) in nj.actionProb
					immediate_reward += aj_prob * IPOMDPs.reward(ipomdp, s, ai, aj)
					for s_prime_index in 1:n_states
						s_prime = states[s_prime_index]
						transition_i =POMDPModelTools.pdf(IPOMDPs.transition(ipomdp,s,ai, aj), s_prime)
						observation_i = POMDPModelTools.pdf(IPOMDPs.observation(ipomdp, s_prime, ai, aj), zi)
						if transition_i != 0.0 && observation_i != 0.0
							part_mult_s_prime = aj_prob * transition_i * observation_i
							for (zj, obs_dict_j) in nj.edges[aj]
								observation_j = POMDPModelTools.pdf(observation(frame_j, s_prime, aj, ai), zj)
								if observation_j != 0.0
									part_mult_zj = part_mult_s_prime * observation_j
									for (nj_prime_id, prob_nj_prime) in obs_dict_j
										#@deb("$action, $state, $observation, $s_prime")
										#@deb("$(node.value[s_prime_index]) * $(p_obs) * $(p_s_prime)")
										future_reward += ni.value[s_prime_index, temp_id_j[controller_index][nj_prime_id]] * part_mult_zj * prob_nj_prime
									end
								end
							end
						end
					end
				end
				new_V[s_index, temp_id_j[controller_index][nj_id]] = (1/n_observations) * immediate_reward + γ* future_reward
			end
		end
	end
	return new_V
end


function escape_optima_standard!(controller::InteractiveController{A, W}, controllers_j::Array{AbstractController,1}, tangent_b::Dict{Int64, Array{Float64}}; add_one = false) where {A, W}
	@deb("Entered escape_optima", :flow)
	#start_time(controller.stats, "escape")
	frame_i = controller.frame
	nodes = controller.nodes
	n_nodes = length(nodes)
	#@deb(n_nodes)
	n_controllers_j = length(controllers_j)
	states = IPOMDPs.states(frame_i)
	n_states = length(states)
	actions_i = actions(frame_i)
	n_actions = length(actions_i)
	observations_i = observations(frame_i)
	n_observations = length(observations_i)

	if length(tangent_b) == 0
		error("tangent_b was empty!")
	end


	temp_id_j = Array{Array{Int64, 1}, 1}(undef, n_controllers_j)
	node_counter = 1
	for controller_index in 1:n_controllers_j
		controller_j = controllers_j[controller_index]
		nodes_j = controller_j.nodes
		#initialize inner array
		temp_id_j[controller_index] = Array{Int64, 1}(undef, length(controller_j.nodes))
		#quick fix to have the values in some order
		for node_id in sort(collect(keys(nodes_j)))
			temp_id_j[controller_index][node_id] = node_counter
			node_counter += 1
		end
	end
	n_nodes_j = node_counter-1
	@deb("total nodes in j: $n_nodes_j", :multiple)



	#new_nodes = full_backup_generate_nodes(controller, controller_j, minval)
	#@deb("Finished generating nodes", :escape)

	# if :escape in debug
	# 	println("new_nodes:")
	# 	for node in new_nodes
	# 		println(node)
	# 	end
	# end



	escaped = false
	reachable_beliefs = Set{Array{Float64}}()
	for (id, start_b) in tangent_b
		#id = collect(keys(tangent_b))[1]
		#start_b = tangent_b[id]
		@deb(start_b)
		@deb("$id - >$start_b", :belief)
		for ai in keys(nodes[id].actionProb)
			for zi in observations_i
				new_b = belief_update(start_b, ai, zi, controller, controllers_j)
				#node = generate_node_directly(controller, controller_j, new_b)
				@deb("from belief $start_b action $ai and obs $zi -> $new_b", :belief)
				if add_one
					escaped = escaped || add_escape_node!(new_b, controller, controllers_j, temp_id_j)
					##stop_time(controller.stats, "escape")

				else
					push!(reachable_beliefs, new_b)
				end
			end
		end
		#break here if you want to improve only the first tangent belief point
		if add_one && escaped
			break
		end
	end
	#by accumulating reachable beliefs into a set duplicates are eliminated = less computation
	if !add_one
		for reachable_b in reachable_beliefs
			escaped = escaped || add_escape_node!(reachable_b, controller, controllers_j, temp_id_j)
		end
	end
	#@deb("$reachable_b")
	#stop_time(controller.stats, "escape")
	return escaped
end

function add_escape_node!(new_b::Array{Float64}, controller::InteractiveController{S, A, W}, controllers_j::Array{AbstractController, 1}, temp_id_j::Array{Array{Int64, 1}, 1}) where {S, A, W}
	best_old_node, best_old_value = get_best_node(new_b, controller.nodes)
	#@assert best_old_node_alt == best_old_node
	if :escape in debug
		println("Best old node:")
		println(best_old_node)
	end

	best_new_node, best_new_value = generate_node_directly(controller, controllers_j, new_b, temp_id_j)
	if best_new_value - best_old_value > config.min_improvement
		@deb("in $new_b node $(best_new_node.id) has $best_new_value > $best_old_value", :escape)
		#reworked_node = rework_node(controller, best_new_node)
		#controller.nodes[reworked_node.id] = reworked_node
		@deb("Added node $(best_new_node.id) to improve belief $new_b", :flow)
		@deb("Improvement $(best_new_value-best_old_value)", :flow)

		checkNode(best_new_node, controller; normalize = config.normalize)
		push!(controller.nodes, best_new_node)

		@deb(controller.nodes[best_new_node.id], :flow)
		return true
	end
	return false
end

function belief_update(start_b::Array{Float64}, ai::A, zi::W, controller::InteractiveController, controllers_j::Array{AbstractController, 1}) where {A, W}
	##start_time(controller.stats, "escape_belief_update")
	frame_i = controller.frame
	n_controllers_j = length(controllers_j)
	states = IPOMDPs.states(frame_i)
	n_states = length(states)
	actions_i = actions(frame_i)
	n_actions = length(actions_i)
	observations_i = observations(frame_i)
	n_observations = length(observations_i)

	temp_id_j = Array{Array{Int64, 1}, 1}(undef, n_controllers_j)
	node_counter = 1
	for controller_index in 1:n_controllers_j
		controller_j = controllers_j[controller_index]
		nodes_j = controller_j.nodes
		#initialize inner array
		temp_id_j[controller_index] = Array{Int64, 1}(undef, length(controller_j.nodes))
		#quick fix to have the values in some order
		for node_id in sort(collect(keys(nodes_j)))
			temp_id_j[controller_index][node_id] = node_counter
			node_counter += 1
		end
	end
	n_nodes_j = node_counter-1
	@deb("total nodes in j: $n_nodes_j", :multiple)
	new_b = zeros(n_states, n_nodes_j)
	normalize = 0.0
	for s_prime_index in 1:n_states
		s_prime = states[s_prime_index]
		for s_index in 1:n_states
			s = states[s_index]
			for controller_index in 1:n_controllers_j
				controller_j = controllers_j[controller_index]
				nodes_j = controller_j.nodes
				frame_j = controller_j.frame
				for nj in nodes_j
					nj_id = nj.id
					@deb("$(start_b[s_index, temp_id_j[nj_id]])")
					if start_b[s_index, temp_id_j[controller_index][nj_id]] == 0.0
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
							observation_j = POMDPModelTools.pdf(observation(frame_j, s_prime, aj, ai), zj)
							if observation_j == 0.0
								continue
							end
							@deb("\t\t $observation_j")
							for (n_prime_j_id, prob_j) in obs_dict
								@deb("start_b = $(start_b[s_index, temp_id_j[nj_id]])")
								@deb("adding $(start_b[s_index, temp_id_j[nj_id]]) * $aj_prob * $transition_i* $observation_i * $observation_j * $prob_j")
								@deb("adding $(start_b[s_index, temp_id_j[nj_id]] * aj_prob * transition_i* observation_i * observation_j * prob_j)")
								new_b[s_prime_index, temp_id_j[controller_index][n_prime_j_id]] += start_b[s_index, temp_id_j[controller_index][nj_id]] * aj_prob * transition_i* observation_i * observation_j * prob_j
								normalize += start_b[s_index, temp_id_j[controller_index][nj_id]] * aj_prob * transition_i* observation_i * observation_j * prob_j
							end
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
	#stop_time(controller.stats, "escape_belief_update")
	return new_b
end

# function rework_node(controller::AbstractController, new_node::Node{A, W}) where {A, W}
# 		id = controller.maxId+1
# 		actionProb = copy(new_node.actionProb)
# 		value = copy(new_node.value)
# 		edges = Dict{A, Dict{W, Vector{Pair{Int64, Float64}}}}()
# 		for (a, obs_dict) in new_node.edges
# 			edges[a] = Dict{W, Vector{Pair{Int64, Float64}}}()
# 			for (z, node_dict) in obs_dict
# 				edges[a][z] = Dict{Int64,Float64}()
# 				for (node, prob) in node_dict
# 					current_controller_node = controller.nodes[node.id]
# 					edges[a][z][current_controller_node] = prob
# 				end
# 			end
# 		end
# 		return Node(id, actionProb,edges, value, Dict{Int64, Vector{Vector{Pair{Int64, Float64}}}}())
# end



function generate_node_directly(controller::InteractiveController{A, W}, controllers_j::Array{AbstractController, 1}, start_b::Array{Float64}, temp_id_j::Array{Array{Int64,1},1}) where {A, W, S}
	#start_time(controller.stats, "escape_generate_node")
	frame_i = controller.frame
	actions_i = actions(frame_i)
	observations_i = observations(frame_i)
	n_observations = length(observations_i)
	# states = states(frame_i)
	# n_states = length(states)
	best_node = nothing
	best_value = 0.0
	for a in actions_i
		#try all actions
		new_node = nothing
		for z_index in 1:n_observations
			#find the best edge (aka the best next node) for each observation
			z = observations_i[z_index]
			#compute the result belief of executing action a and receiving obs z starting from belief b.
			result_b = belief_update(start_b,a,z, controller, controllers_j)
			#get the best node in the controller for the updated beief
			best_next_node, best_value_obs = get_best_node(result_b, controller.nodes)
			new_v = node_value(best_next_node, a, z, controllers_j, frame_i, temp_id_j)
			@deb("new_v = $new_v", :generatenode)
			new_node_partial = build_node(length(controller.nodes)+1, a, z, best_next_node, new_v)
			@deb("new partial node:", :generatenode)
			@deb(new_node_partial, :generatenode)
			#add that edge to the node and update the value vector
			if z_index ==1
				new_node = new_node_partial
			else
				new_node = mergeNode(new_node, new_node_partial, a, length(controller.nodes)+1)
				@deb("after merge:", :generatenode)
				@deb(new_node, :generatenode)
			end
		end
		#compute the best node (choose between actions)
		new_value = sum(start_b[i]*new_node.value[i] for i in 1:length(start_b))
		if best_node == nothing || best_value < new_value
			best_node = new_node
			best_value = new_value
		end
	end
	#stop_time(controller.stats, "escape_generate_node")
	return best_node, best_value
end
