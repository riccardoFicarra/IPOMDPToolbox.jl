#=
ibpisolver.jl:
- Julia version: 1.1.0
- Author: fiki9
- Date: 2019-02-11
=#
	using POMDPs
	using IPOMDPs
	using IPOMDPToolbox
	using Dates
	using Serialization

	"""
	Abstract type used for any controller, interactive or not.
	"""
	abstract type AbstractController end


	function frametype(controller::AbstractController)
		return split("$(typeof(controller.frame))", ".")[3]
	end

	"""
	Snippet to have debug utility. Use @deb(String) to print debug info
	"""
	global debug = Set{Symbol}()

	macro deb(str)
		:( :data in debug && println($(esc(str))) )
	end
	macro deb(str, type)
		:( $type in debug && println($(esc(str)))  )
	end

	include("statistics.jl")
	include("bpipolicyutils.jl")
	include("ibpi.jl")

	"""
	Structure used for global parameters.


	"""
	struct IBPISolver
		# Here should go some settings
		force::Int64
		maxrep::Int64
		minval::Float64
		timeout::Int64
		min_improvement::Float64
	end
	"""
	Default config values
	"""
	config = IBPISolver(0, 10, 1e-10, 300, 1e-10)


	"""
	functions to overwrite default values
	# Arguments
	- `force::Int64`: force all controllers to be initialized with a node with the same action.
	- `maxrep::Int64`: maximum number of iterations after which IBPI stops
	- `minval::Float64`: if a number is below minval it is considered as zero.
	- `timeout::Int64`: number of seconds after which the algorithm stops.
	"""
	function set_solver_params(force::Int64, maxrep::Int64, minval::Float64, timeout::Int64, min_improvement::Float64)
		global config = IBPISolver(force, maxrep, minval, timeout, min_improvement)
	end

	function Base.println(controller::AbstractController)
		for node in controller.nodes
			println(node)
		end
		println("$(length(controller.nodes)) nodes")
	end


	struct IBPIPolicy{S, A, W}
		#Controller level -> controller frames
		#level 1 is the pompdp
		controllers::Array{Array{AbstractController,1}}
		maxlevel::Int64
	end

	"""
	maxlevelframe is the frame of the agent we want to build
	add as many frame arrays as levels, in descending order.
	"""
	function IBPIPolicy(maxlevelframe::IPOMDP{S, A, W}, emulated_frames...; force = 0) where {S, A, W}
		#i dont consider maxlevelframe because we're starting from lv0
		maxlevel = length(emulated_frames)+1
		@deb("maxlevel = $maxlevel", :multiple)
		controllers = Array{Array{AbstractController, 1}}(undef, maxlevel)
		max_level_controller = InteractiveController(maxlevel, maxlevelframe; force = force)
		controllers[maxlevel] = [max_level_controller]
		#emulated_frames is a tuple{vector{IPOMDP}}
		for i in 1:maxlevel-2
			controllers[maxlevel-i] = [InteractiveController(maxlevel-i, emulated_frames[i][frame]; force = force) for frame in 1:length(emulated_frames[i])]
		end
		#pomdp part, level 1
		controllers[1] = [Controller( emulated_frames[maxlevel-1][frame]; force = force) for frame in 1:length(emulated_frames[maxlevel-1])]
		return IBPIPolicy{S, A, W}(controllers, maxlevel)
	end

	function Base.println(policy::IBPIPolicy)
		for l in 1:policy.maxlevel
			println("Level $l")
			for frame_index in 1:length(policy.controllers[l])
				println("Frame $frame_index:  $(frametype(policy.controllers[l][frame_index]))")
				println(policy.controllers[l][frame_index])
			end
		end
	end

	"""
		Return the policy type used by the solver. Since ReductionSolver is an online solver, the policy doesn't really exist.
		It is used as a container to maintain data through time
		solve(solver::ReductionSolver, ipomdp::IPOMDP{S,A,W})
	Return:
		ReductionPolicy{S,A,W}
	"""
	function IPOMDPs.solve(solver::IBPISolver, ipomdp::IPOMDP{S,A,W}) where {S,A,W}

		return IBPIPolicy(ipomdp)
	end

	function eval_and_improve!(policy::IBPIPolicy, level::Int64)
		@deb("called @level $level", :flow)
		improved = false
		if level >= 2
			improved_lower = eval_and_improve!(policy, level-1)
		end
		if level == 1
			tangent_b  = Dict{Int64, Array{Float64}}()
			for controller in policy.controllers[1]
				#experimental: if controller of level 0 has converged skip it to avoid losing time
				if !controller.converged

					println("Level 1, frame $(frametype(controller)): $(length(controller.nodes)) nodes")
					evaluate!(controller)
					@deb(controller, :data)
					start_time = datetime2unix(now())

					improved, tangent_b  = partial_backup!(controller ; add_one = true)
					@deb("Elapsed time for level $level: $(datetime2unix(now()) - start_time)",:stats)
					if improved
						@deb("Improved level 1", :flow)
						@deb(policy.controllers[1], :data)
					else
						@deb("Did not improve level 1", :flow)
						escaped = escape_optima_standard!(controller, tangent_b; add_one = false, minval = 1e-10)
						improved == improved || escaped
						#if the controller failed both improvement and escape mark it as converged.
						controller.converged = !escaped
					end
				else
					println("Level 1, frame $(frametype(controller)): $(length(controller.nodes)) nodes has converged")
				end
			end
		else
			for controller in policy.controllers[level]
				if !controller.converged
					println("Level $level, frame $(frametype(controller)) : $(length(controller.nodes)) nodes")
					evaluate!(controller, policy.controllers[level-1])

					@deb(controller, :data)
					start_time = datetime2unix(now())

					improved, tangent_b = partial_backup!(controller, policy.controllers[level-1]; add_one = true)

					@deb("Elapsed time for level $level: $(datetime2unix(now()) - start_time)", :stats)

					if improved
						@deb("Improved level $level", :flow)
						@deb(controller, :data)
					else
						@deb("Did not improve level $level", :flow)
						escaped = escape_optima_standard!(controller, policy.controllers[level-1], tangent_b ;add_one = false, minval = config.minval)
						improved = improved || escaped
					end
					#a controller has converged only if also all lower level controllers have converged
					controller.converged = !improved && !improved_lower
					if controller.converged
						println("Level $level, frame $(frametype(controller)) : $(length(controller.nodes)) nodes")
					end
				else
					println("Level $level, frame $(frametype(controller)): $(length(controller.nodes)) nodes has converged")
				end
			end
		end
		return improved
	end

	function ibpi!(policy::IBPIPolicy)
		#full backup part to speed up
		for controller in policy.controllers[1]
			evaluate!(controller)
			if length(controller.nodes) <= 1
				full_backup_stochastic!(controller)
				@deb("Level	1 after full backup", :flow)
				@deb(controller, :flow)
			end
		end

		for level in 2:policy.maxlevel
			for controller in policy.controllers[level]
				evaluate!(controller, policy.controllers[level-1])
				checkController(controller, config.minval)
				if length(controller.nodes) <= 1
					full_backup_stochastic!(controller, policy.controllers[level-1]; minval = config.minval)
					@deb("Level $level after full backup", :flow)
					@deb(controller, :flow)
				end
			end
		end
		start_time = datetime2unix(now())

		#start of the actual algorithm

		iteration = 1
		while true
			@deb("Iteration $iteration", :flow)
			improved = eval_and_improve!(policy, policy.maxlevel)
			iteration += 1

			if !improved
				println("Algorithm stopped because it could not improve controllers anymore")
				break
			elseif config.maxrep >= 0 && iteration >= config.maxrep
				println("maxrep exceeded $iteration")
				break
			elseif	datetime2unix(now()) >= start_time+config.timeout
				println("timeout exceeded")
				break
			end
		end
		#only needed when evaluation is cut short
		#but it doesnt take that much time
		for level in 1:policy.maxlevel
			for controller in policy.controllers[level]
				if level == 1
					evaluate!(controller)
				else
					evaluate!(controller, policy.controllers[level-1])
				end
			end
		end
	end

	function save_policy(policy::IBPIPolicy, name::String)
		#@save "savedcontrollers/$name.jld2" policy
		serialize("savedcontrollers/$name.policy", policy)
	end

	function load_policy(name::String)
		#@load "savedcontrollers/$name.jld2" policy
		policy = deserialize("savedcontrollers/$name.policy")
		return policy
	end

	function solve_fresh!(policy::IBPIPolicy{S, A, W}, n_steps::Int64, step_length::Int64, maxsimsteps::Int64, min_improvement::Float64 ; save = "", force = 3, max_iterations = -1) where {S, A, W}

		for step in 1:n_steps
			filename_dst = "$(save)_$(step*step_length)"
			set_solver_params(force,max_iterations,1e-10,step_length*60, min_improvement)

			ibpi!(policy)

			if save != ""
				save_policy(policy, filename_dst)
			end
		end
	end

	function print_solver_stats(policy::IBPIPolicy)
		for level in 1:policy.maxlevel
			println("Level $level")
			for controller in policy.controllers[level]
				println("\tFrame $(frametype(controller))")
				print_solver_stats(controller.stats)
			end
		end
	end

	function continue_solving(src_filename::String, n_steps::Int64, step_length::Int64, maxsimsteps::Int64, min_improvement::Float64; force = 3, max_iterations = -1)
		policy = load_policy(src_filename)
		name = split(src_filename, "_")[1]
		src_duration = parse(Int64, split(src_filename, "_")[2])
		for step in 1:n_steps
			filename_dst = "$(name)_$(src_duration+step*step_length)"
			set_solver_params(force,max_iterations,1e-10,step_length*60, min_improvement)

			ibpi!(policy)
			save_policy(policy, filename_dst)
		end
		return policy
	end

	include("./simulator.jl")
