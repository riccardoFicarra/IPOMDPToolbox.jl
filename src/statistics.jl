#agent stats part

mutable struct agent_stats

	#agent_i
	correct::Int64
	wrong::Int64
	listen::Int64

	correct_z_l::Int64
	correct_z_ol::Int64
	correct_z_or::Int64

	wrong_z_l::Int64
	wrong_z_ol::Int64
	wrong_z_or::Int64
end

agent_stats() = agent_stats(0,0,0,0,0,0,0,0,0)

function Base.println(stats::agent_stats)
	println("Correct doors opened: $(stats.correct)")
	println("Wrong doors opened: $(stats.wrong)")
	println("Listened: $(stats.listen)")
end

function computestats!(stats::agent_stats, ai::A, aj::A, state::S, s_prime::S, zi::W, zj::W) where {S, A, W}
	#action stats
	if ai != :L
		if (ai == :OL && state == :TR) || (ai == :OR && state == :TL)
			stats.correct += 1
		else
			stats.wrong += 1
		end
	else
		stats.listen += 1
	end

	#observation stats
	if ai == :L
		if aj == :L
			if (s_prime == :TL && (zi == :GLS || zi == :GL)) || (s_prime == :TR && (zi == :GRS || zi == :GR))
				stats.correct_z_l += 1
			else
				stats.wrong_z_l += 1
			end
		elseif aj == :OR
			if (s_prime == :TL && zi == :GLCR) || (s_prime == :TR && zi == :GRCR)
				stats.correct_z_or += 1
			else
				stats.wrong_z_or += 1
			end
		elseif aj == :OL
			if (s_prime == :TL && zi == :GLCL) || (s_prime == :TR && zi == :GRCL)
				stats.correct_z_ol += 1
			else
				stats.wrong_z_ol += 1
			end
		end
	end
end


function average_listens(stats::agent_stats)
	return stats.listen / (stats.correct + stats.wrong)
end

function average_correct_obs(stats::agent_stats)
	avg_l = stats.correct_z_l / (stats.wrong_z_l + stats.correct_z_l)
	avg_or = stats.correct_z_or / (stats.wrong_z_or + stats.correct_z_or)
	avg_ol = stats.correct_z_ol / (stats.wrong_z_ol + stats.correct_z_ol)
	println("avg_l: $avg_l")
	println("avg_or: $avg_or")
	println("avg_ol: $avg_ol")
end

mutable struct solver_statistics
	start_time::Float64
	#iteration -> (time, n_nodes)
	data::Array{Tuple{Float64, Int64, Int64}}
end

function solver_statistics()
	return solver_statistics(0, Array{Tuple{Float64, Int64, Int64}}(undef, 0))
end

function set_start_time(solver_statistics::solver_statistics, start_time::Float64)
	solver_statistics.start_time = start_time
end

function log_time_nodes(solver_statistics::solver_statistics, current_time::Float64, n_nodes::Int64, memory::Int64)
	time = current_time - solver_statistics.start_time
	push!(solver_statistics.data, (time, n_nodes, memory))
end


# function start_time(stats::solver_statistics, name::String)
# 	stats.start_times[name] = datetime2unix(now())
# 	@deb("Set start time $(stats.start_times[name]) for $name", :time)
# 	if haskey(stats.timers, name)
# 		while length(stats.timers[name]) < length(stats.n_nodes)
# 			@deb("$(length(stats.timers[name])) < $(length(stats.n_nodes))", :time)
# 			push!(stats.timers[name], 0.0)
# 		end
# 	else
# 		stats.timers[name] = zeros(length(stats.n_nodes))
# 	end
# end
#
# function log_n_nodes(stats::solver_statistics, n_nodes::Int64)
# 	push!(stats.n_nodes, n_nodes)
# end
#
# function stop_time(stats::solver_statistics,name::String)
# 	end_time = datetime2unix(now())
# 	t = length(stats.n_nodes)
# 	@deb("End time $(end_time) for $name", :time)
# 	@deb("Difference: $(end_time - stats.start_times[name])", :time)
# 	if !haskey(stats.start_times, name)
# 		error("Timer $name has not been set")
# 	end
# 	if stats.start_times[name] == 0.0
# 		error("start_time has been used more than one time")
# 	end
# 	if haskey(stats.timers, name)
# 		stats.timers[name][t] +=  end_time - stats.start_times[name]
# 	end
# 	stats.start_times[name] = 0.0
# end
#
# function reset_time(stats::solver_statistics,name::String)
# 	stats.timers[name] = 0.0
# end
#
# function reset_timers(stats::solver_statistics)
# 	@deb("Reset all timers", :time)
# 	stats = solver_statistics()
# end
#
# function print_solver_stats(stats::solver_statistics)
# 	names = sort(collect(keys(stats.timers)))
# 	println("$(stats.n_nodes)")
# 	for name in names
# 		println("$name\n$(stats.timers[name]) ")
# 	end
# end
