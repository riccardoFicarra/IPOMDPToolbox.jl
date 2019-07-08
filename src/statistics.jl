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
    avg_l = stats.listen / (stats.correct + stats.wrong)
    println("Average listens per opening: $avg_l")

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
    n_nodes::Array{Int64, 1}
    timers::Dict{String, Array{Float64, 1}}
    start_times::Dict{String, Float64}
end

solver_statistics() = solver_statistics(Array{Int64, 1}(undef, 0), Dict{String, Array{Float64, 1}}(), Dict{String, Float64}())

function start_time(stats::solver_statistics, n_nodes::Int64, name::String)
    stats.start_times[name] = datetime2unix(now())
    @deb("Set start time $(stats.start_times[name]) for $name", :time)
    push!(stats.n_nodes, n_nodes)
end


function stop_time(stats::solver_statistics,name::String)
    end_time = datetime2unix(now())
    @deb("End time $(end_time) for $name", :time)
    @deb("Difference: $(end_time - stats.start_times[name])", :time)
    if !haskey(stats.start_times, name)
        error("Timer $name has not been set")
    end
    if stats.start_times[name] == 0.0
        error("start_time has been used more than one time")
    end
    if haskey(stats.timers, name)
        push!(stats.timers[name], end_time - stats.start_times[name])
    else
        stats.timers[name] = [ end_time - stats.start_times[name] ]
    end
    stats.start_times[name] = 0.0
end

function reset_time(stats::solver_statistics,name::String)
    stats.timers[name] = 0.0
end

function reset_timers(stats::solver_statistics)
    @deb("Reset all timers", :time)
    stats = solver_statistics()
end

function print_solver_stats(stats::solver_statistics)
    for name in sort(collect(keys(stats.timers)))
        println("$name : $(stats.timers[name])")
    end
end

function print_time_stats(stats::solver_statistics)
    for name in sort(collect(keys(local_stats.timers)))
        println("$name : $(local_stats.timers[name])")
    end
end
