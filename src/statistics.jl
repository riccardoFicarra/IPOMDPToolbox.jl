mutable struct stats

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

stats() = stats(0,0,0,0,0,0,0,0,0)

function computestats(stats::stats, ai::A, aj::A, state::S, s_prime::S, zi::W, zj::W) where {S, A, W}
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
            if (s_prime == :TL && zi == :GLS) || (s_prime == :TR && zi == :GRS)
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
    return stats
end


function average_listens(stats::stats)
    avg_l = stats.listen / (stats.correct + stats.wrong)
    println("Average listens per opening: $avg_l")

end

function average_correct_obs(stats::stats)
    avg_l = stats.correct_z_l / (stats.wrong_z_l + stats.correct_z_l)
    avg_or = stats.correct_z_or / (stats.wrong_z_or + stats.correct_z_or)
    avg_ol = stats.correct_z_ol / (stats.wrong_z_ol + stats.correct_z_ol)
    println("avg_l: $avg_l")
    println("avg_or: $avg_or")
    println("avg_ol: $avg_ol")
end

mutable struct time_statistics
    timers::Dict{String, Float64}
    start_times::Dict{String, Float64}
end

global time_stats = time_statistics(Dict{String, Float64}(), Dict{String, Float64}())

function start_time(name::String)
    global time_stats.start_times[name] = datetime2unix(now())
    @deb("Set start time $(time_stats.start_times[name]) for $name", :time)
end

function stop_time(name::String)
    end_time = datetime2unix(now())
    @deb("End time $(end_time) for $name", :time)
    @deb("Difference: $(end_time - time_stats.start_times[name])", :time)
    if !haskey(time_stats.start_times, name)
        error("Timer $name has not been set")
    end
    if time_stats.start_times[name] == 0.0
        error("start_time has been used more than one time")
    end
    if haskey(time_stats.timers, name)
        global time_stats.timers[name] += end_time - time_stats.start_times[name]
    else
        global time_stats.timers[name] = end_time - time_stats.start_times[name]
    end
    global time_stats.start_times[name] = 0.0
end

function reset_time(name::String)
    global time_stats.timers[name] = 0.0
end

function reset_timers()
    @deb("Reset all timers", :time)
    global time_stats = time_statistics(Dict{String, Float64}(), Dict{String, Float64}())
end

function print_time_stats()
    for (name, time) in time_stats.timers
        println("$name : $time")
    end
end
