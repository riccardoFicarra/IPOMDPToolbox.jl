include("ibpipolicyutils.jl")
IBPIPolicyUtils.debug[]=true
actions = [:a1, :a2, :a3]
observations = [:o1, :o2, :o3]

node = IBPIPolicyUtils.InitialNode(actions, observations)
next = IBPIPolicyUtils.getNextNode(node, IBPIPolicyUtils.getAction(node), :o1)
