function countParams(model)
    modelcpu = cpu(model)
    return sum([length(v) for v in vec.(Flux.params(modelcpu))])
end


