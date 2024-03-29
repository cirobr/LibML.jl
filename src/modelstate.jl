function saveModelState(filename, model)
    modelcpu    = Flux.cpu(model)
    model_state = Flux.state(modelcpu)
    JLD2.jldsave(filename; model_state)
end


function loadModelState!(filename, modelcpu)
    model_state = JLD2.load(filename, "model_state")
    Flux.loadmodel!(modelcpu, model_state)
end


function saveModelStateCB(path, model)
    if path[end] != '/'   path = path * "/"   end
    fpfn = path * "model_state-" * Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS-sss") * ".jld2"
    saveModelState(fpfn, model)    
end
