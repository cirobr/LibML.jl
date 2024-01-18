function saveModelState(fullpathFilename, model)
    modelcpu    = Flux.cpu(model)
    model_state = Flux.state(modelcpu)
    BSON.@save fullpathFilename model_state
end


# modelcpu = Chain(...), needs to be defined before the function call
function loadModelState(fullpathFilename, modelcpu)
    BSON.@load fullpathFilename model_state
    Flux.loadmodel!(modelcpu, model_state)
end


function saveModelStateCB(path, model)
    if path[end] != '/'   path = path * "/"   end
    fpfn = path * "model_state-" * Dates.format(Dates.now(), "yyyy-mm-ddTHH-MM-SS-sss") * ".bson"
    saveModelState(fpfn, model)    
end
