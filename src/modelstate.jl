function saveModelState(fullpathFilename, model)
    modelcpu    = Flux.cpu(model)
    model_state = Flux.state(modelcpu)
    BSON.@save fullpathFilename model_state

    return
end


# modelcpu = Chain(...), needs to be defined before the function call
function loadModelState(fullpathFilename, modelcpu)
    BSON.@load fullpathFilename model_state
    Flux.loadmodel!(modelcpu, model_state)

    return
end


function saveModelStateCB(path, model)
    fpfn = path * "model_state-" * Dates.format(now(), "yyyy-mm-ddTHH-MM-SS-sss") * ".bson"
    saveModelState(fpfn, model)    
end
