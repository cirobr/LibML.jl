# LibML.jl

[![Build Status](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml?query=branch%3Amain)

Machine Learning handy functions in Julia/Flux:
* trainModel!(model, data, optstate, lossfn,
                     epochLosses::Vector{Float32}, stepLosses::Vector{Float32};
                     epochs::Int=1)
* testModel(model, data, lossfns::Vector{Any},
                   stepLosses::Vector{Float32})
* saveModelState(fullpathFilename, model)
* loadModelState(fullpathFilename, modelcpu)
* saveModelStateCB(path="./models/", model)
* IoU(yhat, y)
* IoU_loss(yhat, y)
