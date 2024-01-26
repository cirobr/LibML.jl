# LibML.jl

[![Build Status](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml?query=branch%3Amain)

Machine Learning extension functions to Flux.jl:
* trainModel!(model, data, optstate, lossfn; verbose=false)
* testModel(model, data, lossfns)
* saveModelState(filename, model)
* loadModelState(filename, modelcpu)
* saveModelStateCB(path, model)
* IoU(yhat, y)
* IoU_loss(yhat, y)
* classification_metrics(yhat::Array{Bool}, y::Array{Bool}; verbosity=false)


### v0.3.5
* Added classification_metrics()


### v0.3.4
* First documented version
* Changed modelstate.jl dependency from BSON to JLD2
* Extended testing for modelstate.jl
* Added trainModel!()
* Added progress bars to both trainModel!() and testModel()
