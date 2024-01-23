# LibML.jl

[![Build Status](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml?query=branch%3Amain)

Machine Learning handy functions in Julia/Flux:
* trainModel!(model, data, optstate, lossfn; verbose=false)
* saveModelState(filename, model)
* loadModelState(filename, modelcpu)
* saveModelStateCB(path, model)
* IoU(yhat, y)
* IoU_loss(yhat, y)


### v0.3.4
* First documented version
* Changed modelstate.jl dependency from BSON to JLD2
* Extended testing for modelstate.jl
* Added trainModel!()
* Added progress bars to both trainModel!() and testModel()
