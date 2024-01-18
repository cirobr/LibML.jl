# LibML.jl

[![Build Status](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml?query=branch%3Amain)

Machine Learning handy functions in Julia/Flux:
* saveModelState(fullpathFilename, model)
* loadModelState(fullpathFilename, modelcpu)
* saveModelStateCB(path="./models/", model)
* IoU(yhat, y)
* IoU_loss(yhat, y)
