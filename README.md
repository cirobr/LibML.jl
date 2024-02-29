# LibML.jl

[![Build Status](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/cirobr/LibML.jl/actions/workflows/CI.yml?query=branch%3Amain)

Machine Learning extension functions to Flux.jl:
* trainModel!(model, data, optstate, lossfn; verbose=false)
* evaluateModel(model, data, lossfns)
* saveModelState(filename, model)
* loadModelState(filename, modelcpu)
* saveModelStateCB(path, model)
* countParams(model)
* IoU(yhat, y)
* IoU_loss(yhat, y)
* softKLDiv(yhat, y; T::Float32=1.0f0, dims::Int=1)
* IoUScore(yhat, y; threshold=0.5)
* AccScore(yhat, y; threshold=0.5)
* F1Score(yhat, y; threshold=0.5)
* PrecisionScore(yhat, y; threshold=0.5)
* RecallScore(yhat, y; threshold=0.5)
* FPRScore(yhat, y; threshold=0.5)
* TNRScore(yhat, y; threshold=0.5)


### v0.3.11
* Added new function countParams(model).


### v0.3.10
* Added T as temperature argument in softKLDiv().

### v0.3.9
* IoU_loss() full Float32 calculation.
* IoUScore() based on confusion matrix.


### v0.3.8
Added new loss function, for knowledge distillation projects
* softKLDiv()


### v0.3.7
Added aliases:
* trainEpoch!   = alias to trainModel!
* evaluateEpoch = alias to evaluateModel! and testModel


### v0.3.6
Removed classification_metrics(), replaced by individual score functions with thresholds for yhat, as follows:
* IoUScore
* AccScore
* F1Score
* PrecisionScore
* RecallScore
* FPRScore
* TNRScore


### v0.3.5
* Added classification_metrics()


### v0.3.4
* First documented version
* Changed modelstate.jl dependency from BSON to JLD2
* Extended testing for modelstate.jl
* Added trainModel!()
* Added progress bars to both trainModel!() and testModel()
