Random.seed!(1)
yhat = rand(Bool, 5,5)
y    = rand(Bool, 5,5)

@test LibML.IoU(yhat, y) == 0.42105263f0 || error("IoU failed")
@test LibML.IoU_loss(yhat, y) == 0.57894737f0 || error("IoU_loss failed")
@test LibML.classification_metrics(yhat, y) == (0.56f0, 0.5925926f0)
@test LibML.classification_metrics(yhat, y, verbosity=true) == (0.56f0, 0.5925926f0, 0.61538464f0, 0.5714286f0, 0.45454547f0, 0.54545456f0)
