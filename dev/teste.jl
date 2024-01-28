using LibML
using Random
using Flux
# using CUDA   # not added to LibML environment

Random.seed!(1234)
yhat=rand(Float32, (10,10))
y=rand(Bool, (10,10))
score = LibML.IoUScore(yhat, y; threshold=0.5)

yhat_d = gpu(yhat)
y_d = gpu(y)
score_d = LibML.IoUScore(yhat_d, y_d; threshold=0.5)

score, score_d
