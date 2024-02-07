"""
IoU(yhat, y)
Implements intersection-over-union.
Arguments yhat and y must only have elements between [0:1]. Function does not check for that.
"""
function IoU(yhat::AbstractArray, y::AbstractArray)
    i = yhat .* y
    u = yhat .+ y .- i
    return sum(i) / (sum(u) + eps(Float32)) |> Float32
end



"""
IoU_loss(yhat, y)
Implements the loss function for intersection-over-union.
Arguments yhat and y must only have elements between [0:1]. Function does not check for that.
"""
IoU_loss(yhat::AbstractArray, y::AbstractArray) = 1.0 - IoU(yhat, y) |> Float32



"""
softKLDiv(yhat, y; dims=1)
Implements the soft Kullback-Leibler divergence.
Both arguments yhat and y are transformed to probability distributions using softmax.
Argument dims is the dimension along which the softmax is applied. Default is 1.
"""
function softKLDiv(yhat::AbstractArray, y::AbstractArray; dims::Int=1)
    y2 = Flux.softmax(yhat, dims=dims)
    y1 = Flux.softmax(y,    dims=dims)
    return Flux.kldivergence(y2, y1) |> Float32
end
