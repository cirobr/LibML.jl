function IoU(yhat::AbstractArray, y::AbstractArray)
    yc = clamp.(yhat, 0.0, 1.0)

    intersection = sum(yc .* y)
    union        = sum(yc .+ y) - intersection
    iou = intersection / (union + eps(Float32))

    return iou |> Float32
end

IoU_loss(yhat::AbstractArray, y::AbstractArray) = 1.0f0 - IoU(yhat, y)


tv07_loss(yhat::AbstractArray, y::AbstractArray) = Flux.tversky_loss(yhat, y, beta=0.7)
tv03_loss(yhat::AbstractArray, y::AbstractArray) = Flux.tversky_loss(yhat, y, beta=0.3)
