function IoU(yhat::AbstractArray, y::AbstractArray)
    yc = clamp.(yhat, 0.f0, 1.f0)

    intersection = sum(yc .* y)
    union        = sum(yc .+ y) - intersection
    iou = intersection / (union + eps(Float32))

    return iou |> Float32
end


IoU_loss(yhat::AbstractArray, y::AbstractArray) = 1.0f0 - IoU(yhat, y)
