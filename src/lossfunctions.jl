function IoU(yhat::AbstractArray, y::AbstractArray)
    yc = clamp.(yhat, 0.0, 1.0)
    i = yc .* y
    u = yc .+ y .- i
    return sum(i) / (sum(u) + eps(Float32)) |> Float32
end

IoU_loss(yhat::AbstractArray, y::AbstractArray) = 1.0 - IoU(yhat, y) |> Float32


tv07_loss(yhat::AbstractArray, y::AbstractArray) = Flux.tversky_loss(yhat, y, beta=0.7) |> Float32

tv03_loss(yhat::AbstractArray, y::AbstractArray) = Flux.tversky_loss(yhat, y, beta=0.3) |> Float32
