function IoU(yhat::AbstractArray, y::AbstractArray)
    yc = clamp.(yhat, 0.0, 1.0)
    i = yc .* y
    u = yc .+ y .- i
    return sum(i) / (sum(u) + eps(Float32)) |> Float32
end

IoU_loss(yhat::AbstractArray, y::AbstractArray) = 1.0 - IoU(yhat, y) |> Float32


function classification_metrics(yhat::Array{Bool}, y::Array{Bool}; verbosity=false)
    # tp = true_positive(cm)
    # tn = true_negative(cm)
    # fp = false_positive(cm)
    # fn = false_negative(cm)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yhat, y)

    accuracy  = sm.accuracy(cm)                    # (tp + tn) / (tp + tn + fp + fn)
    precision = sm.positive_predictive_value(cm)   # tp / (tp + fp)
    recall    = sm.true_positive_rate(cm)          # tp / (tp + fn)
    fpr       = sm.false_positive_rate(cm)         # fp / (fp + tn)
    tnr       = sm.true_negative_rate(cm)          # tn / (tn + fp)
    f1        = sm.Functions.fscore(cm)            # 2 * (precision * recall) / (precision + recall)

    if verbosity   return (accuracy, f1, precision, recall, fpr, tnr) .|> Float32
    else           return (accuracy, f1) .|> Float32
    end
end
