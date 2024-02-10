function IoUScore(yhat::AbstractArray, y::AbstractArray; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(cpu(yth), cpu(y))
    tp = sm.true_positive(cm)
    fp = sm.false_positive(cm)
    fn = sm.false_negative(cm)
    return tp / (tp + fp + fn + eps(Float32)) |> Float32
end


function AccScore(yhat::AbstractArray, y::AbstractArray; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(cpu(yth), cpu(y))
    return sm.accuracy(cm) |> Float32
end
const AccuracyScore = AccScore


function F1Score(yhat::AbstractArray, y::AbstractArray; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(cpu(yth), cpu(y))
    return sm.Functions.fscore(cm) |> Float32
end


function PrecisionScore(yhat::AbstractArray, y::AbstractArray; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(cpu(yth), cpu(y))
    return sm.positive_predictive_value(cm) |> Float32
end
const PPVScore = PrecisionScore


function RecallScore(yhat::AbstractArray, y::AbstractArray; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(cpu(yth), cpu(y))
    return sm.true_positive_rate(cm) |> Float32
end
const TPRScore = RecallScore


function FPRScore(yhat::AbstractArray, y::AbstractArray; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(cpu(yth), cpu(y))
    return sm.false_positive_rate(cm) |> Float32
end


function TNRScore(yhat::AbstractArray, y::AbstractArray; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(cpu(yth), cpu(y))
    return sm.true_negative_rate(cm) |> Float32
end
