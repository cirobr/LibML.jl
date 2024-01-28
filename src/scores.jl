function IoUScore(yhat::Array{Float32}, y::Array{Bool}; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    return IoU(yth, y) |> Float32
end


function AccScore(yhat::Array{Float32}, y::Array{Bool}; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.accuracy(cm) |> Float32
end
const AccuracyScore = AccScore


function F1Score(yhat::Array{Float32}, y::Array{Bool}; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.Functions.fscore(cm) |> Float32
end


function PrecisionScore(yhat::Array{Float32}, y::Array{Bool}; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.positive_predictive_value(cm) |> Float32
end
const PPVScore = PrecisionScore


function RecallScore(yhat::Array{Float32}, y::Array{Bool}; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.true_positive_rate(cm) |> Float32
end
const TPRScore = RecallScore


function FPRScore(yhat::Array{Float32}, y::Array{Bool}; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.false_positive_rate(cm) |> Float32
end


function TNRScore(yhat::Array{Float32}, y::Array{Bool}; threshold=0.5)
    yth = map(x -> x > threshold ? true : false, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.true_negative_rate(cm) |> Float32
end
