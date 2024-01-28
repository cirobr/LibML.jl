function IoUScore(yhat, y; threshold=0.5)
    # yth = yhat .> threshold
    yth = map(x -> x > threshold ? 1 : 0, yhat)
    return IoU(yth, y) |> Float32
end


function AccScore(yhat, y; threshold=0.5)
    # yth = yhat .> threshold
    yth = map(x -> x > threshold ? 1 : 0, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.accuracy(cm) |> Float32
end
const AccuracyScore = AccScore


function F1Score(yhat, y; threshold=0.5)
    # yth = yhat .> threshold
    yth = map(x -> x > threshold ? 1 : 0, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.Functions.fscore(cm) |> Float32
end


function PrecisionScore(yhat, y; threshold=0.5)
    # yth = yhat .> threshold
    yth = map(x -> x > threshold ? 1 : 0, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.positive_predictive_value(cm) |> Float32
end
const PPVScore = PrecisionScore


function RecallScore(yhat, y; threshold=0.5)
    # yth = yhat .> threshold
    yth = map(x -> x > threshold ? 1 : 0, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.true_positive_rate(cm) |> Float32
end
const TPRScore = RecallScore


function FPRScore(yhat, y; threshold=0.5)
    # yth = yhat .> threshold
    yth = map(x -> x > threshold ? 1 : 0, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.false_positive_rate(cm) |> Float32
end


function TNRScore(yhat, y; threshold=0.5)
    # yth = yhat .> threshold
    yth = map(x -> x > threshold ? 1 : 0, yhat)
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yth, y)
    return sm.true_negative_rate(cm) |> Float32
end
