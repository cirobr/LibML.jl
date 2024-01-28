function IoUScore(yhat, y; threshold=0.5)
    yt = yhat .> threshold
    return IoU(yt, y) |> Float32
end


function AccScore(yhat, y; threshold=0.5)
    yt = yhat .> threshold
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yt, y)
    return sm.accuracy(cm) |> Float32
end
const AccuracyScore = AccScore


function F1Score(yhat, y; threshold=0.5)
    yt = yhat .> threshold
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yt, y)
    return sm.Functions.fscore(cm) |> Float32
end


function PrecisionScore(yhat, y; threshold=0.5)
    yt = yhat .> threshold
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yt, y)
    return sm.positive_predictive_value(cm) |> Float32
end
const PPVScore = PrecisionScore


function RecallScore(yhat, y; threshold=0.5)
    yt = yhat .> threshold
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yt, y)
    return sm.true_positive_rate(cm) |> Float32
end
const TPRScore = RecallScore


function FPRScore(yhat, y; threshold=0.5)
    yt = yhat .> threshold
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yt, y)
    return sm.false_positive_rate(cm) |> Float32
end


function TNRScore(yhat, y; threshold=0.5)
    yt = yhat .> threshold
    cm = sm.ConfusionMatrix(levels=Bool[0,1])(yt, y)
    return sm.true_negative_rate(cm) |> Float32
end
