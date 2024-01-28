function trainModel!(model, data, optstate, lossfn; verbose=false)
    steplosses = Vector{Float32}(undef, length(data))

    for (i, (X,y)) in pb.ProgressBar( enumerate(data) )
        loss, grads = Flux.withgradient(model) do m
            lossfn(m(X), y)
        end

        Flux.update!(optstate, model, grads[1])
        steplosses[i] = loss
    end

    epochloss = mean(steplosses)

    if verbose   return epochloss, steplosses
    else         return epochloss
    end
end


# example of metrics vector, all metrics with format fn(yhat, y):
# metrics = [LibML.IoUScore,
#            LibML.AccScore,
#            LibML.F1Score,
#            Flux.mse,
# ]

function testModel(model, data, lossfn, metrics)
    lossfns = vcat(lossfn, metrics)
    losses = Array{Float32,2}(undef, (length(data), length(lossfns)))

    for (i, (X,y)) in pb.ProgressBar( enumerate(data) )
        yhat = model(X)

        for (j, lfn) in enumerate(lossfns)
            losses[i,j] = lfn(yhat, y)
        end
    end

    res = vec( mean(losses; dims=1) )
    return res[1], res[2:end]   # return loss, metrics
end
