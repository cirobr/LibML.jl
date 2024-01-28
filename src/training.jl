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
# lossfns = [LibML.IoU_loss,
#            Flux.crossentropy,
#            Flux.mse,
# ]

function testModel(model, data, lossfn, metrics)
    lossfns = vcat(lossfn, metrics)
    losses = Array{Float32,2}(undef, (length(data), length(lossfns)))

    # i=0
    for (i, (X,y)) in pb.ProgressBar( enumerate(data) )
    # for (X,y) in data
        # i+=1
        yhat = model(X)
        lossfn(yhat, y)   # temporary, to get lossfn to compile
        j=0
        # for (j, lossfn) in enumerate(lossfns)
        for lfn in lossfns
            j+=1
            # losses[i,j] = lfn(yhat, y)
        end
    end

    return vec( mean(losses; dims=1) )
end
