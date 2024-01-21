function trainModel!(model, data, optstate, lossfn; verbose=false)
    steplosses = Vector{Float32}(undef, length(data))

    for (i, (X,y)) in enumerate(data)
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


# lossfns is a vector of loss functions
function testModel(model, data, lossfns)
    losses = Array{Float32,2}(undef, (length(data), length(lossfns)))

    for (i, (X,y)) in enumerate(data)
        yhat = model(X)
        for (j, lossfn) in enumerate(lossfns)
            losses[i,j] = lossfn(yhat, y)
        end
    end

    return mean(losses; dims=1)
end
