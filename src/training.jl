function trainModel!(model, data, optstate, lossfn, nepochs::Int=1; verbose=false)
    lendata     = length(data)
    epochLosses = Vector{Float32}(undef, nepochs)
    stepLosses  = Vector{Float32}(undef, nepochs * lendata)

    for epoch in 1:nepochs
        epochsteplosses = Vector{Float32}(undef, lendata)

        for (i, (X,y)) in enumerate(data)
            loss, grads = Flux.withgradient(model) do m
                yhat = m(X)
                lossfn(yhat, y)
            end

            Flux.update!(optstate, model, grads[1])

            epochsteplosses[i] = loss
            stepLosses[(epoch-1)*lendata + i] = loss
        end

        epochLosses[epoch] = mean(epochsteplosses)
    end

    if verbose   return epochLosses, stepLosses
    else         return
    end
end


function testModel(model, data, lossfn)
    losses = Vector{Float32}(undef, length(data))

    for (i, (X,y)) in enumerate(data)
        yhat = model(X)
        losses[i] = lossfn(yhat, y)
    end

    return losses
end
