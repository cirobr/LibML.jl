function trainModel!(model, data, optstate, lossfn; epochs::Int=1, verbose=false)
    lendata     = length(data)
    epochLosses = Vector{Float32}(undef, epochs)
    stepLosses  = Vector{Float32}(undef, epochs * lendata)

    for epoch in 1:epochs
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


# lossfns is a vector of loss functions
function testModel(model, data, lossfns)
    losses = Array{Float32,2}(undef, (length(data), length(lossfns)))

    for (i, (X,y)) in enumerate(data)
        yhat = model(X)
        for (j, lossfn) in enumerate(lossfns)
            losses[i,j] = lossfn(yhat, y)
        end
    end

    return losses
end