function trainModel!(model, data, optstate, lossfn, nepochs::Int=1; verbose=false)
    lendata     = length(data)
    epochLosses = Vector{Float32}(undef, nepochs)
    stepLosses  = Vector{Float32}(undef, nepochs * lendata)

    for epoch in 1:nepochs
        epochsteplosses = Vector{Float32}(undef, lendata)

        for (i, (X,y)) in enumerate(data)
            loss, grads = Flux.withgradient(m) do m
                yhat = model(X)
                lossfn(yhat, y)
            end

            Flux.update!(optstate, m, grads[1])

            epochsteplosses[i] = loss
            stepLosses[(epoch-1)*lendata + i] = loss
        end

        epochLosses[epoch] = mean(epochsteplosses)
    end

    if verbose   return epochLosses, stepLosses
    else         return
    end
end