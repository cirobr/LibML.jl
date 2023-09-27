# libraries
import CUDA: CuIterator


# Flux v0.14
function trainModel!(model, data, opt_state, lossf, losses)

    # warningMessage = false
    for (X, y) in CuIterator(data)
        # Any code inside here is differentiated.
        # Evaluation of the model and loss must be inside!
        # grads = Flux.gradient(model) do m
        val, grads = Flux.withgradient(model) do m
            lossf(m(X), y)   # val = last code line within do
        end

        # Save the loss from the forward pass. (Done outside of gradient.)
        push!(losses, val)

        # Detect loss of Inf or NaN. Print a warning, and then skip update!
        if !isfinite(val)    # & !warningMessage
            @warn "skipping loss equal to $val"
            # warningMessage = true
            continue
        end

        # Update the parameters so as to reduce the objective,
        # according the chosen optimisation rule:
        Flux.update!(opt_state, model, grads[1])
    end

    return
end