module LibML


# libraries
export IoU, IoU_loss
export saveModelState, loadModelState, saveModelStateCB
export trainModel!, testModel

import Flux
import Statistics: mean
import StatisticalMeasures; sm=StatisticalMeasures
import ProgressBars; pb = ProgressBars   # training/validation with progress bars
import JLD2
import Dates

include("./modelstate.jl")      # saveModelState, loadModelState
include("./lossfunctions.jl")   # IoU, classification_metrics
include("./training.jl")        # trainModel!, testModel


end   # module