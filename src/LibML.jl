module LibML


# libraries
export IoU, IoU_loss, softKLDiv
export saveModelState, loadModelState, saveModelStateCB
export trainModel!, evaluateModel
export trainEpoch!, testModel, evaluateEpoch
export IoUScore, AccScore, AccuracyScore, F1Score, PrecisionScore, PPVScore, RecallScore, TPRScore, FPRScore, TNRScore

import Flux
import Flux: cpu
import Statistics: mean
import StatisticalMeasures; sm=StatisticalMeasures
import ProgressBars; pb = ProgressBars   # training/validation with progress bars
import JLD2
import Dates

include("./lossfunctions.jl")   # IoU, classification_metrics
include("./modelstate.jl")      # saveModelState, loadModelState
include("./scores.jl")          # IoUScore, AccScore, F1Score, PrecisionScore, RecallScore, FPRScore, TNRScore
include("./training.jl")        # trainModel!, testModel


end   # module