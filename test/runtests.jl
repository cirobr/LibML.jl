using LibML
using Test

@testset "LibML.jl" begin
    yhat = [0.751649  0.505124  0.3802    0.234363   0.42201
            0.978538  0.828006  0.30808   0.0104223  0.990135
            0.471218  0.118804  0.38585   0.501337   0.40801
            0.152852  0.379363  0.273147  0.373496   0.296733
            0.993718  0.131977  0.733328  0.396807   0.422871] .|> Float32
    y = yhat .- 0.01 .|> Float32
    
    @test LibML.IoU(yhat, y) == 0.44305855f0 || error("IoU failed")
    @test LibML.IoU_loss(yhat, y) == 0.55694145f0 || error("IoU_loss failed")
end
