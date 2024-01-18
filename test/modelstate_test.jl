m1 = Flux.Dense(5,5)
s1 = Flux.state(m1)
p1 = Flux.params(m1)
fpfn = "./modelname.bson"
@test saveModelState(fpfn, m1) === nothing || error("saveModelState failed")


m2 = Flux.Dense(5,5)
loadModelState(fpfn, m2)
p2 = Flux.params(m2)
@test p1 == p2 || error("loadModelState failed")
rm(fpfn)


path = "./testpath"
mkpath(path)
@test saveModelStateCB(path, m1) === nothing || error("saveModelStateCB failed")
rm(path, recursive=true, force=true)
