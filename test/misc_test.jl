model = Flux.Conv((3,3), 3=>1, relu)
@test countParams(model) == 28