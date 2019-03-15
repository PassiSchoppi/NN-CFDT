training_mode = 0
neuronsPerLayer = 0
layer = 0
training_length = 400
counter = 0

while True:
    training_mode = counter % 3
    if training_mode == 0:
        neuronsPerLayer = neuronsPerLayer + 5
    if neuronsPerLayer % 40 == 0:
        neuronsPerLayer = 5
        layer = layer + 5
    if layer % 40 == 0:
        layer = 5
        training_length = training_length + 100
    print("trm: "+str(training_mode))
    print("npl: "+str(neuronsPerLayer))
    print("lay: "+str(layer))
    print("trl: "+str(training_length))
    print()
    counter = counter + 1
