lines = [line.rstrip('\n') for line in open("../wrt/NeuNetNorm/key_data.txt")]

neurons_per_layer = str(lines[0])[21:len(lines[0])-1]
layer = str(lines[2])[9:len(lines[2])-1]
records = str(lines[4])[10:len(lines[4])-1]
training_length = str(lines[5])[19:len(lines[5])-1]

print("neurons_per_layer: "+str(neurons_per_layer))
print("layer: "+str(layer))
print("records: "+str(records))
print("training_length: "+str(training_length))
