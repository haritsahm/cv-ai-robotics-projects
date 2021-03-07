import fcnn

model = fcnn.build_model(3, 1, padding='same', activation='swish')

print(model.summary())