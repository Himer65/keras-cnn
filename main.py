import numpy as np
from tensorflow import keras 

from utils import newModel, visualiz

"""Единица под индексом 0 - это кадры из аниме Песнь ночных сов.
   Единица под индексом 1 - это кадры из аниме Наруто.
   Единица под индексом 2 - это кадры из аниме Клинок рассекающий демонов.
   Единица под индексом 3 - это кадры из аниме Евангелион.
   Единица под индексом 4 - это кадры из аниме Перерождение дяди.
   Посмотреть на графики обучения можно в файлах ./model/history.png
   Используемый датасет ./model/dataset.npz"""

data = np.load('./model/dataset.npz')
x = data['x']; y = data['y']
epoch = 15

model = newModel()
model.compile(optimizer=keras.optimizers.Adadelta(learning_rate=0.1),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

callback = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=5)
history = model.fit(x=x, y=y, epochs=epoch,
                    shuffle=True, batch_size=50,
                    validation_split=0.04,
                    callbacks=[callback])
model.save('./model/base.h5')
visualiz(history, epoch, 3) #кривая валидации такая ломанная из-за маленького пакета, но больше взять я не мог, т.к. данных и так мало
