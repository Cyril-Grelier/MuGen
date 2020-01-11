# import src.VAE
#
# src.VAE.train()

# import src.train
import numpy as np

from src.data import get_drum
from src.models import get_model

model = get_model('src/rnn_10_classes.h5')

for i in range(1, 100):
    try:
        data = get_drum('datasets/quantized_rythm_dataset/0/random_' + str(i) + '.mid')
        prediction = model.predict(np.stack([data.astype(dtype=float)]))
        index_max = np.argmax(prediction)
        pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][index_max]
        print(f'{i} : {pred}')
    except Exception:
        pass

for i in range(1, 100):
    try:
        data = get_drum('datasets/quantized_rythm_dataset/100/generated_' + str(i) + '.mid')
        prediction = model.predict(np.stack([data.astype(dtype=float)]))
        index_max = np.argmax(prediction)
        pred = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10][index_max]
        print(f'{i} : {pred}')
    except Exception:
        pass
