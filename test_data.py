from data_loader import imagenet_loader
import numpy as np
loader=imagenet_loader()
batch=next(loader.train_next_batch(batch_size=1))

print(batch[0])
print(batch[1])
print(np.shape(batch[0]))
print(np.shape(batch[1]))
