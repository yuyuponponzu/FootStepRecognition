import sys
# clone した convnet-drawer ディレクトリのパスを追加する。
sys.path.append('convnet-drawer')

from convnet_drawer import Model, Conv2D, MaxPooling2D, Flatten, Dense
"""
model = Model(input_shape=(128,291))
model.add(Conv2D(32,(1,8),strides=(1,2)))
model.add(Conv2D(32,(8,1),strides=(2,1)))
model.add(Conv2D(64,(1,8),strides=(1,2)))
model.add(Conv2D(64,(8,1),strides=(2,1)))
"""
"""
model = Model(input_shape=(128,291))
model.add(Conv2D(32,(1,16),strides=(1,2)))
model.add(Conv2D(32,(16,1),strides=(2,1)))
model.add(Conv2D(64,(1,16),strides=(1,2)))
model.add(Conv2D(64,(16,1),strides=(2,1)))
"""
"""
model = Model(input_shape=(128,291))
model.add(Conv2D(32,(1,32),strides=(1,2)))
model.add(Conv2D(32,(32,1),strides=(2,1)))
model.add(Conv2D(64,(1,32),strides=(1,2)))
model.add(Conv2D(64,(32,1),strides=(2,1)))
"""
"""
model = Model(input_shape=(128,291))
model.add(Conv2D(32,(1,64),strides=(1,2)))
model.add(Conv2D(32,(64,1),strides=(2,1)))
model.add(Conv2D(64,(1,64),strides=(1,2)))
model.add(Conv2D(64,(64,1),strides=(2,1)))
"""

model.add(Conv2D(128,(1,16),strides=(1,2)))
model.add(Conv2D(128,(16,1),strides=(2,1)))
model.add(GlobalAveragePooling2D())
model.add(Dense(4))


