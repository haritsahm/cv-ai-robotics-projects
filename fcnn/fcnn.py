from typing import Optional, Union, Callable, List

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam

import fcnn.metrics

class ConvBlock(layers.Layer):

		def __init__(self, filters, kernel_size, dropout_rate, padding, activation, **kwargs):
				super(ConvBlock, self).__init__(**kwargs)
				self.filters=filters
				self.kernel_size=kernel_size
				self.dropout_rate=dropout_rate
				self.padding=padding
				self.activation=activation

				self.conv2d = layers.Conv2D(filters=filters, kernel_size=(kernel_size, kernel_size),
											padding=padding)
				self.batchnorm = layers.BatchNormalization()
				self.dropout = layers.Dropout(rate=dropout_rate)
				self.activation = layers.Activation(activation)

		def call(self, inputs, training=None, **kwargs):
				x = inputs
				x = self.conv2d(x)
				x = self.batchnorm(x)
				x = self.activation(x)
				if training:
					x = self.dropout(x)

				return x

		def get_config(self):
				return dict(filters=self.filters,
										kernel_size=self.kernel_size,
										dropout_rate=self.dropout_rate,
										padding=self.padding,
										activation=self.activation,
										**super(ConvBlock, self).get_config(),
										)

def build_model(nx: Optional[int] = None,
								ny: Optional[int] = None,
								channels: int = 1,
								num_classes: int = 2,
								dropout_rate: int = 0.5,
								padding:str="valid",
								activation:Union[str, Callable]="relu") -> Model:

		inputs = Input(shape=(nx, ny, channels), name="inputs")

		x = inputs
		# --- ENCODE ---
		# Conv1
		out = ConvBlock(16, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(x)
		before_maxpool1 = out
		maxpool1 = layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding=padding)(out)

		# Conv2
		out = ConvBlock(32, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 
		# Conv3 
		out = ConvBlock(32, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 

		# Concat 1
		out = layers.concatenate([out, maxpool1], 3)
		concat1 = out

		# Conv4 
		out = layers.MaxPool2D(pool_size=[2,2], strides=[2,2], padding=padding)(out)
		maxpool2 = out
		out = ConvBlock(64, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 

		# Conv 5
		out = ConvBlock(64, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 

		# Concat 2
		out = layers.concatenate([out, maxpool2], 3)
		concat2 = out

		# Conv6
		out = ConvBlock(128, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 
		# Conv7 
		out = ConvBlock(128, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 

		# --- DECODE ---
		# Concat 4
		out = layers.experimental.preprocessing.Resizing(75, 100)(out)
		out = layers.concatenate([out, concat1], 3)

		# Conv 13
		out = ConvBlock(64, 1, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 
		# Conv 14
		out = ConvBlock(32, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 
		# Conv 15
		out = ConvBlock(32, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 

		# Concat 5
		out = layers.experimental.preprocessing.Resizing(150, 200)(out)
		out = layers.concatenate([out, before_maxpool1], 3)

		# Conv 16
		out = ConvBlock(16, 1, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 
		# Conv 17
		out = ConvBlock(16, 3, dropout_rate=dropout_rate, padding=padding, activation=activation)(out) 

		outputs = layers.Conv2D(filters=num_classes,
								kernel_size=(3, 3),
								strides=1,
								padding=padding, name='outputs')(out)

		model = Model(inputs, outputs, name="fcnn")

		return model

def finalize_model(model: Model,
									 loss: Optional[Union[Callable, str]]=losses.categorical_crossentropy,
									 optimizer: Optional= None,
									 metrics:Optional[List[Union[Callable,str]]]=None,
									 dice_coefficient: bool=False,
									 auc: bool=False,
									 mean_iou: bool=False,
									 **opt_kwargs):
		"""
		Configures the model for training by setting, loss, optimzer, and tracked metrics
		:param model: the model to compile
		:param loss: the loss to be optimized. Defaults to `categorical_crossentropy`
		:param optimizer: the optimizer to use. Defaults to `Adam`
		:param metrics: List of metrics to track. Is extended by `crossentropy` and `accuracy`
		:param dice_coefficient: Flag if the dice coefficient metric should be tracked
		:param auc: Flag if the area under the curve metric should be tracked
		:param mean_iou: Flag if the mean over intersection over union metric should be tracked
		:param opt_kwargs: key word arguments passed to default optimizer (Adam), e.g. learning rate
		"""

		if optimizer is None:
				optimizer = Adam(**opt_kwargs)

		if metrics is None:
				metrics = ['categorical_crossentropy',
									 'categorical_accuracy',
									 ]

		if mean_iou:
				metrics += [fcnn.metrics.mean_iou]

		if dice_coefficient:
				metrics += [fcnn.metrics.dice_coefficient]

		if auc:
				metrics += [tf.keras.metrics.AUC()]

		model.compile(loss=loss, optimizer=optimizer, metrics=metrics)