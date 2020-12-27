import tensorflow as tf

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, ofilters, kernel_size, stride, padding='SAME', bias=False):
        super(Conv2D, self).__init__()
        self.OF = ofilters
        self.ks = kernel_size
        self.stride = stride
        self.pad=padding
        self.bias = bias
        # self.W = tf.Variable(tf.random.normal((kernel_size, kernel_size, ifilters, ofilters), 0., 0.02), trainable=True)
        if bias:
            self.b = tf.Variable(tf.zeros((ofilters,), tf.float32), trainable=True)
        else:
            self.b = tf.constant(tf.zeros((ofilters,), tf.float32))
        self.u = tf.Variable(tf.random.normal((ofilters, 1)), trainable=False)
    
    def build(self, x):
        self.IF = x[-1]
        self.W = tf.Variable(tf.random.normal((self.ks, self.ks, self.IF, self.OF), 0.0, 0.02, dtype=tf.float32), trainable=True)
    
    def Wbar(self):
        W = tf.reshape(self.W, (-1, self.W.shape[-1]))
        v = tf.math.l2_normalize(tf.matmul(W, self.u)) # ks*ks*ifilters, 1
        self.u.assign(tf.math.l2_normalize(tf.matmul(tf.transpose(W),v))) # ofilters, 1
        uWv = tf.squeeze(tf.matmul(tf.transpose(v), tf.matmul(W, self.u)))
        return tf.reshape(W, (self.ks, self.ks, self.IF, self.OF))/uWv
    
    def call(self, x):
        return tf.nn.conv2d(x, self.Wbar(), strides=self.stride, padding=self.pad) + self.b

class Conv2DTranspose(tf.keras.layers.Layer):
    def __init__(self, ifilters, ofilters, kernel_size, stride, padding='SAME', bias=False):
        super(Conv2DTranspose, self).__init__()
        self.IF = ifilters
        self.OF = ofilters
        self.ks = kernel_size
        self.stride = stride
        self.pad=padding
        self.bias = bias
        self.W = tf.Variable(tf.random.normal((kernel_size, kernel_size, ofilters, ifilters), 0., 0.02), trainable=True)
        if bias:
            self.b = tf.Variable(tf.random.normal((ofilters,), 0., 0.02), trainable=True)
        else:
            self.b = tf.constant(tf.zeros((ofilters,), tf.float32))
        self.u = tf.Variable(tf.random.normal((ofilters*self.ks**2, 1)), trainable=False)
    
    def Wbar(self):
        W = tf.reshape(self.W, (-1, self.W.shape[-1]))
        v = tf.math.l2_normalize(tf.matmul(tf.transpose(W), self.u)) # ifilters, 1
        self.u.assign(tf.math.l2_normalize(tf.matmul(W, v))) # ofilters*ks*ks, 1
        uWv = tf.squeeze(tf.matmul(tf.transpose(self.u), tf.matmul(W, v)))
        return tf.reshape(W, (self.ks, self.ks, self.OF, self.IF))/uWv
    
    def call(self, x):
        first = -1 if x.shape[0]==None else x.shape[0]
        oshape = (first, x.shape[1]*self.stride, x.shape[2]*self.stride, self.OF)
        return tf.nn.conv2d_transpose(x, self.Wbar(), oshape, strides=(self.stride,self.stride), padding=self.pad) + self.b

class Dense(tf.keras.layers.Layer):
    def __init__(self, units, scale=1):
        super(Dense, self).__init__()
        self.units = units
        self.scale = scale
        self.b = tf.Variable(tf.random.normal((units,), 0.0, 1., dtype=tf.float32), trainable=True)
        self.u = tf.Variable(tf.random.normal((units, 1), dtype=tf.float32), trainable=False)
    
    def build(self, x):
        self.idim = x[-1]
        self.W = tf.Variable(tf.random.normal((self.idim, self.units), 0.0, 1., dtype=tf.float32), trainable=True)
    
    def Wbar(self):
        W = self.W
        v = tf.math.l2_normalize(tf.matmul(W, self.u)) # idim, 1
        self.u.assign(tf.math.l2_normalize(tf.matmul(tf.transpose(W), v))) # units, 1
        uWv = tf.squeeze(tf.matmul(tf.transpose(v), tf.matmul(W, self.u)))/self.scale
        return W/uWv
    
    def call(self, x):
        return tf.matmul(x, self.Wbar()) + self.b
    