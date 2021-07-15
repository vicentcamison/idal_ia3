import tensorflow as tf


class NoisyLayer:
    """
    Implementation of NoisyLayer from paper: https://arxiv.org/pdf/1706.10295.pdf
    Independent Gaussian noise:
        μ is sampled from independent uniform distributions U[−sqrt(3/p), +sqrt(3/p)]
        σ is simply set to 0.017
    Factorized Gaussian noise:
        μ is sampled from inputs of independent uniform distributions U[−sqrt(1/p), +sqrt(1/p)]
        σ is initialised to a constant σ0/sqrt(p).  The hyper-parameter σ0 is set to 0.5.
    """

    def __init__(self,
                 inputs_dim=None,
                 units=None,
                 activation_function=None,
                 name=None,
                 noise_type='factorized'):
        self.inputs_dim = inputs_dim
        self.units = units
        self.activation = activation_function
        self.noise_type = noise_type
        self.name = name
        if self.noise_type == 'factorized':  # Factorized Gaussian noise

            bound = pow(self.inputs_dim, -0.5)
            self.initializer = tf.initializers.random_uniform(minval=-bound,
                                                              maxval=bound)
            bound_sigma = 0.5 * pow(self.inputs_dim, -0.5)
            self.sigma_init = tf.constant_initializer(bound_sigma)
        else:
            bound = pow(3 / self.inputs_dim, 0.5)
            self.initializer = tf.initializers.random_uniform(minval=-bound,
                                                              maxval=bound)
            self.sigma_init = tf.constant_initializer(0.017)

        self.sigma_kernel = tf.compat.v1.get_variable(name='_'.join([self.name, 'sigma_k']),
                                                      shape=(self.inputs_dim, self.units),
                                                      initializer=self.sigma_init,
                                                      trainable=True)
        self.sigma_bias = tf.compat.v1.get_variable(name='_'.join([self.name, 'sigma_b']),
                                                    shape=(self.units,),
                                                    initializer=self.sigma_init,
                                                    trainable=True)

        self.epsilon_kernel = tf.compat.v1.get_variable(name='_'.join([self.name, 'eps_k']),
                                                        shape=(self.inputs_dim, self.units),
                                                        initializer=tf.zeros_initializer,
                                                        trainable=False)

        self.epsilon_bias = tf.compat.v1.get_variable(name='_'.join([self.name, 'eps_b']),
                                                      shape=(self.units,),
                                                      initializer=tf.zeros_initializer,
                                                      trainable=False)

        self.output = tf.keras.layers.Dense(units=self.units,
                                            activation=self.activation,
                                            kernel_initializer=self.initializer,
                                            name='_'.join([self.name, 'mu']))

        self.noise_bool = tf.compat.v1.get_variable(name='_'.join([self.name, 'active_noise']),
                                                    initializer=tf.constant(True),
                                                    use_resource=True,
                                                    dtype=tf.bool,
                                                    trainable=False)
        self.eps_mask_k = tf.compat.v1.get_variable(name='_'.join([self.name, 'mask_k']),
                                                    shape=(self.inputs_dim, self.units),
                                                    initializer=tf.ones_initializer,
                                                    trainable=False)
        self.eps_mask_b = tf.compat.v1.get_variable(name='_'.join([self.name, 'mask_b']),
                                                    shape=(self.units,),
                                                    initializer=tf.ones_initializer,
                                                    trainable=False)

    def __call__(self, inputs):
        masked_eps_sigma = tf.multiply(self.eps_mask_k, self.epsilon_kernel)
        masked_eps_bias = tf.multiply(self.eps_mask_b, self.epsilon_bias)

        perturbed_sigma = tf.multiply(self.sigma_kernel, masked_eps_sigma)
        perturbed_bias = tf.multiply(self.sigma_bias, masked_eps_bias)
        if self.activation is not None:
            perturbed_output = self.activation(tf.matmul(inputs, perturbed_sigma) + perturbed_bias)
        else:
            perturbed_output = tf.matmul(inputs, perturbed_sigma) + perturbed_bias

        return self.output(inputs) + perturbed_output

    def activate_noise_op(self):
        ones_kernel_op = tf.compat.v1.assign(ref=self.eps_mask_k,
                                             value=tf.ones(shape=(self.inputs_dim, self.units)))
        ones_bias_op = tf.compat.v1.assign(ref=self.eps_mask_b,
                                           value=tf.ones(shape=(self.units,)))
        return tf.group(ones_kernel_op, ones_bias_op)

    def deactivate_noise_op(self):
        zero_kernel_op = tf.compat.v1.assign(ref=self.eps_mask_k,
                                             value=tf.zeros(shape=(self.inputs_dim, self.units)))
        zero_bias_op = tf.compat.v1.assign(ref=self.eps_mask_b,
                                           value=tf.zeros(shape=(self.units,)))
        return tf.group(zero_kernel_op, zero_bias_op)

    def sample_noise_op(self):
        if self.noise_type == 'factorized':  # Factorized Gaussian noise
            def f(e_list):
                return tf.multiply(tf.sign(e_list), tf.pow(tf.abs(e_list), 0.5))

            noise_1 = f(tf.compat.v1.random_normal(tf.TensorShape([self.inputs_dim, 1]), dtype=tf.float32))
            noise_2 = f(tf.compat.v1.random_normal(tf.TensorShape([1, self.units]), dtype=tf.float32))
            noise_kernel = tf.matmul(noise_1, noise_2)
            noise_bias = tf.compat.v1.random_normal(shape=(self.units,),
                                                    mean=0.0,
                                                    stddev=1.0,
                                                    dtype=tf.dtypes.float32)
        else:  # Independent Gaussian noise
            noise_kernel = tf.compat.v1.random_normal(shape=(self.inputs_dim, self.units),
                                                      mean=0.0,
                                                      stddev=1.0,
                                                      dtype=tf.dtypes.float32)
            noise_bias = tf.compat.v1.random_normal(shape=(self.units,),
                                                    mean=0.0,
                                                    stddev=1.0,
                                                    dtype=tf.dtypes.float32)

        sample_kernel_op = tf.compat.v1.assign(ref=self.epsilon_kernel,
                                               value=noise_kernel)
        sample_bias_op = tf.compat.v1.assign(ref=self.epsilon_bias,
                                             value=noise_bias)
        return tf.group(sample_kernel_op, sample_bias_op)


class NoisyLayer2:
    """
    Implementation of NoisyLayer from paper: https://arxiv.org/pdf/1706.10295.pdf
    Independent Gaussian noise:
        μ is sampled from independent uniform distributions U[−sqrt(3/p), +sqrt(3/p)]
        σ is simply set to 0.017
    Factorized Gaussian noise:
        μ is sampled from inputs of independent uniform distributions U[−sqrt(1/p), +sqrt(1/p)]
        σ is initialised to a constant σ0/sqrt(p).  The hyper-parameter σ0 is set to 0.5.
    """

    def __init__(self,
                 inputs_dim=None,
                 units=None,
                 activation_function=None,
                 name=None,
                 noise_type='factorized',
                 noise_slots=2):
        self.inputs_dim = inputs_dim
        self.units = units
        self.activation = activation_function
        self.noise_type = noise_type
        self.name = name
        self.noise_slots = noise_slots
        if self.noise_type == 'factorized':  # Factorized Gaussian noise

            bound = pow(self.inputs_dim, -0.5)
            self.initializer = tf.initializers.random_uniform(minval=-bound,
                                                              maxval=bound)
            bound_sigma = 0.5 * pow(self.inputs_dim, -0.5)
            self.sigma_init = tf.constant_initializer(bound_sigma)
        else:
            bound = pow(3 / self.inputs_dim, 0.5)
            self.initializer = tf.initializers.random_uniform(minval=-bound,
                                                              maxval=bound)
            self.sigma_init = tf.constant_initializer(0.017)

        self.sigma_kernel = tf.compat.v1.get_variable(name='_'.join([self.name, 'sigma_k']),
                                                      shape=(self.inputs_dim, self.units),
                                                      initializer=self.sigma_init,
                                                      trainable=True)
        self.sigma_bias = tf.compat.v1.get_variable(name='_'.join([self.name, 'sigma_b']),
                                                    shape=(self.units,),
                                                    initializer=self.sigma_init,
                                                    trainable=True)

        self.epsilon_kernel = tf.compat.v1.get_variable(name='_'.join([self.name, 'eps_k']),
                                                        shape=(self.inputs_dim, self.units),
                                                        initializer=tf.zeros_initializer,
                                                        trainable=False)

        self.epsilon_bias = tf.compat.v1.get_variable(name='_'.join([self.name, 'eps_b']),
                                                      shape=(self.units,),
                                                      initializer=tf.zeros_initializer,
                                                      trainable=False)

        self.output = tf.keras.layers.Dense(units=self.units,
                                            activation=self.activation,
                                            kernel_initializer=self.initializer,
                                            name='_'.join([self.name, 'mu']))

        self.noise_bool = tf.compat.v1.get_variable(name='_'.join([self.name, 'active_noise']),
                                                    initializer=tf.constant(True),
                                                    use_resource=True,
                                                    dtype=tf.bool,
                                                    trainable=False)
        self.eps_mask_k = tf.compat.v1.get_variable(name='_'.join([self.name, 'mask_k']),
                                                    shape=(self.inputs_dim, self.units),
                                                    initializer=tf.ones_initializer,
                                                    trainable=False)
        self.eps_mask_b = tf.compat.v1.get_variable(name='_'.join([self.name, 'mask_b']),
                                                    shape=(self.units,),
                                                    initializer=tf.ones_initializer,
                                                    trainable=False)

        self.noise_kernel_buffer = [tf.compat.v1.get_variable(name='_'.join([self.name, 'eps_k', '_' + str(i)]),
                                                              shape=(self.inputs_dim, self.units),
                                                              initializer=tf.zeros_initializer,
                                                              trainable=False)
                                    for i, _ in enumerate(range(self.noise_slots))]
        self.noise_bias_buffer = [tf.compat.v1.get_variable(name='_'.join([self.name, 'eps_b', '_' + str(i)]),
                                                            shape=(self.units,),
                                                            initializer=tf.zeros_initializer,
                                                            trainable=False)
                                  for i, _ in enumerate(range(self.noise_slots))]

    def __call__(self, inputs):
        masked_eps_sigma = tf.multiply(self.eps_mask_k, self.epsilon_kernel)
        masked_eps_bias = tf.multiply(self.eps_mask_b, self.epsilon_bias)

        perturbed_sigma = tf.multiply(self.sigma_kernel, masked_eps_sigma)
        perturbed_bias = tf.multiply(self.sigma_bias, masked_eps_bias)
        if self.activation is not None:
            perturbed_output = self.activation(tf.matmul(inputs, perturbed_sigma) + perturbed_bias)
        else:
            perturbed_output = tf.matmul(inputs, perturbed_sigma) + perturbed_bias

        return self.output(inputs) + perturbed_output

    def activate_noise_op(self):
        ones_kernel_op = tf.compat.v1.assign(ref=self.eps_mask_k,
                                             value=tf.ones(shape=(self.inputs_dim, self.units)))
        ones_bias_op = tf.compat.v1.assign(ref=self.eps_mask_b,
                                           value=tf.ones(shape=(self.units,)))
        return tf.group(ones_kernel_op, ones_bias_op)

    def deactivate_noise_op(self):
        zero_kernel_op = tf.compat.v1.assign(ref=self.eps_mask_k,
                                             value=tf.zeros(shape=(self.inputs_dim, self.units)))
        zero_bias_op = tf.compat.v1.assign(ref=self.eps_mask_b,
                                           value=tf.zeros(shape=(self.units,)))
        return tf.group(zero_kernel_op, zero_bias_op)

    def sample_noise_op(self):
        kernel_ops, bias_ops = [], []
        for i in range(self.noise_slots):
            if self.noise_type == 'factorized':  # Factorized Gaussian noise
                def f(e_list):
                    return tf.multiply(tf.sign(e_list), tf.pow(tf.abs(e_list), 0.5))

                noise_1 = f(tf.compat.v1.random_normal(tf.TensorShape([self.inputs_dim, 1]), dtype=tf.float32))
                noise_2 = f(tf.compat.v1.random_normal(tf.TensorShape([1, self.units]), dtype=tf.float32))
                noise_kernel = tf.matmul(noise_1, noise_2)
                noise_bias = tf.compat.v1.random_normal(shape=(self.units,),
                                                        mean=0.0,
                                                        stddev=1.0,
                                                        dtype=tf.dtypes.float32)
            else:  # Independent Gaussian noise
                noise_kernel = tf.compat.v1.random_normal(shape=(self.inputs_dim, self.units),
                                                          mean=0.0,
                                                          stddev=1.0,
                                                          dtype=tf.dtypes.float32)
                noise_bias = tf.compat.v1.random_normal(shape=(self.units,),
                                                        mean=0.0,
                                                        stddev=1.0,
                                                        dtype=tf.dtypes.float32)

            kernel_ops.append(tf.compat.v1.assign(ref=self.noise_kernel_buffer[i], value=noise_kernel))
            bias_ops.append(tf.compat.v1.assign(ref=self.noise_bias_buffer[i], value=noise_bias))

        return tf.group(*kernel_ops, *bias_ops)

    def apply_noise_i(self, i=0):
        sample_kernel_op = tf.compat.v1.assign(ref=self.epsilon_kernel,
                                               value=self.noise_kernel_buffer[i])
        sample_bias_op = tf.compat.v1.assign(ref=self.epsilon_bias,
                                             value=self.noise_bias_buffer[i])
        return tf.group(sample_kernel_op, sample_bias_op)
