from tensorflow.keras import layers, Model
import tensorflow as tf
from noisy_layer import NoisyLayer, NoisyLayer2


class DqnAdv(Model):
    def __init__(self, num_inputs: int, num_actions: int, num_hidden: int = 50, name: str = 'dqn_adv'):
        super(DqnAdv, self).__init__(name=name)
        self.layer1 = layers.Dense(num_inputs, activation='relu')
        self.hidden = layers.Dense(num_hidden, activation='relu')
        self.advantage = layers.Dense(num_actions)
        self.value = layers.Dense(1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        hidden = self.hidden(x)
        adv = self.advantage(hidden)
        value = self.value(hidden)

        return value + adv - tf.reduce_mean(adv, axis=1, keepdims=True)

    def get_config(self):
        """ More info about get_config() here:
        https://colab.research.google.com/github/keras-team/keras-io/blob/master
        /guides/ipynb/serialization_and_saving.ipynb#scrollTo=MyWg5idfORK7
        """
        pass


class DqnNoisy(Model):
    def get_config(self):
        pass

    def __init__(self, num_inputs: int, num_actions: int, num_hidden: int = 50, name: str = 'dqn_noisy'):
        super(DqnNoisy, self).__init__(name=name)
        self.layer1 = layers.Dense(num_inputs, activation='relu')
        self.hidden = layers.Dense(num_hidden, activation='relu')
        self.noise = layers.GaussianNoise(stddev=0.1)
        self.q_values = layers.Dense(num_actions)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        hidden = self.hidden(x)
        noise = self.noise(hidden)
        q_values = self.q_values(noise)

        return q_values


class DqnAdvNoisy(Model):
    def __init__(self, num_inputs: int, num_actions: int, num_hidden: int = 50, name: str = 'dqn_noisy_adv'):
        super(DqnAdvNoisy, self).__init__(name=name)
        self.layer1 = layers.Dense(num_inputs, activation='relu')
        self.hidden = layers.Dense(num_hidden, activation='relu')
        self.noise = layers.GaussianNoise(stddev=0.1)
        self.advantage = layers.Dense(num_actions)
        self.value = layers.Dense(1)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        hidden = self.hidden(x)
        noise = self.noise(hidden)
        adv = self.advantage(noise)
        value = self.value(hidden)

        return value + adv - tf.reduce_mean(adv, axis=1, keepdims=True)

    def get_config(self):
        return {"name": self.name}


class DqnNoisyCustom(Model):
    def __init__(self, num_inputs: int, num_actions: int, num_hidden: int = 50, name: str = 'dqn_noisy2'):
        super(DqnNoisyCustom, self).__init__(name=name)
        self.has_custom_noisy_layer = True
        self.layer1 = layers.Dense(num_inputs, activation='relu')
        self.noisy_hidden = NoisyLayer(inputs_dim=num_inputs,
                                       units=num_hidden,
                                       activation_function=tf.nn.relu,
                                       name='hidden')
        self.q_values = layers.Dense(num_actions)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        hidden = self.noisy_hidden(x)
        q_values = self.q_values(hidden)

        return q_values

    def get_config(self):
        return {"name": self.name}


class DqnNoisyCustomAdv(Model):
    def __init__(self, num_inputs: int, num_actions: int, num_hidden: int = 50, name: str = 'dqn_noisy2_adv'):
        super(DqnNoisyCustomAdv, self).__init__(name=name)
        self.has_custom_noisy_layer = True
        self.layer1 = layers.Dense(num_inputs, activation='relu')
        self.noisy_hidden = NoisyLayer(inputs_dim=num_inputs,
                                       units=num_hidden,
                                       activation_function=tf.nn.relu,
                                       name='hidden')
        self.q_values = layers.Dense(num_actions)

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        hidden = self.noisy_hidden(x)
        q_values = self.q_values(hidden)

        return q_values

    def get_config(self):
        return {"name": self.name}


# from tf_agents.agents.categorical_dqn import categorical_dqn_agent
class DqnCat(Model):
    def __init__(self, num_inputs: int,
                 num_actions: int,
                 v_min: int,
                 v_max: int,
                 cat: int = 51,
                 num_hidden: int = 50,
                 name: str = 'dqn_cat'):
        super(DqnCat, self).__init__(name=name)
        self.num_actions = num_actions
        self.v_min = tf.cast(v_min, dtype=tf.float32)
        self.v_max = tf.cast(v_max, dtype=tf.float32)
        self.cat = cat
        self.v_support = tf.reshape(tf.linspace(self.v_min, self.v_max, self.cat, name="support"),
                                    shape=[self.cat, 1])
        self.layer1 = layers.Dense(num_inputs, activation='relu')
        self.hidden = layers.Dense(num_hidden, activation='relu')
        self.q_cat_list = [layers.Dense(cat) for _ in range(self.num_actions)]

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        hidden = self.hidden(x)
        q_cat_values = tf.stack([q_cat(hidden) for q_cat in self.q_cat_list], axis=2)

        q_values = tf.squeeze(tf.tensordot(a=tf.nn.softmax(q_cat_values, axis=1),
                                           b=self.v_support,
                                           axes=[[1], [0]]),
                              axis=2)

        return q_values, q_cat_values

    def get_config(self):
        return {"name": self.name}


class DqnNoisyCustom2(Model):
    def __init__(self, num_inputs: int, num_actions: int, num_hidden: int = 50, name: str = 'dqn_noisy2_adv'):
        super(DqnNoisyCustom2, self).__init__(name=name)
        self.has_custom_noisy_layer = True
        self.layer1 = layers.Dense(num_inputs, activation='relu')
        self.noisy_hidden = NoisyLayer(inputs_dim=num_inputs,
                                       units=num_hidden,
                                       activation_function=tf.nn.relu,
                                       name='hidden')
        self.q_values = NoisyLayer(inputs_dim=num_hidden,
                                   units=num_actions,
                                   activation_function=None,
                                   name='q_values')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        hidden = self.noisy_hidden(x)
        q_values = self.q_values(hidden)

        return q_values

    def get_config(self):
        return {"name": self.name}

    def sample_noise(self):
        return tf.group(self.noisy_hidden.sample_noise_op(),
                        self.q_values.sample_noise_op())

    def deactivate_noise(self):
        return tf.group(self.noisy_hidden.deactivate_noise_op(),
                        self.q_values.deactivate_noise_op())

    def activate_noise(self):
        return tf.group(self.noisy_hidden.activate_noise_op(),
                        self.q_values.activate_noise_op())


class DqnNoisyCustom3(Model):
    def __init__(self, num_inputs: int, num_actions: int, num_hidden: int = 50, name: str = 'dqn_noisy2_adv'):
        super(DqnNoisyCustom3, self).__init__(name=name)
        self.has_custom_noisy_layer = True
        self.layer1 = layers.Dense(num_inputs, activation='relu')
        self.noisy_hidden = NoisyLayer2(inputs_dim=num_inputs,
                                        units=num_hidden,
                                        activation_function=tf.nn.relu,
                                        name='hidden')
        self.q_values = NoisyLayer2(inputs_dim=num_hidden,
                                    units=num_actions,
                                    activation_function=None,
                                    name='q_values')

    @tf.function
    def call(self, inputs, training=None, mask=None):
        x = self.layer1(inputs)
        hidden = self.noisy_hidden(x)
        q_values = self.q_values(hidden)

        return q_values

    def get_config(self):
        return {"name": self.name}

    def sample_noise(self):
        return tf.group(self.noisy_hidden.sample_noise_op(),
                        self.q_values.sample_noise_op())

    def deactivate_noise(self):
        return tf.group(self.noisy_hidden.deactivate_noise_op(),
                        self.q_values.deactivate_noise_op())

    def activate_noise(self):
        return tf.group(self.noisy_hidden.activate_noise_op(),
                        self.q_values.activate_noise_op())

    def apply_noise(self, i=0):
        return tf.group(self.noisy_hidden.apply_noise_i(i),
                        self.q_values.apply_noise_i(i))
