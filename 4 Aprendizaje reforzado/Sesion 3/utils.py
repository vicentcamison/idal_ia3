import numpy as np
import gym
from collections import deque
import tensorflow as tf


def discrete_input(state_discrete: tuple, env_dim: gym.spaces.tuple.Tuple):
    one_hot_state = []
    for i_pos, dim in zip(state_discrete, env_dim):
        temp = np.zeros(dim.n)
        temp[i_pos] = 1
        one_hot_state.append(temp)

    return np.concatenate(one_hot_state)


def distr_projection(next_distr, rewards, dones, v_min, v_max, n_atoms, gamma):
    """
    Perform distribution projection aka Catergorical Algorithm from the
    "A Distributional Perspective on RL" paper
    """
    rewards = rewards.flatten()
    dones = dones.flatten()
    batch_size = len(rewards)
    proj_distr = np.zeros((batch_size, n_atoms), dtype=np.float32)
    delta_z = (v_max - v_min) / (n_atoms - 1)
    for atom in range(n_atoms):
        tz_j = np.minimum(v_max, np.maximum(v_min, rewards + (v_min + atom * delta_z) * gamma))
        b_j = (tz_j - v_min) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        proj_distr[eq_mask, l[eq_mask]] += next_distr[eq_mask, atom]
        ne_mask = u != l
        proj_distr[ne_mask, l[ne_mask]] += next_distr[ne_mask, atom] * (u - b_j)[ne_mask]
        proj_distr[ne_mask, u[ne_mask]] += next_distr[ne_mask, atom] * (b_j - l)[ne_mask]

    if dones.any():
        proj_distr[dones] = 0.0
        tz_j = np.minimum(v_max, np.maximum(v_min, rewards[dones]))
        b_j = (tz_j - v_min) / delta_z
        l = np.floor(b_j).astype(np.int64)
        u = np.ceil(b_j).astype(np.int64)
        eq_mask = u == l
        eq_dones = dones.copy()
        eq_dones[dones] = eq_mask
        if eq_dones.any():
            proj_distr[eq_dones, l[eq_mask]] = 1.0
        ne_mask = u != l
        ne_dones = dones.copy()
        ne_dones[dones] = ne_mask
        if ne_dones.any():
            proj_distr[ne_dones, l[ne_mask]] = (u - b_j)[ne_mask]
            proj_distr[ne_dones, u[ne_mask]] = (b_j - l)[ne_mask]

    return proj_distr


def softmax(x, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    :param x: ND-Array. Probably should be floats.
    :param theta: float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0 (optional).
    :param axis: axis to compute values along. Default is the
        first non-singleton axis (optional).
    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(x)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(x.shape) == 1:
        p = p.flatten()

    return p


def get_n_experience(buffer: deque, gamma: float):
    state_n = buffer[0][0]
    action_n = buffer[-1][1]
    reward_n = np.sum([buffer[i][2] * gamma ** i for i in range(len(buffer))], dtype=np.float32)
    next_state_n = buffer[-1][3]
    done_n = buffer[-1][4]
    return state_n, action_n, reward_n, next_state_n, done_n


# ### UNFINISHED WORK TO IMPLEMENT THE FUNCTION IN TENSORFLOW. Tensor slicing limits the possibility.
# def distr_projection(next_distr, rewards, dones, v_min, v_max, n_atoms, gamma):
#     """
#     Perform distribution projection aka Catergorical Algorithm from the
#     "A Distributional Perspective on RL" paper
#     """
#     # rewards = rewards.flatten()
#     # dones = dones.flatten()
#     batch_size = len(rewards)
#     proj_distr = tf.zeros((batch_size, n_atoms), dtype=tf.float32)
#     delta_z = (v_max - v_min) / (n_atoms - 1)
#
#     support = tf.tile(tf.expand_dims(tf.linspace(v_min, v_max, n_atoms), 0), [batch_size, 1])
#     rewards_tiled = tf.tile(tf.expand_dims(rewards, 1), [1, n_atoms])
#     tz_j_mat = tf.clip_by_value(support * gamma + rewards_tiled, v_min, v_max)
#     b_j_mat = (tz_j_mat - v_min) / delta_z
#     lower_mat = tf.math.floor(b_j_mat)
#     upper_mat = tf.math.ceil(b_j_mat)
#
#     eq_mask_mat = tf.cast(upper_mat == lower_mat, dtype=tf.float32)
#
#     proj_distr += next_distr * eq_mask_mat
#
#     ne_mask_mat = tf.cast(upper_mat != lower_mat, dtype=tf.float32)
#     proj_distr += next_distr * (upper_mat - b_j_mat) * ne_mask_mat
#     proj_distr += next_distr * (b_j_mat - lower_mat) * ne_mask_mat
#
#     for atom in range(n_atoms):
#         tz_j = tf.minimum(v_max, tf.maximum(v_min, rewards + (v_min + atom * delta_z) * gamma))
#         b_j = (tz_j - v_min) / delta_z
#         lower = tf.cast(tf.math.floor(b_j), dtype=tf.int64)
#         upper = tf.cast(tf.math.ceil(b_j), dtype=tf.int64)
#         eq_mask = upper == lower
#         proj_distr[eq_mask, lower[eq_mask]] += next_distr[eq_mask, atom]
#         ne_mask = upper != lower
#         proj_distr[ne_mask, lower[ne_mask]] += next_distr[ne_mask, atom] * (upper - b_j)[ne_mask]
#         proj_distr[ne_mask, upper[ne_mask]] += next_distr[ne_mask, atom] * (b_j - lower)[ne_mask]
#
#     if dones.any():
#         proj_distr[dones] = 0.0
#         tz_j = np.minimum(v_max, np.maximum(v_min, rewards[dones]))
#         b_j = (tz_j - v_min) / delta_z
#         lower = np.floor(b_j).astype(np.int64)
#         upper = np.ceil(b_j).astype(np.int64)
#         eq_mask = upper == lower
#         eq_dones = dones.copy()
#         eq_dones[dones] = eq_mask
#         if eq_dones.any():
#             proj_distr[eq_dones, lower[eq_mask]] = 1.0
#         ne_mask = upper != lower
#         ne_dones = dones.copy()
#         ne_dones[dones] = ne_mask
#         if ne_dones.any():
#             proj_distr[ne_dones, lower[ne_mask]] = (upper - b_j)[ne_mask]
#             proj_distr[ne_dones, upper[ne_mask]] = (b_j - lower)[ne_mask]
#
#     return proj_distr


def agent_eval(env: gym.Env, model: tf.keras.Model, max_steps: int = 200):
    if hasattr(model, 'has_custom_noisy_layer'):
        model.deactivate_noise()
    state = env.reset()
    episode_reward = 0

    for time_step in range(1, max_steps):

        state = tf.convert_to_tensor(state)
        state = tf.expand_dims(state, 0)

        action = np.argmax(model(state))

        # Apply the sampled action in our environment
        state, reward, done, _ = env.step(action)
        episode_reward += reward

        if done:
            break

    if hasattr(model, 'has_custom_noisy_layer'):
        model.activate_noise()
    env.reset()

    return episode_reward
