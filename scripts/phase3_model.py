"""
phase3_model.py
===============
Phase 3 multi-task LSTM: one shared encoder, two output heads.

    Input (batch, 60, 21)
      -> LSTM(64, return_sequences=True)
      -> Dropout(0.2)
      -> [baseline]  LSTM(32)            -> shared vector (32)
      -> [attention] TemporalAttention   -> shared vector (64)
      -> Dense(32, relu) + Dropout(0.2)  -> shared representation
      -> head A: Dense(3, softmax)   -> DOWN / FLAT / UP probabilities
      -> head B: Dense(1, linear)    -> next-session percentage return

Both variants expose the SAME input and output interfaces. TensorFlow/Keras
only; no CNN, transformer, or traditional ML estimator is used anywhere.
"""

from __future__ import annotations

import os
import random

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from phase3_config import (
    ARCHITECTURE,
    INPUT_SHAPE,
    LOSS_CONFIG,
    N_CLASSES,
    SEEDS,
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
def set_seeds() -> None:
    """Fix Python/NumPy/TensorFlow seeds before any model is created."""
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    random.seed(SEEDS["python"])
    np.random.seed(SEEDS["numpy"])
    tf.random.set_seed(SEEDS["tensorflow"])


# ---------------------------------------------------------------------------
# Optional attention candidate (additive temporal attention)
# ---------------------------------------------------------------------------
@keras.utils.register_keras_serializable(package="phase3")
class TemporalAttention(layers.Layer):
    """Learns a softmax distribution over the 60 timesteps of an LSTM output.

    Input : (batch, time, features)
    Output: (batch, features) — context vector = weighted sum over time.

    Implemented with plain Keras weights (no nested sublayers) so that a model
    containing it can be saved/loaded with no custom_objects argument.
    """

    def __init__(self, score_units: int = 8, **kwargs):
        super().__init__(**kwargs)
        self.score_units = int(score_units)

    def build(self, input_shape):
        feat_dim = int(input_shape[-1])
        self.w1 = self.add_weight(
            name="attn_w1", shape=(feat_dim, self.score_units),
            initializer="glorot_uniform", trainable=True,
        )
        self.b1 = self.add_weight(
            name="attn_b1", shape=(self.score_units,),
            initializer="zeros", trainable=True,
        )
        self.w2 = self.add_weight(
            name="attn_w2", shape=(self.score_units,),
            initializer="glorot_uniform", trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        hidden = tf.tanh(tf.tensordot(inputs, self.w1, axes=([2], [0])) + self.b1)
        scores = tf.tensordot(hidden, self.w2, axes=([2], [0]))  # (batch, time)
        weights = tf.nn.softmax(scores, axis=1)                  # (batch, time)
        context = tf.reduce_sum(inputs * weights[..., tf.newaxis], axis=1)
        return context

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        config = super().get_config()
        config.update({"score_units": self.score_units})
        return config


# ---------------------------------------------------------------------------
# Model builders
# ---------------------------------------------------------------------------
def build_multitask_lstm(
    variant: str = "baseline",
    n_features: int = INPUT_SHAPE[1],
    seq_len: int = INPUT_SHAPE[0],
    name: str | None = None,
) -> keras.Model:
    """Build (uncompiled) multi-task LSTM. variant: 'baseline' | 'attention'."""
    if variant not in ("baseline", "attention"):
        raise ValueError("variant must be 'baseline' or 'attention'")

    inputs = keras.Input(shape=(seq_len, n_features), name="sequence_input")

    x = layers.LSTM(
        ARCHITECTURE["lstm_1_units"], return_sequences=True, name="lstm_1"
    )(inputs)
    x = layers.Dropout(ARCHITECTURE["dropout_rate"], name="dropout_1")(x)

    if variant == "attention":
        x = TemporalAttention(
            ARCHITECTURE["attention_score_units"], name="temporal_attention"
        )(x)
    else:
        x = layers.LSTM(ARCHITECTURE["lstm_2_units"], name="lstm_2")(x)

    shared = layers.Dense(
        ARCHITECTURE["shared_dense_units"],
        activation="relu",
        kernel_regularizer=keras.regularizers.l2(ARCHITECTURE["l2_reg"]),
        name="shared_dense",
    )(x)
    shared = layers.Dropout(ARCHITECTURE["dropout_rate"], name="dropout_2")(shared)

    classification = layers.Dense(
        N_CLASSES, activation="softmax", name="classification"
    )(shared)
    regression = layers.Dense(
        1, activation="linear", name="regression"
    )(shared)

    return keras.Model(
        inputs=inputs,
        outputs={"classification": classification, "regression": regression},
        name=name or f"phase3_multitask_lstm_{variant}",
    )


def compile_model(model: keras.Model) -> keras.Model:
    """Compile with sparse CE + Huber and explicit loss weights."""
    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=LOSS_CONFIG["learning_rate"], name="adam"
        ),
        loss={
            "classification": keras.losses.SparseCategoricalCrossentropy(),
            "regression": keras.losses.Huber(delta=ARCHITECTURE["huber_delta"]),
        },
        loss_weights=dict(LOSS_CONFIG["loss_weights"]),
        metrics={
            "classification": [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy")
            ],
            "regression": [keras.metrics.MeanAbsoluteError(name="mae")],
        },
    )
    return model


def build_and_compile_model(variant: str = "baseline") -> keras.Model:
    """Convenience: seed RNGs, build and compile a Phase 3 model."""
    set_seeds()
    return compile_model(build_multitask_lstm(variant=variant))


def model_signature(model: keras.Model) -> dict:
    """Return a serializable description of interfaces and compilation."""
    return {
        "input_shape": list(model.input_shape),
        "output_shapes": {
            name: list(tensor.shape)
            for name, tensor in zip(model.output_names, model.outputs)
        },
        "total_params": int(model.count_params()),
        "trainable_params": int(
            sum(np.prod(w.shape) for w in model.trainable_weights)
        ),
        "loss": {
            "classification": "sparse_categorical_crossentropy",
            "regression": f"huber(delta={ARCHITECTURE['huber_delta']})",
        },
        "loss_weights": dict(LOSS_CONFIG["loss_weights"]),
        "optimizer": "adam",
        "learning_rate": float(LOSS_CONFIG["learning_rate"]),
        "batch_size": None,  # filled by the training entry point
    }
