import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model


class GradCAM:
    def __init__(self, model, target_layer_name):
        self.model = model
        self.target_layer_name = target_layer_name
        self.last_conv_layer = self._get_layer(target_layer_name)
        self.grad_model = Model(
            inputs=self.model.input,
            outputs=[self.model.output, self.last_conv_layer.output],
        )

    def _get_layer(self, layer_name):
        for layer in reversed(self.model.layers):
            if layer.name == layer_name:
                return layer

        for layer in reversed(self.model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                return layer

        raise ValueError(f"Layer {layer_name} not found and no Conv2D layer is available")

    def generate_heatmap(self, image, class_index=None):
        image = np.expand_dims(image, axis=0)
        with tf.GradientTape() as tape:
            preds, conv_outputs = self.grad_model(image)
            if class_index is None:
                class_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, class_index]

        grads = tape.gradient(class_channel, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
        heatmap = tf.maximum(heatmap, 0)
        max_value = tf.reduce_max(heatmap)
        heatmap = tf.where(max_value > 0, heatmap / max_value, heatmap)
        return heatmap.numpy()

    def overlay_heatmap(self, image, heatmap, alpha=0.55):
        image = np.asarray(image).astype("float32")
        if image.ndim != 3:
            raise ValueError("Expected an RGB image with shape (H, W, C)")
        if image.max() > 1.0:
            image = image / 255.0
        heatmap = np.asarray(heatmap).astype("float32")
        heatmap = np.clip(heatmap, 0.0, 1.0)

        if heatmap.shape[:2] != image.shape[:2]:
            from PIL import Image as PILImage

            heatmap_image = PILImage.fromarray(np.uint8(255 * heatmap))
            heatmap_image = heatmap_image.resize((image.shape[1], image.shape[0]), PILImage.Resampling.LANCZOS)
            heatmap = np.asarray(heatmap_image).astype("float32") / 255.0

        from matplotlib import cm

        color_mapped = cm.jet(heatmap)[:, :, :3]
        overlay = image * (1 - alpha) + color_mapped * alpha
        return np.clip(overlay, 0.0, 1.0)
