from tensorflow.keras.applications import EfficientNetB0, MobileNetV2, ResNet50, VGG19
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.losses import BinaryCrossentropy, SparseCategoricalCrossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

SUPPORTED_MODELS = {
    "efficientnetb0": EfficientNetB0,
    "mobilenetv2": MobileNetV2,
    "resnet50": ResNet50,
    "vgg19": VGG19
}


def get_model_head_config(num_classes):
    if num_classes <= 2:
        return {
            "units": 1,
            "activation": "sigmoid",
            "loss": "binary",
        }

    return {
        "units": num_classes,
        "activation": "softmax",
        "loss": "sparse_categorical",
    }


def build_transfer_model(model_name, num_classes, input_shape=(224, 224, 3)):
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model '{model_name}'. Choose from: {sorted(SUPPORTED_MODELS)}")

    head_config = get_model_head_config(num_classes)
    base_class = SUPPORTED_MODELS[model_name]
    base_model = base_class(weights=None, include_top=False, input_shape=input_shape)
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation="relu")(x)
    predictions = Dense(head_config["units"], activation=head_config["activation"])(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    loss = BinaryCrossentropy() if head_config["loss"] == "binary" else SparseCategoricalCrossentropy()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss=loss,
        metrics=["accuracy"],
    )
    return model


class FederatedClient:
    def __init__(self, client_id, server_address, model_name="efficientnetb0", num_classes=None):
        self.client_id = client_id
        self.server_address = server_address
        self.model_name = model_name
        self.num_classes = num_classes or 2
        self.model = self.build_model(self.model_name, self.num_classes)

    def build_model(self, model_name, num_classes):
        return build_transfer_model(model_name, num_classes)

    def connect_to_server(self):
        print(f"Client {self.client_id} connecting to server at {self.server_address}")

    def train_model(self, train_data, val_data, epochs=5, batch_size=32, callbacks=None):
        train_images, train_labels = train_data
        val_images, val_labels = val_data

        train_images = preprocess_input(train_images)
        val_images = preprocess_input(val_images)

        history = self.model.fit(
            train_images,
            train_labels,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_images, val_labels),
            callbacks=callbacks or [],
            verbose=0,
        )
        print(f"Client {self.client_id} finished training the model")
        return history.history

    def evaluate_model(self, test_data):
        test_images, test_labels = test_data
        test_images = preprocess_input(test_images)

        loss, accuracy = self.model.evaluate(test_images, test_labels, verbose=0)
        print(f"Client {self.client_id} model evaluation - Loss: {loss}, Accuracy: {accuracy}")
        return accuracy

    def send_model_updates(self):
        weights = self.model.get_weights()
        print(f"Client {self.client_id} sending model updates to server")
        return weights