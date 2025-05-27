import segmentation_models_pytorch as smp


class SegmentationModel:
    def __init__(self, model_name, encoder_name, encoder_weights, num_classes, activation=None):
        self.model_name = model_name
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.num_classes = num_classes
        self.activation = activation

    def load_model(self):
        ModelClass = getattr(smp, self.model_name)
        model = ModelClass(
            encoder_name=self.encoder_name,
            encoder_weights=self.encoder_weights,
            classes=self.num_classes,
            activation=self.activation
        )
        return model
