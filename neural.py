# pyright: reportMissingImports=false, reportUndefinedVariable=false

import json, io
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path
from typing import Any
import validation as val
from custom_types import * 
from initializer import Initializer
from activation import Activation


class NeuralNetwork:
    """
    Create a custom NeuralNetwork with given layer_sizes.
    Accepts custom initializer and activation configs.
    Contains query (fowards pass) and load/save code.
    """
    _default_layer_sizes = (3, 3, 3)
    _default_activation_config = Activation._default_config
    _default_initializer_config = { 
        "weight_config": Initializer._default_weight_config,
        "bias_config": Initializer._default_bias_config }


    def __init__(
            self, 
            name: str = "unnamed",
            layer_sizes: LayerSize = _default_layer_sizes, 
            activation_config: dict[str, Any] = _default_activation_config,
            initializer_config: dict[str, dict[str, Any]] = _default_initializer_config, 
        ) -> None: 
        """
        Create a new NeuralNetwork instance from given layer_sizes.
        Optionally use custom Initializer and activation function.
        
        :param layer_sizes: number of neurons in each layer
        :param initializer: Initializer to first fill weights and biases
        :param activation: activation function of each non-input neuron
        """
        # ensure all layer sizes are greater 0
        val.check_condition(
            all(size > 0 for size in layer_sizes),
            "layer sizes must be greater 0")
        
        # make activation and initializer from configs
        activation = Activation(activation_config)
        initializer = Initializer(**initializer_config)

        # initialize weights and biases
        self.weights = initializer.get_weights(layer_sizes)
        self.biases = initializer.get_biases(layer_sizes)

        # save name and layer size and activation
        self.name = name # public access
        self._layer_sizes = layer_sizes
        self._activation = activation


    def query(self, input: Vector) -> tuple[Vector, dict[str, list[Vector]]]:
        """
        Forward pass input signal through the network.
        Signal x^(k) evolves according to 
                x^(k) = sigma(W^(k)x^(k-1) + b^(k))
        where sigma is the activation function.        

        :param input: input signal (same size as input layer)
        :return: output signal (same size as output layer)
        :return: all other pre and post signals als dict
        """
        # ensure input signal has correct size 
        val.check_condition(
            input.shape == (self._layer_sizes[0], 1),
            "input must be vector with size of input layer")
        
        # first layer no activation hence pre and 
        # post signals are the same z^(0) = x^(0)
        pre_signal = post_signal = input
        signals = { "pre": [pre_signal], "post": [post_signal] }

        # forward pass signal through the network
        for weight, bias in zip(self.weights, self.biases):
            pre_signal = weight @ post_signal + bias
            post_signal = self._activation.eval(pre_signal)

            signals["pre"].append(pre_signal)
            signals["post"].append(post_signal)

        output = signals["post"][-1]
        return output, signals

    @property
    def params(self) -> tuple[list[Matrix], list[Vector]]: 
        """Get the weights and biases of this NeuralNetwork"""
        return self.weights, self.biases

    @property
    def depth(self) -> int: 
        """Get the number of layers (depth) of this NeuralNetwork."""
        return len(self._layer_sizes)


    def save_data(
            self,
            train_accuracy: list[int],
            train_configs: dict[dict[str, Any]]
        ) -> None:
        """
        Save compressed parameter file for the neural network.
        Weights and biases are indexed as W^(k) and b^(k).
        The layer sizes and the activation config of the net
        are saved as metadata. Both files are then zipped.

        :param train_accuracy: 
            list of accuracies after each training epoch
            to be optionally saved as train_accuracy.csv
        """
        base_dir = Path(__file__).resolve().parent
        zip_path = base_dir / f"{self.name}.zip"

        # save indices with 3 padded zeros for correct sort
        weights = {f"W^({k:03d})": W for k, W in enumerate(self.weights)}
        biases  = {f"b^({k:03d})": b for k, b in enumerate(self.biases)}

        metadata = {
            "name": self.name,
            "layer_sizes": self._layer_sizes,
            "activation_config": self._activation.config
        }

        with ZipFile(zip_path, "w", compression=ZIP_DEFLATED) as zip:
            # save params to buffer then zip it
            npz_buffer = io.BytesIO()
            np.savez_compressed(npz_buffer, allow_pickle=False, **weights, **biases)
            zip.writestr("parameters.npz", npz_buffer.getvalue())

            # metadata and configs can be ziped directly
            zip.writestr("metadata.json", json.dumps(metadata))
            zip.writestr("train_configs.json", json.dumps(train_configs))

            # save accuracies to buffer then zip it
            csv_buffer = io.StringIO()
            np.savetxt(csv_buffer, train_accuracy, delimiter=",")
            zip.writestr("train_accuracy.csv", csv_buffer.getvalue())


    @classmethod
    def from_data(cls, zip_path: str) -> "NeuralNetwork":
        """
        Create a new NeuralNetwork instance from model.zip.
        It should contain metadata.json and parameters.npz.
        
        :param zip_path: path to model.zip
        :return: NeuralNetwork as described
        """
        base_dir = Path(__file__).resolve().parent

        with ZipFile(base_dir / zip_path) as zip:
            # params have to be read to buffer first
            with zip.open("parameters.npz") as file:
                buffer = io.BytesIO(file.read())
                params = np.load(buffer, allow_pickle=False)

            # metadata can be read direcly
            with zip.open("metadata.json") as file:
                metadata = json.load(file)

        # extract weights and biases into ordered lists
        weights = [params[key] for key in sorted(params.files) if key.startswith("W")]
        biases = [params[key] for key in sorted(params.files) if key.startswith("b")]

        initializer_config = { 
            "weight_config": { "name": "copy", "arrays": weights },
            "bias_config": { "name": "copy", "arrays": biases }}

        return NeuralNetwork(
            metadata["name"], metadata["layer_sizes"], 
            metadata["activation_config"], initializer_config)
