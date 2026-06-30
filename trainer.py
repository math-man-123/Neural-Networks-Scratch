# pyright: reportMissingImports=false, reportUndefinedVariable=false

import numpy as np
import copy, random, time, random
from typing import Any
from neural import NeuralNetwork
from data_loader import DataLoader
from augmentor import Augmentor
from schedule import LearningSchedule
from loss import LossFunction
from optimizer import Optimizer
from custom_types import * 


class Trainer:
    """
    Trainer class handels all needed components to train a neural network.
    Use this class to create a neural network from given training data.
    """
    def __init__(
            self,
            network_config: dict[str, Any], 
            activation_config: dict[str, Any],
            initializer_config: dict[str, Any],
            data_config: dict[str, Any],
            augment_config: dict[str, Any],
            learning_config: dict[str, Any],
            loss_config: dict[str, Any],
            optimizer_config: dict[str, Any],
            trainer_config: dict[str, Any]
        ) -> None:
        """
        Create a new Trainer instance from given configs as listed.

        :param network_config:
            -> name, layer_sizes

        :param activation_config:
            -> name, optional params

        :param initializer_config:
            -> weight_config (name, optional params)
            -> bias_config (name, optional params)
        
        :param data_config:
            -> loader_config (name, optional params)
            -> train_data_path, test_data_path

        :param augment_config:
            -> name, optional params

        :param learning_config:
            -> schedule_config (low, high, frac)
            -> warmup_config (name, optional params)
            -> decay_config (name, optional params)

        :param loss_config:
            -> name, optional params

        :param optimizer_config:
            -> name, optional params

        :param trainer_config:
            -> batch_size, max_batch, test_after, max_epoch, augment_num
        """
        # save full config to bundle with finished net
        self._train_configs = {
            "network_config": network_config,
            "activation_config": activation_config,
            "initializer_config": initializer_config,
            "data_config": data_config,
            "augment_config": augment_config,
            "learning_config": learning_config,
            "loss_config": loss_config,
            "optimizer_config": optimizer_config,
            "trainer_config": trainer_config,
        }

        # setup specific trainer params
        self._batch_num = 0
        self._batch_size = trainer_config["batch_size"]
        self._max_batch = trainer_config["max_batch"]
        self._test_after = trainer_config["test_after"]
        self._max_epoch = trainer_config["max_epoch"]
        
        # calculate total training batches
        self._total = self._max_batch  * self._max_epoch

        # copy learning config and add total step number
        learning_config = copy.deepcopy(learning_config)
        learning_config["schedule_config"]["total"] = self._total

        # setup neural network 
        self._network = NeuralNetwork(
            **network_config, 
            activation_config=activation_config,
            initializer_config=initializer_config)

        # copy data config so not to make any changes
        data_config = copy.deepcopy(data_config)

        # setup data loader
        loader_config = data_config.pop("loader_config")
        self._data_loader = DataLoader(loader_config)

        #setup data augmentor
        self._augmentor = Augmentor(augment_config)

        # load training data and create augment 
        train_data_path = data_config.pop("train_data_path")
        data = self._data_loader.load(train_data_path)
        
        augment = []
        for _ in range(trainer_config["augment_num"]):
            augment.append(self._augmentor.create(data))
        
        # combine data and shuffle it
        combined_data = data + [x for sub in augment for x in sub]
        random.shuffle(combined_data)
        self._train_data = combined_data

        # load test data
        test_data_path = data_config.pop("test_data_path")
        self._test_data = self._data_loader.load(test_data_path)

        # setup learning schedule
        self._learning_schedule = LearningSchedule(**learning_config)

        # setup loss function
        self._loss = LossFunction(
            loss_config=loss_config,
            activation_config=activation_config)
        
        #setup optimizer
        self._optimizer = Optimizer(optimizer_config)

    
    def _test_accuracy(self, data_name: str = "test") -> float:
        """
        Test current network accuracy on dataset with given name.
        A answer is counted correct if the element with the highest
        evaluation in the output vector of the network mathes the
        target element. 
        
        :param data_name: "test" or "train"
        :return: accuracy as a percentage value
        """
        dataset = getattr(self, f"_{data_name}_data")

        correct = 0
        for input, target in dataset:
            # get output from current network based on input
            output, _ = self._network.query(input)

            # check if digits with highest evaluation match
            if np.argmax(output) == np.argmax(target):
                correct += 1

        return correct / len(dataset)
    

    def _shuffle_data(self, data_name: str = "train") -> None:
        """
        Shuffle dataset with given name.
        
        :param data_name: "test" or "train"
        """
        random.shuffle(getattr(self, f"_{data_name}_data"))


    def _get_batch_gradients(self) -> tuple[list[Matrix], list[Vector]]:
        """
        Calculate the next batch gradients from the training data.
        Automatically selects a new batch each time it is called
        and shuffles training data once full epoch was trained.
        
        :return: averaged batch gradients (weight and bias)
        """
        start = self._batch_num * self._batch_size
        end = min(start + self._batch_size, len(self._train_data))
        batch = self._train_data[start : end]

        # initialize accumulators with zero
        depth = self._network.depth
        weight_batch_grads = [0] * depth
        bias_batch_grads = [0] * depth

        # grab weights once as they only change in between batches
        weights = self._network.weights
        for input, target in batch:
            # get weight and bias gradients
            output, signals = self._network.query(input)
            weight_grads, bias_grads = self._loss.get_gradients(
                signals, weights, output, target)

            # accumulate weight and bias gradients
            weight_batch_grads = [
                acc + grad for acc, grad 
                in zip(weight_batch_grads, weight_grads)]
            bias_batch_grads = [
                acc + grad for acc, grad
                in zip(bias_batch_grads, bias_grads)]
            
        # batch size might differ from self._batch_size
        batch_size = len(batch) 
        weight_batch_grads = [
            grad / batch_size for grad in weight_batch_grads]
        bias_batch_grads = [
            grad / batch_size for grad in bias_batch_grads]

        # if final batch was reached reset counter and shuffle
        self._batch_num = (self._batch_num + 1) % self._max_batch
        if self._batch_num == 0: self._shuffle_data()

        return weight_batch_grads, bias_batch_grads


    def _update_network(self):
        """
        Update network parameter by a single batch step.
        Repeatably calling this method trains the network.
        """
        # get weight and bias steps from batched optimizer
        weight_grads, bias_grads = self._get_batch_gradients()
        weight_steps, bias_steps = self._optimizer.get_steps(
            weight_grads, bias_grads, *self._network.params
        )
        
        # moderated update of each weight matrix and bias vector
        learning_rate = self._learning_schedule.get_next_rate()
        param_steps = zip(weight_steps, bias_steps)
        for idx, (weight_step, bias_step) in enumerate(param_steps):
            self._network.weights[idx] -= learning_rate * weight_step
            self._network.biases[idx] -= learning_rate * bias_step


    def train(self) -> None:
        """
        Train a neural network as described in given configs.
        """
        accuracy = [self._test_accuracy()]
        print(f"starting training network {self._network.name}")
        print(f"initial accuracy: {accuracy[0]}")

        total = self._total
        max_batch = self._max_batch
        max_epoch = self._max_epoch

        delta_time = 0.0
        epoch = 0
        last_delta = 0.0  # delta vs previous epoch

        for batch in range(total):
            start_time = time.monotonic()
            self._update_network()

            # calc accuracy after each given batch number
            if batch % self._test_after == 0:
                acc = self._test_accuracy()
                accuracy.append(acc)
                last_delta = accuracy[-1] - accuracy[-2]
            
            if batch % self._max_batch == 0 and batch != 0:
                epoch += 1

            end_time = time.monotonic()
            delta_time += (end_time - start_time)
            avg = delta_time / (batch + 1)
            remaining_time = avg * (total - (batch + 1))

            epoch_msg = f"epoch {min(epoch + 1, max_epoch)}/{max_epoch}"
            batch_msg = f"batch {batch % max_batch + 1}/{max_batch}"
            update_msg = f"progress: {epoch_msg} {batch_msg}"

            # show the most recently known epoch accuracy
            accuracy_msg = f"last test accuracy: {accuracy[-1]} (+{last_delta:.4f})"
            time_msg = f"estimated remaining time: {round(remaining_time)} sec"

            print(f"\r\033[K{update_msg}\n\033[K{accuracy_msg}\n\033[K{time_msg}", end="", flush=True)
            print("\033[2A", end="", flush=True)

        final_delta = accuracy[-1] - accuracy[0]
        accuracy_msg = f"last test accuracy: {accuracy[-1]} (+{final_delta:.4f})"
        print(f"\n\033[K{accuracy_msg}\n\033[Saving final model data as zip")

        self._network.save_data(train_accuracy=accuracy, train_configs=self._train_configs)
