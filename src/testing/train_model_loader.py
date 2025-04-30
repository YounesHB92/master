from src.utils import load_env_variables
from src.training import EpochRunner

_ = load_env_variables()

class TrainModelLoader(EpochRunner):
    def __init__(self, model, loss_metrics, device, test_loader, config_name):
        super().__init__(model, loss_metrics, device)
        self.test_loader = test_loader
        self.config_name = config_name

    def test(self):
        test_loss, test_metrics = self._run_one_epoch(self.test_loader, train=False)
        print(f"\tTest Loss: {test_loss:.4f}")
        _, _ = self._print_metrics(test_metrics, indent_level=1)