from src.training.classification import Trainer as ClassificationTrainer

class_trainer = ClassificationTrainer(
    features_file = "features_clean.csv"
)

results = class_trainer.train()
class_trainer.report(results)