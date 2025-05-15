from src.features import FeatureExtractor
from pprint import pprint

feature_ex_ = FeatureExtractor(print_=False, save=True, saving_name="test")
feature_ex_.run_all(stopper=10)

print(feature_ex_.features)

