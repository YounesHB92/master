from src.features import FeatureExtractor
from pprint import pprint

feature_ex_ = FeatureExtractor(print_=False, save=True, saving_name="features")
feature_ex_.run_all()