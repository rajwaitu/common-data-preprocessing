from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer

def apply_standard_scaler(iterable):
    standard_scaler = StandardScaler()
    return standard_scaler.fit_transform(iterable)

def apply_minmax_scaler(iterable):
    minmax_scaler = MinMaxScaler()
    return minmax_scaler.fit_transform(iterable)

def apply_maxabs_scaler(iterable):
    maxabs_scaler = MaxAbsScaler()
    return maxabs_scaler.fit_transform(iterable)

def apply_robust_scaler(iterable):
    robust_scaler = RobustScaler()
    return robust_scaler.fit_transform(iterable)

def apply_normalizer(iterable):
    normalizer = Normalizer()
    return normalizer.fit_transform(iterable)

def apply_binarizer(iterable):
    binarizer = Binarizer()
    return binarizer.fit_transform(iterable)

def apply_imputer(iterable,strategy='mean'):
    imputer = Imputer(strategy, axis=1)
    return imputer.fit_transform(iterable)

def apply_polynomial_features(iterable,degree=3):
    pol = PolynomialFeatures(degree)
    return pol.fit_transform(iterable)

def apply_functional_transformer(mappingfunction):
    return FunctionTransformer(mappingfunction, validate=False)


