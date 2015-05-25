from abc import ABCMeta, abstractmethod

class _AnnotatorBase(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def requires(self):
        pass
    
    @abstractmethod
    def ndarray_requires(self):
        pass

    @abstractmethod
    def returns(self):
        pass

    @abstractmethod
    def ndarray_returns(self):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def build(self):
        pass

    @abstractmethod
    def process(self, input_df, ndarray_data):
        pass
