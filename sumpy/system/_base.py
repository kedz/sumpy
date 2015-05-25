from sumpy.annotators._annotator_base import _AnnotatorBase
from abc import ABCMeta, abstractmethod
import pandas as pd
import numpy as np
import networkx as nx

class _SystemBase(object):
    """Abstract base class for summarizer systems."""

    __metaclass__ = ABCMeta

    def __init__(self, verbose=False):
        self.verbose = verbose
        self._dependency_graph = None
        self._annotators = None
        self._pipeline = None

    @abstractmethod
    def build_summary(self, input_df, ndarray_data):
        pass

    def summarize(self, inputs):

        if not hasattr(self, "_pipeline") or self._pipeline is None:
            self.build_pipeline()

        input_df, ndarray_data = self.prepare_inputs(inputs)
        processed_df, processed_ndarray_data = self.process_input(
            input_df, ndarray_data)
         
        return self.build_summary(processed_df, processed_ndarray_data)

    def build_pipeline(self):
        self.build_dependency_graph()
        self._pipeline = []
        for node in nx.topological_sort(self._dependency_graph):
            if node in self._annotators:
                self._pipeline.append(self._annotators[node])
                if self.verbose:
                    print "{} ({}) build".format(self.__class__.__name__,
                        self._annotators[node].name(self))
                self._annotators[node].build(self)

    def prepare_inputs(self, inputs, ndarray_data=None):
        
        requires = set()
        returns = set()
        ndarray_requires = set()
        ndarray_returns = set()

        for ann in self._pipeline:
            requires.update(ann.requires(self))
            returns.update(ann.returns(self))
            ndarray_requires.update(ann.ndarray_requires(self))
            ndarray_returns.update(ann.ndarray_returns(self))
        
        # Allocate keys for ndarray dependencies.
        if ndarray_data is None:
            ndarray_data = {}
        for key in ndarray_requires.union(ndarray_returns):
            if key not in ndarray_data:
                ndarray_data[key] = None

        # Allocate columns for dataframe data dependencies.
        all_cols = list(requires.union(returns))
        if isinstance(inputs, list) or isinstance(inputs, tuple):
            df = pd.DataFrame([{"doc id": doc_id, "doc text": doc_text}
                                 for doc_id, doc_text in enumerate(inputs)],
                                columns=["doc id"] + all_cols)
           # df.set_index("doc id", inplace=True)
            return df, ndarray_data
        #elif isinstance(inputs, pd.DataFrame):
            

        else:
            raise Exception("Bad input: list of strings or dataframe only.")

    def process_input(self, input_df, ndarray_data):
        cols = set(input_df.columns.tolist())
        for ann in self._pipeline:
            
            for rtype in ann.returns(self):
                assert rtype in cols
            
            for req in ann.requires(self):
                assert req in cols
            
            run_stage = input_df[ann.returns(self)].isnull().any().any() \
                or np.any([ndarray_data[rtype] is None
                           for rtype in ann.ndarray_returns(self)])

            if run_stage:

                if self.verbose:
                    print "{} ({}) process".format(
                        self.__class__.__name__, ann.name(self))

                input_df, ndarray_data = ann.process(
                    self, input_df, ndarray_data)

        return input_df, ndarray_data

    def build_dependency_graph(self):
        G = nx.DiGraph()
        self._annotators = {}

        def check_mixins(clazz, visited=set()):
            if not issubclass(clazz, _SystemBase):
                if issubclass(clazz, _AnnotatorBase):
                    name = clazz.name(self)
                    self._annotators[name] = clazz
                    for req in clazz.requires(self):
                        G.add_edge(req, name)
                    for req in clazz.ndarray_requires(self):
                        G.add_edge(req, name)

                    for rtype in clazz.returns(self):
                        G.add_edge(name, rtype)
                    for rtype in clazz.ndarray_returns(self):
                        G.add_edge(name, rtype)

            visited.add(clazz)
            for base in clazz.__bases__:
                if base in visited:
                    continue
                if not issubclass(base, _AnnotatorBase):
                    continue
                if base == _AnnotatorBase:
                    continue
                check_mixins(base, visited)

        check_mixins(self.__class__) 
        self._dependency_graph = G

    def print_dependency_graph(self, filename=None, to_iPython=True):
        import pygraphviz as pgv
        if not hasattr(self, "_dependency_graph") or \
                self._dependency_graph is None:
            self.build_dependency_graph()

        if filename is None:
            filename = "sumpy.tmp.png"

        G = pgv.AGraph(strict=False, directed=True) 
        for node in self._dependency_graph:
            if node in self._annotators:
                G.add_node(node)
                G.get_node(node).attr["shape"] ="rectangle"
            elif node.startswith("f:"):
                G.add_node(node)
                G.get_node(node).attr["shape"] ="parallelogram"
                for edge in self._dependency_graph.in_edges(node):
                    G.add_edge(edge[0], edge[1], color="green")
            else:
                for in_edge in self._dependency_graph.in_edges(node):
                    for out_edge in self._dependency_graph.out_edges(node):
                        G.add_edge(in_edge[0], out_edge[1], 
                                   label=node, key=node)

        G.layout("dot")
        G.draw(filename)
        if to_iPython is True:
            from IPython.display import Image 
            return Image(filename=filename)
