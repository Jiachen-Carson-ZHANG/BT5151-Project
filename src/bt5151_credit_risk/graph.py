from bt5151_credit_risk.state import CreditRiskState

try:
    from langgraph.graph import StateGraph as _StateGraph
except ModuleNotFoundError:
    _StateGraph = None


class _FallbackGraph:
    def __init__(self, _state_type):
        self.state_type = _state_type
        self.nodes = {}

    def add_node(self, name, fn):
        self.nodes[name] = fn


def _identity_node(state):
    return state


def build_graph():
    graph = _StateGraph(CreditRiskState) if _StateGraph is not None else _FallbackGraph(CreditRiskState)
    for name in [
        "preprocess-data",
        "train-models",
        "evaluate-models",
        "select-model",
        "run-inference",
        "explain-risk",
        "recommend-action",
    ]:
        graph.add_node(name, _identity_node)
    return graph
