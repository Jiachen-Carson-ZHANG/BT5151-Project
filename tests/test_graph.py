from bt5151_credit_risk.graph import build_graph


def test_graph_contains_required_nodes():
    graph = build_graph()
    expected_nodes = {
        "preprocess-data",
        "train-models",
        "evaluate-models",
        "select-model",
        "run-inference",
        "explain-risk",
        "recommend-action",
    }
    assert expected_nodes.issubset(set(graph.nodes.keys()))
