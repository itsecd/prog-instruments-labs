import pytest
import os

from unittest.mock import patch, MagicMock

from fiona.features import vertex_count

import graph
import save_and_load_graph


def test_create():
    _graph = graph.Graph()

    assert _graph is not None


def test_add_vertex():
    _graph = graph.Graph()
    vertex = 'A'
    _graph.add_vertex(vertex)

    assert vertex in _graph._graph


def test_has_edge():
    _graph = graph.Graph()

    _graph.add_vertex("A")
    _graph.add_vertex("B")

    _graph.add_edge("A", "B", 2.0)

    assert _graph.has_edge("A", "B")
    assert _graph.has_edge("B", "A")


def test_remove_vertex():
    _graph = graph.Graph()
    vertex = 'A'
    _graph.add_vertex(vertex)
    _graph.remove_vertex(vertex)

    assert (vertex in _graph._graph) == False


def test_remove_edge():
    _graph = graph.Graph()

    _graph.add_vertex("A")
    _graph.add_vertex("B")
    _graph.add_vertex("C")

    _graph.add_edge("A", "B", 2.0)
    _graph.add_edge("A", "C", 5.0)

    _graph.remove_edge("A", "B")

    assert _graph.has_edge("A", "B") == False
    assert _graph.has_edge("B", "A") == False

    assert _graph.has_edge("A", "C")
    assert _graph.has_edge("C", "A")


@pytest.fixture(autouse=True)
def create_graph():
    _graph = graph.Graph()

    _graph.add_vertex("A")
    _graph.add_vertex("B")
    _graph.add_vertex("C")
    _graph.add_vertex("D")

    _graph.add_edge("A", "B", 2.0)
    _graph.add_edge("B", "C", 17.0)
    _graph.add_edge("A", "C", 15.7805)
    _graph.add_edge("D", "C", 7.0)
    return _graph


def test_vertices(create_graph):
    vertices = create_graph.vertices()
    assert vertices == ["A", "B", "C", "D"]


def test_edges(create_graph):
    edges = create_graph.edges()
    assert edges == [('A', 'B', 2.0), ('A', 'C', 15.7805), ('B', 'A', 2.0), ('B', 'C', 17.0), ('C', 'B', 17.0),
                     ('C', 'A', 15.7805), ('C', 'D', 7.0), ('D', 'C', 7.0)]


@pytest.mark.parametrize('task_for_walk', [
    ["A", [('A', 0), ('B', 1), ('C', 1), ('D', 2)]],
    ["B", [('B', 0), ('A', 1), ('C', 1), ('D', 2)]],
    ["C", [('C', 0), ('B', 1), ('A', 1), ('D', 1)]],
    ["D", [('D', 0), ('C', 1), ('B', 2), ('A', 2)]],
])
def test_breadth_search(create_graph, task_for_walk):
    result_of_walking = create_graph.breadth_search(task_for_walk[0])
    assert result_of_walking == task_for_walk[1]

@pytest.mark.parametrize('task_for_dijkstra', [
    ["D", "A", {'A': 22.7805, 'B': 24.0, 'C': 7.0, 'D': 0} , ['D', 'C', 'A']] ,
    ["A", "C", {'A': 0, 'B': 2.0, 'C': 15.7805, 'D': 22.7805}, ['A', 'C']],
    ["C", "B", {'A': 15.7805, 'B': 17.0, 'C': 0, 'D': 7.0}, ['C', 'B']],
    ["D", "B", {'A': 22.7805, 'B': 24.0, 'C': 7.0, 'D': 0}, ['D', 'C', 'B']],
])
def test_dijkstra(create_graph, task_for_dijkstra):
    distance, path = create_graph.dijkstra(task_for_dijkstra[0], task_for_dijkstra[1])
    assert distance == task_for_dijkstra[2]
    assert path == task_for_dijkstra[3]


