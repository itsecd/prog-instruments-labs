import json

import graph


def save_graph_to_json(graph: object, filename: str):
    """
    Функция сохраняет граф в формате JSON.
    :param graph: Объект графа.
    :param filename: Имя файла для сохранения.
    """
    graph_data = {
        "vertices": graph.vertices(),
        "edges": [(edge[0], edge[1], edge[2]) for edge in graph.edges()]
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(graph_data, f, ensure_ascii=False, indent=4)


def load_graph_from_json(filename: str):
    """
    Функция загружает граф из JSON.
    :param filename: Имя файла с графом.
    :return: Объект графа.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        graph_data = json.load(f)

    new_graph = graph.Graph()
    for vertex in graph_data["vertices"]:
        new_graph.add_vertex(vertex)
    for edge in graph_data["edges"]:
        new_graph.add_edge(edge[0], edge[1], edge[2])
    return new_graph
