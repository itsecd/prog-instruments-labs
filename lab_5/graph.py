import csv
import math
import openrouteservice

from collections import deque
from queue import PriorityQueue

from shapely import distance

import save_and_load_graph as sl


class Graph:
    class Edge:
        def __init__(self, vertex_to: str, weight: float) -> None:
            """
            Конструктор для ребра графа.
            :param vertex_to: Метка до какой вершины есть путь
            :param weight: Вес пути
            """
            self._vertex_to = vertex_to
            self._weight = weight

    def __init__(self) -> None:
        """
        Констурктор для графа. Граф инициализируется пустым словарём.
        """
        self._graph = {}

    def add_vertex(self, vertex: str) -> None:
        """
        Метод проверяет есть ли вершина в графе, если вершина существует, то метод сообщает об этом пользователю,
        если же не существует, то создает в словаре новый элемент с уникальным именем - vertex.
        :param vertex: Метка вершины
        """
        if vertex not in self._graph:
            self._graph[vertex] = []
        else:
            print(f"Вершина {vertex} уже существует.")

    def remove_vertex(self, vertex) -> None:
        """
        Метод проверяет есть ли вершина в графе, если вершина не существует, то метод сообщает об этом пользователю,
        если же существует, то метод удаляет элемент с меткой вершины(vertex), а затем проверяет наличие ребер
        из других вершин в удаленную и удаляет эти ребра, если все же они есть.
        :param vertex: Метка вершины
        """
        if vertex in self._graph:
            for edge in self._graph[vertex]:
                self.remove_edge(vertex, edge._vertex_to)
            del self._graph[vertex]
        else:
            print(f"Вершины {vertex} нет в графе.")

    def has_edge(self, vertex_from: str, vertex_to: str)->bool:
        """
        Функция возвращает True, если ребра нет в графе и False, если есть.
        :param vertex_from: Вершина, из которой начинается ребро.
        :param vertex_to: Вершина, в которой заканчивается ребро.
        :param weight: Вес пути
        :return: Булевое значение
        """
        if vertex_from in self._graph and vertex_to in self._graph:
            for edge in self._graph[vertex_from]:
                if edge._vertex_to == vertex_to:
                    return True
            return False
        if vertex_from not in self._graph:
            print(f"Вершины {vertex_from} нет в графе.")
        else:
            print(f"Вершины {vertex_to} нет в графе.")

    def add_edge(self, vertex_from: str, vertex_to: str, weight) -> None:
        """
        Метод добавляет два ребра, состоящие из вершины vertex_from в вершину vertex_to и из vertex_to
        в vertex_from, в граф.
        :param vertex_from: Метка вершины, из которой выходит ребро.
        :param vertex_to: Метка вершины, в которой заканчивается ребро.
        :param weight: Вес пути
        """
        if vertex_from in self._graph and vertex_to in self._graph and self.has_edge(vertex_from, vertex_to) == False:
            self._graph[vertex_from].append(self.Edge(vertex_to, weight))
            self._graph[vertex_to].append(self.Edge(vertex_from, weight))
            return

        if vertex_from not in self._graph:
            print(f"Вершины {vertex_from} нет в графе.")
        else:
            print(f"Вершины {vertex_to} нет в графе.")

    def remove_edge(self, vertex_from: str, vertex_to: str) -> None:
        """
         Метод удаляет ребро, состоящие из вершины vertex_from в вершину vertex_to, из графа, путем удаления
         списка из элемента по метке vertex_from. Если такого ребра нет, то метод сообщает об этом пользователю.
        :param vertex_from: Метка вершины, из которой выходит ребро
        :param vertex_to: Метка вершины, в которою заканчивается ребро
        """
        if (vertex_from in self._graph) and (vertex_to in self._graph):
            self._graph[vertex_from] = [edge for edge in self._graph[vertex_from] if edge._vertex_to != vertex_to]
            self._graph[vertex_to] = [edge for edge in self._graph[vertex_to] if edge._vertex_to != vertex_from]
        else:
            print(f"Ребра из {vertex_from} в {vertex_to} нет в графе.")

    def vertices(self) -> list[str]:
        """
        Метод возвращяет список всех вершин графа.
        :return: Список вершин
        """
        return list(self._graph.keys())

    def edges(self) -> list[tuple]:
        """
        Метод возвращяет список всех ребер, существующие в графе.
        :return: Список ребер
        """
        edges_list = []
        for vertex_from, edges in self._graph.items():
            for edge in edges:
                edges_list.append((vertex_from, edge._vertex_to, edge._weight))
        return edges_list

    def breadth_search(self, vertex_start: str) -> list:
        """
        Метод проводит поиск в глубину для вершины vertex_start.
        :param vertex_start: Вершина для которой происходит поиск в глубину
        :return result_of_walking: Список, содержащий списки типа: "вершина", расстояние до вершины
        """
        result_of_walking = []

        walked = {vertex: {"color": "white", "distance": None, "prev": None} for vertex in self.vertices()}
        v_queue = deque()

        v_queue.append(vertex_start)
        walked[vertex_start]["color"] = "grey"
        walked[vertex_start]["distance"] = 0
        result_of_walking.append((vertex_start, 0))

        while v_queue:
            u = v_queue.popleft()
            for edge in self._graph[u]:
                vert = edge._vertex_to
                if walked[vert]["color"] == "white":
                    walked[vert]["color"] = "grey"
                    walked[vert]["prev"] = u
                    walked[vert]["distance"] = walked[u]["distance"] + 1
                    result_of_walking.append((vert, walked[vert]["distance"]))
                    v_queue.append(vert)
            walked[u]["color"] = "black"

        return result_of_walking

    def dijkstra(self, start_vertex: str, to_vertex: str) -> tuple:
        """
        Метод реализует алгоритм Дейкстра для поиска кратчайшего пути от вершины start_vertex до вершины to_vertex.
        :param start_vertex: Стартовая вершина
        :param to_vertex: Целевая вершина
        :return: Список с кратчайшими расстояниями от стартовой вершины целеваой вершины
        """
        distances = {vertex: float("inf") for vertex in self.vertices()}
        previous_vertices = {vertex: None for vertex in self.vertices()}
        distances[start_vertex] = 0

        pq = PriorityQueue()
        pq.put((0, start_vertex))

        while not pq.empty():
            current_distance, u = pq.get()

            if current_distance > distances[u]:
                continue

            for edge in self._graph[u]:
                vert = edge._vertex_to
                weight = edge._weight
                distance = current_distance + weight

                if distance < distances[vert]:
                    distances[vert] = distance
                    previous_vertices[vert] = u
                    pq.put((distance, vert))

        path = []
        current = to_vertex
        while previous_vertices[current] is not None:
            path.append(current)
            current = previous_vertices[current]
        path.append(start_vertex)
        path.reverse()

        return distances, path

if __name__ == "__main__":
    _graph =Graph()

    _graph.add_vertex("A")
    _graph.add_vertex("B")
    _graph.add_vertex("C")
    _graph.add_vertex("D")

    _graph.add_edge("A", "B", 2.0)
    _graph.add_edge("B", "C", 17.0)
    _graph.add_edge("A", "C", 15.7805)
    _graph.add_edge("D", "C", 7.0)

    result_of_walking = _graph.breadth_search("D")
    print(result_of_walking)

    distance, path = _graph.dijkstra("D", "B")
    print(distance)
    print(path)