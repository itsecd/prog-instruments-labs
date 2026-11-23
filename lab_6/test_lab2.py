"""Юнит-тесты для lab2.py (нормы, обусловленность, метод Гаусса, матрицы Гильберта)."""

import runpy

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

import lab2


class PlotSpy:
    """Заглушка вместо matplotlib.pyplot: запоминает вызовы и ничего не рисует."""

    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        def method(*args, **kwargs):
            self.calls.append((name, args, kwargs))

        return method

    def names(self):
        return [name for name, _, _ in self.calls]

    def find(self, name):
        return [(args, kwargs) for n, args, kwargs in self.calls if n == name]


# ---------------------------------------------------------------- hilbert


def test_hilbert_known_values():
    expected = np.array([
        [1, 1 / 2, 1 / 3],
        [1 / 2, 1 / 3, 1 / 4],
        [1 / 3, 1 / 4, 1 / 5],
    ])
    assert np.allclose(lab2.hilbert(3), expected)


@pytest.mark.parametrize("size", range(1, 11))
def test_hilbert_shape_symmetry_and_corners(size):
    h = lab2.hilbert(size)
    assert h.shape == (size, size)
    assert np.allclose(h, h.T)
    assert h[0, 0] == 1.0
    assert h[-1, -1] == pytest.approx(1 / (2 * size - 1))


# ------------------------------------------------------- вспомогательные шаги Гаусса


@pytest.mark.parametrize(
    "matrix, column, expected_row",
    [
        ([[1, 2], [3, 4]], 0, 1),                       # максимум ниже диагонали
        ([[5, 0], [1, 1]], 0, 0),                       # максимум уже на диагонали
        ([[1, 2, 3], [0, -7, 1], [0, 2, 1]], 1, 1),     # отрицательное значение, берём по модулю
        ([[1, 2, 3], [0, 1, 1], [0, -4, 1]], 1, 2),     # строки выше столбца не учитываются
        ([[9, 0], [0, 1]], 1, 1),                       # последний столбец
    ],
)
def test_choose_pivot_row(matrix, column, expected_row):
    assert lab2.choose_pivot_row(np.array(matrix, dtype=float), column) == expected_row


def test_swap_rows_changes_both_matrix_and_vector():
    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    b = np.array([10.0, 20.0])

    lab2.swap_rows(a, b, 0, 1)

    assert np.array_equal(a, [[3.0, 4.0], [1.0, 2.0]])
    assert np.array_equal(b, [20.0, 10.0])


def test_forward_elimination_makes_matrix_upper_triangular():
    a = lab2.MATRIX_A.copy()
    b = lab2.VECTOR_B.copy()

    lab2.forward_elimination(a, b)

    assert np.allclose(np.tril(a, k=-1), 0.0)


def test_back_substitution_on_triangular_system():
    a = np.array([[2.0, 1.0], [0.0, 4.0]])
    b = np.array([4.0, 8.0])

    assert np.allclose(lab2.back_substitution(a, b), [1.0, 2.0])


# ------------------------------------------------------------- gauss_solve


GAUSS_SYSTEMS = [
    pytest.param(lab2.MATRIX_A, lab2.VECTOR_B, id="variant-4"),
    pytest.param([[4.0]], [8.0], id="1x1"),
    pytest.param([[0.0, 1.0], [1.0, 0.0]], [2.0, 3.0], id="needs-row-swap"),
    pytest.param([[0.0, 2.0, 1.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0]], [1.0, 2.0, 3.0], id="zero-on-diagonal"),
    pytest.param(lab2.hilbert(4), [1.0, 2.0, 3.0, 4.0], id="hilbert-4"),
]


@pytest.mark.parametrize("matrix, rhs", GAUSS_SYSTEMS)
def test_gauss_solve_matches_numpy(matrix, rhs):
    matrix = np.array(matrix, dtype=float)
    rhs = np.array(rhs, dtype=float)

    x = lab2.gauss_solve(matrix, rhs)

    assert np.allclose(x, np.linalg.solve(matrix, rhs))
    assert np.allclose(matrix @ x, rhs)


@pytest.mark.parametrize(
    "matrix",
    [
        [[1.0, 2.0], [2.0, 4.0]],                                  # линейно зависимые строки
        [[0.0, 0.0], [0.0, 0.0]],                                  # нулевая матрица
        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]],       # вырожденность выявляется на втором шаге
    ],
)
def test_gauss_solve_raises_on_singular_matrix(matrix):
    matrix = np.array(matrix)
    rhs = np.ones(len(matrix))

    with pytest.raises(ValueError, match="вырождена"):
        lab2.gauss_solve(matrix, rhs)


def test_gauss_solve_does_not_modify_arguments():
    matrix = lab2.MATRIX_A.copy()
    rhs = lab2.VECTOR_B.copy()

    lab2.gauss_solve(matrix, rhs)

    assert np.array_equal(matrix, lab2.MATRIX_A)
    assert np.array_equal(rhs, lab2.VECTOR_B)


def test_gauss_solve_accepts_integer_input():
    x = lab2.gauss_solve(np.array([[2, 0], [0, 4]]), np.array([2, 8]))

    assert np.allclose(x, [1.0, 2.0])


# ----------------------------------------------------------- вывод на экран


def test_print_banner(capsys):
    lab2.print_banner("Первая", "Вторая")

    lines = capsys.readouterr().out.splitlines()
    assert lines == ["=" * lab2.LINE_WIDTH, "Первая", "Вторая", "=" * lab2.LINE_WIDTH]


def test_print_task_title(capsys):
    lab2.print_task_title("Задание X")

    assert capsys.readouterr().out == "\n--- Задание X ---\n"


# Матрица варианта симметрична, из-за чего 1-норма и бесконечная норма у неё равны.
# Вторая, несимметричная матрица нужна, чтобы тест различал эти нормы.
NORM_CASES = [
    pytest.param(
        lab2.MATRIX_A,
        {"1-норма:": "9.0000", "Бесконечная норма:": "9.0000", "Норма Фробениуса:": "9.5394"},
        id="variant-4-symmetric",
    ),
    pytest.param(
        np.array([[1.0, 2.0], [3.0, 4.0]]),
        {"1-норма:": "6.0000", "Бесконечная норма:": "7.0000", "Норма Фробениуса:": "5.4772"},
        id="asymmetric",
    ),
]


@pytest.mark.parametrize("matrix, expected_by_label", NORM_CASES)
def test_task_matrix_norms_prints_expected_values(capsys, matrix, expected_by_label):
    lab2.task_matrix_norms(matrix)

    lines = capsys.readouterr().out.splitlines()
    for label, expected_value in expected_by_label.items():
        matching = [line for line in lines if line.startswith(label)]
        assert len(matching) == 1
        assert matching[0].endswith(expected_value)


@pytest.mark.parametrize("matrix", [case.values[0] for case in NORM_CASES], ids=[case.id for case in NORM_CASES])
@pytest.mark.parametrize("label, order", [(row[1], row[2]) for row in lab2.NORMS])
def test_task_condition_numbers_uses_matching_norm(capsys, matrix, label, order):
    expected = np.linalg.norm(matrix, ord=order) * np.linalg.norm(np.linalg.inv(matrix), ord=order)

    lab2.task_condition_numbers(matrix)

    lines = capsys.readouterr().out.splitlines()
    matching = [line for line in lines if line.startswith(label)]
    assert len(matching) == 1
    assert matching[0].endswith(f"{expected:.4f}")


# ----------------------------------------------- тесты со стабами и подменой зависимостей


def test_task_gauss_reports_error_against_numpy(monkeypatch, capsys):
    """Стаб gauss_solve с заранее известной погрешностью: task_gauss должен её напечатать."""
    received = []

    def stub_gauss_solve(matrix, rhs):
        received.append((matrix, rhs))
        return np.linalg.solve(matrix, rhs) + 1e-3

    monkeypatch.setattr(lab2, "gauss_solve", stub_gauss_solve)

    lab2.task_gauss(lab2.MATRIX_A, lab2.VECTOR_B)

    assert len(received) == 1
    assert received[0][0] is lab2.MATRIX_A and received[0][1] is lab2.VECTOR_B
    assert "Максимальная погрешность (бесконечная норма): 1.00e-03" in capsys.readouterr().out


def test_plot_condition_growth_builds_log_plot_without_window(monkeypatch):
    """Вместо matplotlib подставляем шпиона и проверяем, как строится график."""
    spy = PlotSpy()
    monkeypatch.setattr(lab2, "plt", spy)
    sizes = [2, 3, 4]
    conds = [27.0, 748.0, 28375.0]

    lab2.plot_condition_growth(sizes, conds)

    assert spy.find("figure")[0][1]["figsize"] == lab2.PLOT_FIGSIZE
    semilogy_args, semilogy_kwargs = spy.find("semilogy")[0]
    assert semilogy_args[:2] == (sizes, conds)
    assert semilogy_kwargs["color"] == lab2.PLOT_COLOR
    assert {"title", "xlabel", "ylabel", "grid"} <= set(spy.names())
    assert spy.names().count("show") == 1
    assert spy.names()[-1] == "show"  # показ окна идёт после всей настройки


def test_task_hilbert_collects_condition_numbers_and_plots_them(monkeypatch, capsys):
    plotted = []
    monkeypatch.setattr(lab2, "plot_condition_growth", lambda sizes, conds: plotted.append((list(sizes), list(conds))))

    lab2.task_hilbert([2, 3, 4])

    assert len(plotted) == 1
    sizes, conds = plotted[0]
    assert sizes == [2, 3, 4]
    assert conds == sorted(conds) and len(set(conds)) == 3       # обусловленность растёт с размером
    assert conds[0] == pytest.approx(27.0)                        # cond_inf(H_2) = 1.5 * 18
    out = capsys.readouterr().out
    assert "n =  2: cond(H) = 2.70e+01" in out
    assert out.count("cond(H)") == 3


def test_main_runs_all_tasks_in_order(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(lab2, "task_matrix_norms", lambda m: calls.append(("norms", m)))
    monkeypatch.setattr(lab2, "task_condition_numbers", lambda m: calls.append(("cond", m)))
    monkeypatch.setattr(lab2, "task_gauss", lambda m, b: calls.append(("gauss", m, b)))
    monkeypatch.setattr(lab2, "task_hilbert", lambda s: calls.append(("hilbert", s)))

    lab2.main()

    assert [c[0] for c in calls] == ["norms", "cond", "gauss", "hilbert"]
    assert calls[0][1] is lab2.MATRIX_A
    assert calls[2][2] is lab2.VECTOR_B
    assert calls[3][1] == lab2.HILBERT_SIZES
    out = capsys.readouterr().out
    assert "ЛАБОРАТОРНАЯ РАБОТА №2" in out
    assert "ВСЕ ЗАДАНИЯ ВЫПОЛНЕНЫ" in out


# ----------------------------------------------------- запуск файла целиком


def test_importing_module_has_no_side_effects(capsys):
    runpy.run_path(lab2.__file__, run_name="lab2_imported")

    assert capsys.readouterr().out == ""


def test_running_as_script_executes_main(monkeypatch, capsys):
    monkeypatch.setattr(matplotlib.pyplot, "show", lambda *args, **kwargs: None)

    runpy.run_path(lab2.__file__, run_name="__main__")

    out = capsys.readouterr().out
    assert "Задание 4: Исследование плохо обусловленных матриц" in out
    assert out.rstrip().endswith("=" * lab2.LINE_WIDTH)