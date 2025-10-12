import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logsumexp


def fast_multivariate_normal_pdf(grid_points, means, cov):
    """
    Векторизованное вычисление PDF для множества точек и средних
    grid_points: [N, 2] - точки сетки Y
    means: [M, 2] - средние значения (координаты данных)
    cov: [2, 2] или [M, 2, 2] - ковариационные матрицы
    """
    # Размерности
    N, dim = grid_points.shape
    M = means.shape[0]

    # Разница: [N, M, 2]
    diff = grid_points[:, np.newaxis, :] - means[np.newaxis, :, :]

    if cov.ndim == 2:
        # Одна ковариационная матрица для всех
        cov_inv = np.linalg.inv(cov)
        det = np.linalg.det(cov)

        # Квадратичная форма: (x-mu)^T Sigma^-1 (x-mu)
        exponent = np.einsum('nmi,ij,nmj->nm', diff, cov_inv, diff)

        # PDF формула
        norm_factor = 1.0 / np.sqrt((2 * np.pi) ** dim * det)
        pdf_values = norm_factor * np.exp(-0.5 * exponent)

    else:
        # Разные ковариационные матрицы для каждой точки
        pdf_values = np.zeros((N, M))
        for i in range(M):
            cov_inv = np.linalg.inv(cov[i])
            det = np.linalg.det(cov[i])

            exponent = np.einsum('ni,ij,nj->n', diff[:, i, :], cov_inv, diff[:, i, :])

            norm_factor = 1.0 / np.sqrt((2 * np.pi) ** dim * det)
            pdf_values[:, i] = norm_factor * np.exp(-0.5 * exponent)

    return pdf_values  # [N, M] - grid_points x means


def add_optimized_methods():
    """Добавляет оптимизированные методы в класс dataset"""
    from IB import dataset

    def fast_coord_to_pxy(self, total_bins=2500, pad=None, drop_distant=True):
        """Оптимизированная версия coord_to_pxy только для uniform smoothing"""

        if self.smoothing_type != 'uniform':
            print("Warning: Using original method for non-uniform smoothing")
            return self.coord_to_pxy(total_bins, pad, drop_distant)

        print("Using VECTORIZED PDF computation for uniform smoothing")

        # Оригинальный код подготовки
        Y, bins1, bins2, y1v, y2v, Ygrid = self.make_bins(total_bins=total_bins, pad=pad)
        print(f"Debug: Ygrid shape: {Ygrid.shape}, coord shape: {self.coord.shape}")

        # ОПТИМИЗАЦИЯ: векторизованное вычисление PDF
        S = (self.s ** 2) * np.eye(2)

        # Вычисляем PDF: результат [Y, X] (Y точек сетки x X данных)
        py_x = fast_multivariate_normal_pdf(Ygrid, self.coord, S)  # [Y, X]
        print(f"Debug: py_x shape after computation: {py_x.shape}")

        # Нормализация по столбцам (для каждого x)
        py_x = py_x / np.sum(py_x, axis=0, keepdims=True)  # Векторизованная нормализация!
        print(f"Debug: py_x shape after normalization: {py_x.shape}")

        # Продолжение оригинального кода
        self.py_x = py_x  # [Y, X]
        self.Y = Y
        self.Ygrid = Ygrid
        self.px = (1 / self.X) * np.ones(self.X, dtype=self.dt)

        # pxy = p(x,y) = p(y|x) * p(x) [=] X x Y
        # py_x: [Y, X], px: [X] -> pxy: [Y, X] -> транспонируем в [X, Y]
        self.pxy = (self.py_x * self.px).T  # [X, Y]
        print(f"Debug: pxy shape: {self.pxy.shape}")

        self.process_pxy(drop_zeros=True)
        print(f"Vectorized: I(X;Y) = {self.ixy:.3f}")

    # Добавляем метод в класс
    dataset.fast_coord_to_pxy = fast_coord_to_pxy
    print("Optimized methods added to dataset class")


# Дополнительная функция для отладки размерностей
def debug_dimensions(ds):
    """Выводит размерности матриц для отладки"""
    print(f"Debug dimensions:")
    print(f"  py_x shape: {ds.py_x.shape} [Y, X]")
    print(f"  px shape: {ds.px.shape} [X]")
    print(f"  pxy shape: {ds.pxy.shape} [X, Y]")
    print(f"  X: {ds.X}, Y: {ds.Y}")