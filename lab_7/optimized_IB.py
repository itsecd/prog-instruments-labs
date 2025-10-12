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

    return pdf_values


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

        # ОПТИМИЗАЦИЯ: векторизованное вычисление PDF
        S = (self.s ** 2) * np.eye(2)
        py_x = fast_multivariate_normal_pdf(Ygrid, self.coord, S)
        py_x = py_x.T  # [Y, X]

        # Нормализация
        for x in range(self.X):
            py_x[:, x] = py_x[:, x] / np.sum(py_x[:, x])

        # Продолжение оригинального кода
        self.py_x = py_x
        self.Y = Y
        self.Ygrid = Ygrid
        self.px = (1 / self.X) * np.ones(self.X, dtype=self.dt)
        self.pxy = (self.py_x * self.px).T  # Более эффективно чем multiply+tile

        self.process_pxy(drop_zeros=True)
        print(f"Vectorized: I(X;Y) = {self.ixy:.3f}")

    # Добавляем метод в класс
    dataset.fast_coord_to_pxy = fast_coord_to_pxy
    print("Optimized methods added to dataset class")


# Альтернативная версия для прямого использования
class OptimizedDataset:
    """Оптимизированная версия класса dataset (альтернативный подход)"""

    def __init__(self, *args, **kwargs):
        from IB import dataset
        self._ds = dataset(*args, **kwargs)

    def __getattr__(self, name):
        """Делегируем все атрибуты оригинальному dataset"""
        return getattr(self._ds, name)

    def fast_coord_to_pxy(self, total_bins=2500, pad=None, drop_distant=True):
        """Оптимизированная версия coord_to_pxy"""
        return self._ds.fast_coord_to_pxy(total_bins, pad, drop_distant)