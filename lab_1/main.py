import numpy as np
import matplotlib.pyplot as plt

print("=" * 60)
print("ЛАБОРАТОРНАЯ РАБОТА №2")
print("=" * 60)

# ==============================================
# ЗАДАНИЕ 1: Вычисление норм матрицы
# ==============================================

A = np.array([
    [5.0, -2.0, 0.0],
    [-2.0, 5.0, -2.0],     
    [0.0, -2.0, 5.0]       
])

print("\n--- Задание 1: Нормы матрицы ---")
print("Матрица A:")
print(A)

norm1 = np.linalg.norm(A, ord=1)       
norm_inf = np.linalg.norm(A, ord=np.inf)  
norm_fro = np.linalg.norm(A, ord='fro')    

print(f"1-норма: {norm1:.4f}")
print(f"Бесконечная норма: {norm_inf:.4f}")
print(f"Норма Фробениуса: {norm_fro:.4f}")

# ==============================================
# ЗАДАНИЕ 2: Обусловленность матрицы
# ==============================================

cond_1 = np.linalg.cond(A, p=1)          
cond_inf = np.linalg.cond(A, p=np.inf)   
cond_fro = np.linalg.cond(A, p='fro')    
print("\n--- Задание 2: Число обусловленности ---")
print(f"cond_1 (по 1-норме): {cond_1:.4f}")
print(f"cond_inf (по бесконечной норме): {cond_inf:.4f}")
print(f"cond_fro (по норме Фробениуса): {cond_fro:.4f}")

# ==============================================
# ЗАДАНИЕ 3: Решение СЛАУ методом Гаусса
# ==============================================

def gauss_solve(A, b):
    """
    Функция для решения СЛАУ Ax = b методом Гаусса с выбором главного элемента по столбцу.
    
    Параметры:
    A - квадратная матрица коэффициентов
    b - вектор правой части
    
    Возвращает:
    x - вектор решения
    """
    n = len(A)
    a = A.astype(float).copy()
    bb = b.astype(float).copy()
    
    print("\nПрямой ход метода Гаусса:")
    
    for i in range(n):
        print(f"\nШаг {i+1}:")
        print(f"Текущая матрица:\n{a}")
        print(f"Текущий вектор b: {bb}")
        
        max_row = i                    
        max_val = abs(a[i][i])         
        for k in range(i + 1, n):      
            if abs(a[k][i]) > max_val:
                max_val = abs(a[k][i])
                max_row = k
        
        if max_row != i:
            print(f"Перестановка строк {i+1} и {max_row+1}")
            a[[i, max_row]] = a[[max_row, i]]  
            bb[[i, max_row]] = bb[[max_row, i]]  
            print(f"После перестановки:\n{a}")
            print(f"Вектор b: {bb}")
        
        if abs(a[i][i]) < 1e-15:
            raise ValueError(f"Матрица вырождена: главный элемент a[{i}][{i}] = {a[i][i]}")
        
        print(f"Главный элемент: {a[i][i]:.6f}")
        
        for j in range(i + 1, n):
            factor = a[j][i] / a[i][i]  
            print(f"  Исключение: строка {j+1} = строка {j+1} - ({factor:.6f}) * строка {i+1}")
            a[j][i:] -= factor * a[i][i:]  
            bb[j] -= factor * bb[i]         
    
    print(f"\nМатрица после прямого хода (верхнетреугольная):\n{a}")
    print(f"Вектор b после прямого хода: {bb}")
    
    
    print("\nОбратный ход:")
    x = np.zeros(n)  
    for i in range(n - 1, -1, -1):
        
        x[i] = (bb[i] - np.dot(a[i][i + 1:], x[i + 1:])) / a[i][i]
        print(f"x[{i+1}] = {x[i]:.6f}")
    
    return x

b = np.array([1.0, -1.0, 1.0])

print("\n--- Задание 3: Решение СЛАУ методом Гаусса ---")
print("Вектор правой части b:")
print(b)

try:
    x_gauss = gauss_solve(A, b)
    print("\nРешение методом Гаусса (с выбором главного элемента):")
    print(x_gauss)
    
    x_numpy = np.linalg.solve(A, b)
    print("\nРешение встроенным методом numpy.linalg.solve:")
    print(x_numpy)
    
    error = np.linalg.norm(x_gauss - x_numpy, ord=np.inf)
    print(f"\nМаксимальная погрешность (бесконечная норма): {error:.2e}")
    
    
    if error < 1e-10:
        print("Решения совпадают с высокой точностью")
    else:
        print("Имеется заметная погрешность")
        
except ValueError as e:
    print(f"\nОШИБКА: {e}")

print("\n" + "=" * 60)

# ==============================================
# ЗАДАНИЕ 4: Исследование погрешности для плохо обусловленных матриц
# ==============================================

print("\n--- Задание 4: Исследование плохо обусловленных матриц ---")

def hilbert(n):
    """
    Функция для создания матрицы Гильберта размера n.
    Матрица Гильберта является классическим примером плохо обусловленной матрицы.
    Элементы матрицы: H[i,j] = 1/(i+j+1)
    """
    H = np.zeros((n, n))  
    for i in range(n):
        for j in range(n):
            H[i, j] = 1.0 / (i + j + 1)  
    return H
sizes = [2, 3, 4, 5, 6, 7, 8, 9, 10]
cond_numbers = []  

for n in sizes:
    H = hilbert(n)  
    cond = np.linalg.cond(H, p=np.inf)  
    cond_numbers.append(cond)  
    print(f"n = {n:2d}: cond(H) = {cond:.2e}")  

plt.figure(figsize=(8, 5)) 
plt.semilogy(sizes, cond_numbers, 'o-', color='red', linewidth=2)  
plt.title('Рост числа обусловленности матрицы Гильберта')  
plt.xlabel('Размер матрицы n')  
plt.ylabel('Число обусловленности (логарифм. шкала)')  
plt.grid(True, which='both', linestyle='--', alpha=0.7)  
plt.show()  