import numpy as np
from IB import dataset
from optimized_IB import add_optimized_methods


def test_simple_case():
    """Простой тест с известными размерами"""
    print("=== Simple Debug Test ===")

    # Создаем очень простой dataset
    coords = np.array([[0, 0], [1, 0], [0, 1]], dtype=np.float32)
    labels = np.array([0, 1, 1])

    ds = dataset(coord=coords, labels=labels, name="debug",
                 smoothing_type='uniform', s=1.0)

    add_optimized_methods()

    print(f"Dataset: X={ds.X} points")

    try:
        print("Testing optimized method...")
        ds.fast_coord_to_pxy(total_bins=25)
        print("✅ SUCCESS!")
        print(f"I(X;Y) = {ds.ixy:.3f}")
    except Exception as e:
        print(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


def compare_with_original():
    """Сравниваем с оригиналом на маленьких данных"""
    print("\n=== Comparison with Original ===")

    coords = np.array([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.float32)
    labels = np.array([0, 1, 1, 0])

    # Оригинал
    ds_orig = dataset(coord=coords.copy(), labels=labels.copy(),
                      name="orig", smoothing_type='uniform', s=1.0)
    ds_orig.coord_to_pxy(total_bins=25)
    print(f"Original: I(X;Y) = {ds_orig.ixy:.3f}")

    # Оптимизированный
    ds_opt = dataset(coord=coords.copy(), labels=labels.copy(),
                     name="opt", smoothing_type='uniform', s=1.0)
    add_optimized_methods()
    ds_opt.fast_coord_to_pxy(total_bins=25)
    print(f"Optimized: I(X;Y) = {ds_opt.ixy:.3f}")


if __name__ == "__main__":
    test_simple_case()
    compare_with_original()