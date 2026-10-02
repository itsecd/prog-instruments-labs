"""
This is a sample Python file that demonstrates various code smells.
"""

# Long function with poor variable names and repetitive code
def calculate_statistics(l):
    n = len(l)
    
    # Calculate mean
    s = 0
    for i in range(n):
        s += l[i]
    m = s / n
    
    # Calculate median
    sorted_l = sorted(l)
    if n % 2 == 0:
        idx1 = n // 2 - 1
        idx2 = n // 2
        med = (sorted_l[idx1] + sorted_l[idx2]) / 2
    else:
        idx = n // 2
        med = sorted_l[idx]
    
    # Calculate variance
    var_sum = 0
    for i in range(n):
        diff = l[i] - m
        var_sum += diff * diff
    v = var_sum / n
    
    # Calculate standard deviation
    std = v ** 0.5
    
    # Calculate min and max
    mn = l[0]
    mx = l[0]
    for i in range(1, n):
        if l[i] < mn:
            mn = l[i]
        if l[i] > mx:
            mx = l[i]
    
    # Calculate range
    r = mx - mn
    
    # Calculate quartiles
    q1_idx = n // 4
    q3_idx = 3 * n // 4
    q1 = sorted_l[q1_idx]
    q3 = sorted_l[q3_idx]
    
    # Calculate IQR
    iqr = q3 - q1
    
    # Calculate outliers
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outliers = []
    for i in range(n):
        if l[i] < lower_bound or l[i] > upper_bound:
            outliers.append(l[i])
    
    return {
        'mean': m,
        'median': med,
        'variance': v,
        'standard_deviation': std,
        'min': mn,
        'max': mx,
        'range': r,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'outliers': outliers
    }

# Function with duplicate code
def process_data(data):
    # Process integers
    if isinstance(data, int):
        result = data * 2
        print(f"Processing integer: {data}")
        print(f"Result: {result}")
        return result
    
    # Process floats
    elif isinstance(data, float):
        result = data * 2.5
        print(f"Processing float: {data}")
        print(f"Result: {result}")
        return result
    
    # Process strings
    elif isinstance(data, str):
        result = data.upper()
        print(f"Processing string: {data}")
        print(f"Result: {result}")
        return result
    
    # Process lists
    elif isinstance(data, list):
        result = [item * 2 for item in data]
        print(f"Processing list: {data}")
        print(f"Result: {result}")
        return result
    
    # Process dictionaries
    elif isinstance(data, dict):
        result = {key: value * 2 for key, value in data.items()}
        print(f"Processing dictionary: {data}")
        print(f"Result: {result}")
        return result
    
    # Default case
    else:
        print(f"Unknown data type: {type(data)}")
        return None

if __name__ == "__main__":
    # Test calculate_statistics
    data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]
    stats = calculate_statistics(data)
    print("Statistics:", stats)
    
    # Test process_data
    print("\nProcessing different data types:")
    process_data(42)
    process_data(3.14)
    process_data("hello")
    process_data([1, 2, 3])
    process_data({"a": 1, "b": 2}) 