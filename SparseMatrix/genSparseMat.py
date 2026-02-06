import numpy as np

def generate_sparse_matrix(rows, cols, density=0.2, value_range=(1, 9), seed=None):
    """
    生成隨機稀疏矩陣

    Args:
        rows (int): 矩陣列數
        cols (int): 矩陣行數
        density (float): 非零元素比例 (0~1)
        value_range (tuple): 非零值範圍 (min, max)
        seed (int or None): 隨機種子（None = 每次不同）

    Returns:
        np.ndarray: (rows x cols) 稀疏矩陣
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.random.rand(rows, cols) < density
    values = np.random.randint(value_range[0], value_range[1] + 1, size=(rows, cols))

    return mask * values


if __name__=="__main__":
    rows, cols = 5, 5
    mat = generate_sparse_matrix(rows, cols, density=0.4)
    print(mat)