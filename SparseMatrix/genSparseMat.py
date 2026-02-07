import numpy as np
from pyData2C import export_arrays_to_c_header

def print_center_title(title:str, width=60, fill_char="="):
    """
    印出置中標題，例如：
    ======== Sparse Matrix ========

    Args:
        title (str): 標題文字
        width (int): 總寬度（包含填充符號）
        fill_char (str): 填充字元
    """
    title_str = f" {title} "
    if len(title_str) >= width:
        print(title_str)
    else:
        print(title_str.center(width, fill_char))

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

def Mat2COO(mat:np.ndarray) -> tuple[list[int],list[int],list]:
    """ 
    將稀疏矩陣轉成 COO 儲存格式 
    return row_indices,col_indice,values
    """
    if (len(mat.shape)!=2):
        print(f"Error : Matrix Shape is 2 dim but get {len(mat.shape)} dim") 
        return None
    
    rows, cols = mat.shape

    # 非零資料位置
    row_indices = []
    col_indices = []
    # 非零資料
    values = []

    for r in range(rows):
        for c in range(cols):
            if mat[r][c]==0: continue
            # else
            row_indices.append(r)
            col_indices.append(c)
            values.append(mat[r][c])

    return row_indices, col_indices, values


if __name__=="__main__":
    rows, cols = 5, 5
    mat = generate_sparse_matrix(rows, cols, density=0.4)
    
    print_center_title(title="Sparse Matrix")
    print(mat)

    ##################### COO Format ############################
    print_center_title("COO Format")
    row_indices, col_indices, values = Mat2COO(mat)
    print("row indices:",row_indices)
    print("col_indices:",col_indices)
    print("values:", values)

    export_arrays_to_c_header(
        out_path="SparseMatrix/sparse_data.h",
        variables={
            # scalar
            "A_ROWS": rows,
            "A_COLS": cols,
            "A_NNZ":  len(values),

            # COO
            "A_row_idx": row_indices,
            "A_col_idx": col_indices,
            "A_values":  values,

            # dense matrix（debug / golden 用）
            "A_dense": mat,
        },
        array_overrides={
            # index / value 型別你可以在這裡完全控制
            "A_row_idx": {"ctype": "int32_t"},
            "A_col_idx": {"ctype": "int32_t"},
            "A_values":  {"ctype": "int32_t"},
            "A_dense":   {"ctype": "int32_t"},
        },
        includes=[
            "<cstdint>",   # 固定寬度整數
        ],
        storage="static const",  # header-only，include 即用
    )