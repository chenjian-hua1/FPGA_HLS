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

def Mat2CSR(mat: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    """
    Dense matrix -> CSR format

    Returns:
        row_ptr : List[int]  (len = rows + 1) # 每 row 從哪個地方開始，取 row 上的nnz col_idx value \n
        col_idx : List[int]  (len = nnz) # 全部 row 的 nnz col idx \n
        values  : List[int]  (len = nnz) \n
    """
    rows, cols = mat.shape

    row_ptr: list[int] = [0] * (rows + 1)
    col_idx: list[int] = []
    values:  list[int] = []

    nnz = 0
    for r in range(rows):
        row_ptr[r] = nnz
        for c in range(cols):
            v = mat[r, c]
            if v != 0:
                col_idx.append(int(c))
                values.append(int(v))
                nnz += 1

    row_ptr[rows] = nnz  # CSR 最後一格一定是 nnz

    return row_ptr, col_idx, values

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

    #################### CSR Format #############################
    print_center_title("CSR Format")
    row_ptr, col_indices, values = Mat2CSR(mat)
    print("row pointer",row_ptr)
    print("col_indices",col_indices)
    print("values", values)

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

            # CSR
            "A_row_ptr": row_ptr,

            # dense matrix（debug / golden 用）
            "A_dense": mat,
        },
        array_overrides={
            # index / value 型別你可以在這裡完全控制
            "A_row_idx": {"ctype": "int32_t"},
            "A_col_idx": {"ctype": "int32_t"},
            "A_values":  {"ctype": "int32_t"},

            "A_row_ptr": {"ctype": "int32_t"},
            "A_dense":   {"ctype": "int32_t"},
        },
        includes=[
            "<cstdint>",   # 固定寬度整數
        ],
        storage="static const",  # header-only，include 即用
    )