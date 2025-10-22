from sympy import *
from sympy.core.function import AppliedUndef
from sympy.core.numbers import Pi, Exp1,ImaginaryUnit,Infinity,NegativeInfinity,NaN,ComplexInfinity
from sympy.matrices import MatrixBase
from sympy.core.relational import Relational
from sympy import Derivative
from sympy.logic.boolalg import And, Or, Not

import re
import numpy as np
import timeout_decorator
from .extended_zss import ext_distance
from .latex_pre_process import *
from sympy.simplify import *
# from graphviz import Digraph


"""
Guide:
You only need to use EED and install the following packages:
- sympy
- numpy
- latex2sympy2_extended
- timeout_decorator
"""

"""
There are four main categories:

Constants: such as integers, decimals, or mathematical constants like π and e.
Variables: letters like x, y, z, or specified terms in problems (e.g., ħ, c, G).
Functions: sine, cosine, exponential, logarithm, etc.
Operators: basic binary operations including addition, multiplication, and exponentiation.
"""
# The costs can be modified if you think their values are different
insert_cost={"number":1,"symbol":1,"operator":1,"function":1,"matrix":1,"relation":1}
delete_cost={"number":1,"symbol":1,"operator":1,"function":1,"matrix":1,"relation":1}
update_cost={"number":1,"symbol":1,"operator":1,"function":1,"matrix":1,"relation":1}

change_type_cost=1 #the cost of an update between different types,can be set to higher

bar_size=5 # the minimum size of triggering cluster discount
discount_slope=0.6 #discount

simplify_time_limit=30 #set the time limit of simplify
equals_time_limit=10 #set the time limit of equals

def update_func(x,y):
    
    if x.label==y.label:
        return 0
    
    elif x.label.split("_")[0]==y.label.split("_")[0]:
        return update_cost[x.label.split("_")[0]]
    return change_type_cost
def remove_func(x):
    return delete_cost[x.label.split("_")[0]]

def remove_tree_func(x):
    if not x.children:
        return remove_func(x)
    s=calc_tree_size(x)
    return min(s,discount_slope*(s-bar_size)+bar_size)


def insert_func(x):
    return insert_cost[x.label.split("_")[0]]
def insert_tree_func(x):
    return remove_tree_func(x)



def calc_tree_size(node):
    """
    Calculate the size of a subtree based on its total insertion cost.
    The function computes the size of a subtree by summing up the insertion 
    costs of the current node and all its descendant nodes. If the subtree 
    size has already been calculated and stored in `node.subtree_size`, it 
    returns the cached value to avoid redundant computation.
    Args:
        node (Node): The root node of the subtree for which the size is to 
                     be calculated
    Returns:
        int: The total size of the subtree, calculated as the sum of the 
             insertion costs of the current node and all its descendants.
    Notes:
        - The `insert_cost` dictionary is assumed to be globally defined 
          and maps node labels to their respective insertion costs.
        - The function modifies the `subtree_size` attribute of the input 
          node to store the calculated subtree size for future use.
    """
    """The size of a subtree equals to its total insertion cost"""
    
    total = insert_cost[node.label.split("_")[0]]
    
    if node.children and node.subtree_size !=0:

        return node.subtree_size
    
    for child in node.children:
        total += calc_tree_size(child)
    
    node.subtree_size=total

    return total
"""
Scoring function from relative distance
"""
def score_calc(tree_dist,tree_size):

    if tree_dist==0.:
        return 100
    return max(0,100*discount_slope-100*tree_dist/tree_size)

def numeric_score_calc(student_answer_exp, ground_truth_exp):
    """
    数值类型专用评分函数
    根据绝对误差和相对误差的组合标准进行评分
    """
    # ===================================================================
    #  参数设定区 (调整评分的严格程度)
    # ===================================================================
    
    # 100分标准 (最严格)
    RelTol_100_strict = 0.005  # 0.5%
    # RelTol_100_guard = 0.05    # 5% (护栏)
    
    # 90分标准 (较严格)
    RelTol_90 = 0.01   # 1%
    
    # 80分标准 (较宽松)
    RelTol_80 = 0.02   # 2%
    
    try:
        # 如果ground_truth_exp是等式，提取右边的值
        if hasattr(ground_truth_exp, 'rhs'):
            ground_truth_value = ground_truth_exp.rhs
            print(f"     Detected equation, using rhs: {ground_truth_value}")
        else:
            ground_truth_value = ground_truth_exp
            
        # 尝试将SymPy表达式转换为数值
        ground_truth = float(ground_truth_value.evalf())
        student_answer = float(student_answer_exp.evalf())
        
        print(f"     ground_truth (float): {ground_truth}")
        print(f"     student_answer (float): {student_answer}")
        
        # 预处理：处理正确答案为0的特殊情况
        if ground_truth == 0:
            if student_answer == 0:
                return 100
            else:
                    return 0
        
        # 第一步：符号一致性检查 (拦截重大概念错误)
        if ground_truth * student_answer < 0:
            return 0
        
        # 第二步：计算误差
        absolute_error = abs(student_answer - ground_truth)
        relative_error = absolute_error / abs(ground_truth)

        
        # 判断100分
        is_extremely_close = (relative_error <= RelTol_100_strict)
        
        if is_extremely_close:
            return 100
        
        # 判断90分 
        elif relative_error <= RelTol_90:
            return 90
        
        # 判断80分 
        elif relative_error <= RelTol_80:
            return 80
        
        # 所有标准都不满足
        else:
            return 0
            
    except Exception as e:
        print(f"  -> numeric_score_calc error: {e}")
        # 如果数值转换失败，回退到原来的评分方法
        return 0

@timeout_decorator.timeout(30, timeout_exception=TimeoutError)
def simplify_with_timeout(expr):
    return simplify(expr)
def time_simplify(expr):
    try:
        result=simplify_with_timeout(expr)
        return result
    except TimeoutError:
        return expr

@timeout_decorator.timeout(10, timeout_exception=TimeoutError)
def equal_with_timeout(expr1,expr2):
    return expr1.equals(expr2)
def time_equal(expr1,expr2):
    try:
        result=equal_with_timeout(expr1,expr2)
        return result
    except TimeoutError:
        return False


def sympy_to_tree(expr):
    """
    Convert a SymPy expression into a tree structure.
    This function takes a SymPy expression and recursively converts it into a tree
    representation using `TreeNode` objects. Each node in the tree is labeled based
    on the type of the SymPy expression (e.g., number, symbol, operator, or function),
    and its children represent the arguments of the expression.
    Args:
        expr (sympy.Basic): The SymPy expression to be converted.
    Returns:
        TreeNode: The root node of the tree representation of the SymPy expression.
    Raises:
        ValueError: If the SymPy expression contains an unsupported type.
    Supported Types:
        - Numbers: Integer, Pi, Exp1, Float, Rational, Infinity, NegativeInfinity
        - Symbols: Symbol
        - Binary Operators: Add, Mul, Pow
        - Functions: Any subclass of `sympy.Function`
    Example:
        >>> from sympy import symbols, sin, pi
        >>> x, y = symbols('x y')
        >>> expr = x + y * sin(pi)
        >>> tree = sympy_to_tree(expr)
        >>> print(tree)
    """
  

    """Convert the sympy expression to a tree"""
    if isinstance(expr, MatrixBase):
        # 遍历矩阵所有元素，递归转换
        children = []
        for i in range(expr.rows):
            for j in range(expr.cols):
                children.append(sympy_to_tree(expr[i, j]))
        return TreeNode(label=f"matrix_{expr.rows}x{expr.cols}", children=children)

    elif isinstance(expr, (Integer, Pi, Exp1, ImaginaryUnit, Float, Rational, Infinity, NegativeInfinity, NaN, ComplexInfinity)):
        return TreeNode(label="number_" + str(expr), children=[])
    elif isinstance(expr, Symbol):
        return TreeNode(label="symbol_" + str(expr), children=[])
    elif isinstance(expr, (Add, Mul, Pow)):
        op_name = type(expr).__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="operator_" + op_name, children=children)
    elif isinstance(expr, Function):
        func_name = expr.func.__name__
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="function_" + func_name, children=children)
    elif isinstance(expr, Relational):  # 支持不等式和等式
        op_name = type(expr).__name__  # 如 StrictLessThan, LessThan, Equality 等
        children = [sympy_to_tree(expr.lhs), sympy_to_tree(expr.rhs)]
        return TreeNode(label="relation_" + op_name, children=children)
    elif isinstance(expr, Derivative):
        # expr.expr 是被求导的表达式
        # expr.variables 是求导变量的元组
        children = [sympy_to_tree(expr.expr)] + [sympy_to_tree(v) for v in expr.variables]
        return TreeNode(label="function_Derivative", children=children)
     # 新增支持逻辑表达式
    elif isinstance(expr, And):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="logic_And", children=children)
    elif isinstance(expr, Or):
        children = [sympy_to_tree(arg) for arg in expr.args]
        return TreeNode(label="logic_Or", children=children)
    elif isinstance(expr, Not):
        children = [sympy_to_tree(expr.args[0])]
        return TreeNode(label="logic_Not", children=children)
    else:
        raise ValueError(f"Unsupported SymPy type: {type(expr)} Expression: {expr}")

class TreeNode:
    def __init__(self, label, children=None,node_type='other'):
        self.label = label
        self.children = children if children is not None else []
        self.node_type=node_type
        self.subtree_size=0
    def get_children(self):
        return self.children
    
    def __str__(self):
        return self.label

def print_tree(node, indent=0):
    """Print a tree structure"""
    print('  ' * indent + f'└─ {node.label}')
    for child in node.children:
        print_tree(child, indent + 1)

class LaTeXError(Exception):
    def __init__(self, message="LaTeXError"):
        super().__init__(message)

class SymPyError(Exception):
    def __init__(self, message="SymPyError"):
        super().__init__(message)

class TreeError(Exception):
    def __init__(self, message="TreeError"):
        super().__init__(message)

class DistError(Exception):
    def __init__(self, message="DistanceError"):
        super().__init__(message)

def Equation_standardize(latex):
    return latex.args[0] - latex.args[1]

def extract_interval(latex):
    # 使用普通字符串（非 raw string），所以反斜杠都写成 \\ 来转义
    interval_pattern = re.compile(
        "^\s*"                                 # 开头空格
        "(?:\\\\left)?\s*"                     # 可选 \left
        "([\(\[])\s*"                          # 第1组：左括号
        "(.*?)\s*,\s*"                         # 第2组：下界
        "(.*?)\s*"                             # 第3组：上界
        "(?:\\\\right)?\s*"                    # 可选 \right
        "([\)\]])\s*$"                         # 第4组：右括号
    )
    # 分析和输出
    match = interval_pattern.match(latex)
    if match:
        left_bracket, lower_bound, upper_bound, right_bracket = match.groups()
        return True, left_bracket, lower_bound, upper_bound, right_bracket
    else:
        return False, None, None, None, None
    
def judge_interval(latex):
    latex=latex.replace('$','')
    match, left_bracket, lower_bound, upper_bound, right_bracket = extract_interval(latex)
    if match:
        # 判断是否开/闭区间
        is_left_closed = left_bracket == "["
        is_right_closed = right_bracket == "]"
        left_type = "2*" if is_left_closed else "1*"
        right_type = "*4" if is_right_closed else "*3"
        # print(f"原始表达式: {case}")
        # print(f"左括号: {left_bracket}（{left_type}区间）")
        # print(f"右括号: {right_bracket}（{right_type}区间）")
        # print("----------")
        return True, left_type + lower_bound + "+" + upper_bound + right_type
    else:
        return False, latex

def check_latex_wrap(s):
    s = s.strip()
    pattern = r'''
        ^(
            \(.*\) |                            # 普通圆括号 ( )
            \[.*\] |                            # 普通方括号 [ ]
            \\\(.*\\\) |                        # LaTeX inline math: \( \)
            \\\[.*\\\] |                        # LaTeX display math: \[ \]
            \\\\left\(.*\\\\right\) |           # LaTeX \left( \right)
            \\\\left\[.*\\\\right\] |           # LaTeX \left[ \right]
            \$.*\$                              # LaTeX inline math with $...$
        )$
    '''
    return re.match(pattern, s, re.VERBOSE) is not None

def parse_bracketed_string(s):
    # 去除左右括号：支持 (), \left( \right)
    s = s.strip()
    s = re.sub(r'^\\left\(|^\(', '', s)  # 去掉左括号
    s = re.sub(r'\\right\)$|\)$', '', s)  # 去掉右括号
    # 分割内容
    parts = [item.strip() for item in s.split(',')]
    return parts

def strip_dollar_signs(s):
    s = s.strip()
    if s.startswith("$$") and s.endswith("$$"):
        return s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$"):
        return s[1:-1].strip()
    return s

def extract_numeric_part(latex_str: str) -> str:
    """
    数值提取器 (更强大的版本)
    从一个可能包含单位、变量、等式的复杂LaTeX字符串中，
    智能地提取并返回一个纯净的、只包含数字和基本运算符号的字符串。
    """
    if not isinstance(latex_str, str) or not latex_str:
        return ""
                
    s = latex_str.strip()

    # 步骤1: 剥离最外层的LaTeX数学环境定界符
    if s.startswith('$') and s.endswith('$'):
        s = s.strip('$').strip()
    if s.startswith('\\(') and s.endswith('\\)'):
        s = s[2:-2].strip()
    if s.startswith('\\[') and s.endswith('\\]'):
        s = s[2:-2].strip()
                        
    # 步骤2: 如果存在等式或约等号，只取其右侧部分
    # 使用非贪婪匹配 .*? 来确保它不会意外地匹配太多东西
    # 支持 a = b, a \approx b 等多种形式
    equal_sign_pattern = r'.*?(?:=|\\approx|\\sim|\\simeq|\\propto)\s*(.*)'
    match = re.search(equal_sign_pattern, s)
    if match:
        s = match.group(1).strip()

    # 步骤3: 【核心改进】主动匹配并提取科学记数法或普通数值
    # 这个正则表达式可以匹配 -1.28, 1.28e-5, -1.28 \times 10^{-5} 等多种形式
    numeric_pattern = re.compile(
        r"([-+]?\s*(?:\d+\.?\d*|\.\d+)\s*(?:(?:e|E)\s*[-+]?\s*\d+|\\times\s*10\^\{[-+]?\d+\})?)"
    )
                
    match = numeric_pattern.search(s)
                
    if match:
        # 如果成功匹配，直接返回最核心的数值字符串
        numeric_part = match.group(0)
        # 清理一下，将 \times 替换为 *
        cleaned_part = numeric_part.replace('\\times', '*')
        return cleaned_part.strip()
    else:
        return s

def extract_tuple(latex):
    """
    一个更智能、更适合评分的元组/键值对解析器。
    
    核心策略：
    1. 如果表达式是 `(keys) = (values)` 的形式，则【忽略】左边的 `(keys) =` 部分，
       只把右边的 `(values)` 作为要解析的目标。
    2. 如果表达式只是一个元组 `(values)`，则直接解析它。
    3. 最终总是返回一个以数字索引为键的字典，如 {'0': val1, '1': val2, ...}。
    """
    latex = strip_dollar_signs(latex.strip())
    latex = latex.replace(r'\left', '')
    latex = latex.replace(r'\right', '')

    # 步骤 1: 检查是否存在顶层的 '(keys) = (values)' 结构
    paren_level = 0
    top_level_equal_index = -1
    for i, char in enumerate(latex):
        if char in '({[': paren_level += 1
        elif char in ')}]': paren_level -= 1
        elif char == '=' and paren_level == 0:
            top_level_equal_index = i
            break
    
    # 如果找到了这种结构，我们就【只关注等号右边的部分】
    if top_level_equal_index != -1:
        left_part = latex[:top_level_equal_index].strip()
        right_part = latex[top_level_equal_index+1:].strip()
        # 做一个健全性检查，确保等号两边看起来都像元组
        if check_latex_wrap(left_part) and check_latex_wrap(right_part):
            # 这就是关键：我们用右边的部分覆盖掉整个表达式
            latex = right_part

    # 步骤 2: 解析最终的元组字符串 (无论是原始的还是等号右边的)
    if not check_latex_wrap(latex):
        return {}

    # `parse_bracketed_string` 会去掉括号并按逗号分割
    values = parse_bracketed_string(latex)
    
    # 如果是个空元组 "()"，解析后 values 会是空列表
    if not values:
        # 为了与非空元组区分，我们可以返回一个特殊标记或空的dict
        # 这里返回空dict，EED中的逻辑会正确处理
        return {}

    # 将值列表转换为以数字索引为键的字典
    return {str(i): v for i, v in enumerate(values)}

def SEED(answer_latex,test_latex,t,debug_mode=False):
    """
        Computes the similarity score and distance metrics between two LaTeX expressions.
        This function evaluates the equivalence of two mathematical expressions represented 
        in LaTeX format. It uses symbolic computation and tree-based distance metrics to 
        calculate a similarity score and other related metrics.
    
            tuple: A tuple containing the following elements:
                - score (float): The similarity score between the two expressions (0 to 100).
                - relative_distance (float): The normalized distance between the two expressions.
                - answer_tree_size (int): The size of the expression tree for the answer.
                - distance (float): The raw distance between the two expression trees.
        Notes:
            - If either input contains unsupported LaTeX constructs (e.g., integrals or sums), 
              the function returns default values indicating failure.
            - If the test expression is significantly longer than the answer expression, 
              the function assumes they are not equivalent.
            - The function uses symbolic simplification and tree-based distance metrics to 
              evaluate equivalence.
            - In case of errors during processing, the function returns default values unless 
              `debug_mode` is enabled, in which case it raises specific exceptions.
        Exceptions:
            - LaTeXError: Raised when LaTeX conversion to symbolic expressions fails (if `debug_mode` is True).
            - SymPyError: Raised when symbolic simplification or tree construction fails (if `debug_mode` is True).
            - DistError: Raised when distance calculation fails (if `debug_mode` is True).
        Args:
            answer_latex: the latex expression of answer expression
            test_latex: the latex expression of test expression
            debug_mode: whether it raise errors or just skip it
        Returns:
             tuple: A tuple containing the following elements:
                - score (float): The similarity score between the two expressions (0 to 100).
                - relative_distance (float): The normalized distance between the two expressions.
                - answer_tree_size (int): The size of the expression tree for the answer.
                - distance (float): The raw distance between the two expression trees.
    """

    if not test_latex:
        return 0,-1,-1,-1
    if '\\int' in test_latex or '\\int' in answer_latex:
        return 0,-1,-1,-1
    if '\\sum' in test_latex or '\\sum' in answer_latex:
        return 0,-1,-1,1
    if answer_latex==test_latex:
        return 100,0.0,-1,0
    # if len(test_latex)>3*len(answer_latex):
    #     return 0,-1,-1,-1
    
    try:
        if t == 'Tuple':
            answer_dict = extract_tuple(answer_latex)
            test_dict = extract_tuple(test_latex)

            # 如果任一解析失败，则不等
            if not answer_dict or not test_dict:
                return 0, -1, -1, -1

            try:
                # 步骤1: 规范化字典的键，将LaTeX键转换为SymPy表达式键
                norm_answer_dict = {master_convert(k, 'Expression'): v for k, v in answer_dict.items()}
                norm_test_dict = {master_convert(k, 'Expression'): v for k, v in test_dict.items()}
            except Exception as e:
                # 如果键无法解析，则认为格式错误
                if debug_mode: print(f"Error normalizing tuple keys: {e}")
                return 0, -1, -1, -1

            # 步骤2: 比较规范化后的键集合是否完全等价
            if set(norm_answer_dict.keys()) != set(norm_test_dict.keys()):
                return 0, -1, -1, -1

            # 步骤3: 如果键集合等价，则逐个比较对应的值
            scores, rel_distances, tree_sizes, distance_numbers = 0, 0, 0, 0
            size = len(norm_answer_dict)
            if size == 0: # 两个都是空元组
                return 100, 0.0, 0, 0

            for sympy_key, answer_v_latex in norm_answer_dict.items():
                # 使用 SymPy 键安全地获取 test 的值
                test_v_latex = norm_test_dict[sympy_key]
                
                # 递归调用 SEED 比较值
                score, rel_distance, tree_size, distance_number = SEED(answer_v_latex, test_v_latex, 'Expression')
                scores += score
                # 安全地处理 -1 的情况
                if rel_distance != -1: rel_distances += rel_distance
                if tree_size != -1: tree_sizes += tree_size
                if distance_number != -1: distance_numbers += distance_number

            return scores / size, rel_distances / size, tree_sizes / size, distance_numbers / size
        
        elif t=='Interval':
            is_interval, answer_latex= judge_interval(answer_latex)
            is_interval, test_latex= judge_interval(test_latex)
            # if is_interval:t='Interval'
        elif t=='Numeric':
            answer_latex = extract_numeric_part(answer_latex)
            test_latex = extract_numeric_part(test_latex)

            answer_latex = re.sub(r'\\(?![a-zA-Z])', '', answer_latex)
            test_latex = re.sub(r'\\(?![a-zA-Z])', '', test_latex)

        answer_exp = master_convert(answer_latex, t)
        print(f"answer_exp: {answer_exp}")
        test_exp = master_convert(test_latex, t)
        print(f"test_exp: {test_exp}")
        if t =='Equation':
            answer_exp = Equation_standardize(answer_exp)
            test_exp = Equation_standardize(test_exp)

    except Exception as e:
        # raise e
        # print(f"Failed to convert input latex to sympy expression,please check it")
        # if debug_mode:
        #     raise LaTeXError(f"Fail to convert latex.\n GT:{answer_latex}\n GEN:{test_latex}")
        return 0,-1,-1,-1

    try:
        answer_exp,rep1=posify(answer_exp)
        answer_exp=time_simplify(answer_exp)
        
        test_exp,rep2=posify(test_exp)
        test_exp=time_simplify(test_exp)

        answer_exp=answer_exp.subs(rep1)
        test_exp=test_exp.subs(rep2)

        # if False:
        @timeout_decorator.timeout(10, timeout_exception=TimeoutError)
        def subtract_and_simplify_with_timeout(a, b):
            if isinstance(a, Expr) and isinstance(b, Expr):
                return simplify(expand(a - b))
            elif isinstance(a, Matrix) and isinstance(b, Matrix):
                if a.shape == b.shape:
                    return simplify(expand(a - b))
                else:
                    return 1  # 矩阵维度不一致
            else:
                return 1
        
        def safe_subtract_and_simplify(a, b):
            try:
                return subtract_and_simplify_with_timeout(a, b)
            except TimeoutError:
                print("  -> subtract_and_simplify timeout, returning 1")
                return 1  # 超时就认为不相等
            except Exception as e:
                print(f"  -> subtract_and_simplify error: {e}")
                return 1
        zero_exp=safe_subtract_and_simplify(answer_exp,test_exp)
        # zero_exp=time_simplify(expand(answer_exp-test_exp))
        

        # if answer_exp==test_exp or zero_exp==0:
        #     return 100,0.,0,0

        # if time_equal(answer_exp,test_exp):
        #     return 100,0.,0,0
        
        if t == "Equation":
            if answer_exp == test_exp or zero_exp == 0 or answer_exp + test_exp == 0:
                return 100, 0., 0, 0

        if answer_exp == test_exp or zero_exp == 0:
            return 100, 0., 0, 0

        if time_equal(answer_exp, test_exp):
            return 100, 0., 0, 0

    except Exception as e:
        raise e
        # print("Something happened during simplification,returning zero")
        # if debug_mode:
        #     raise SymPyError(f"Failed to simplify the sympy expression. Expressions: answer_exp={answer_exp}, test_exp={test_exp}")
        return 0,-1,-1,-1

    try:
        tree_answer=sympy_to_tree(answer_exp)
        tree_test=sympy_to_tree(test_exp)

    except Exception as e:
        # raise e
        # print("Failed to build expression tree,returning zero")
        # if debug_mode:
        #     raise SymPyError(f"Failed to build the sympy expression tree.\n GT:{answer_exp}\n GEN:{test_exp}")
        return 0,-1,-1,-1

    distance=ext_distance(
                tree_test,
                tree_answer,
                get_children=lambda x:x.get_children(),
                single_insert_cost=insert_func,
                insert_cost=insert_tree_func,
                single_remove_cost=remove_func, 
                remove_cost=remove_tree_func, 
                update_cost=update_func)    

    tree_size=calc_tree_size(tree_answer)
    distance_number=distance

    rel_distance=distance/tree_size
    
    # 如果是Numeric类型，使用数值比较逻辑
    if t == 'Numeric':
        score = numeric_score_calc(test_exp, answer_exp)

        return score, -1, -1, -1
    else:
        score = score_calc(distance_number, tree_size)

        return score,rel_distance,tree_size,distance_number

if __name__ == "__main__":
    # 示例测试：表达式相同应得满分 100
    gt = "$(2 \\arctan (\\frac{v_{0}}{V} \\sqrt{1-V^{2}}), \\pi)"
    pred = "\\left(2 \\arctan\\left(\\frac{v_0 \\sqrt{1 - V^2}}{V}\\right), \\pi\\right)"

    t = "Interval"  # 类型可以是 Expression, Equation, Tuple, Interval, Numeric

    score, rel_distance, tree_size, dist = SEED(gt, pred, t)

    print("\n=== Test Result ===")
    print(f"GT LaTeX:      {gt}")
    print(f"Predicted:     {pred}")
    print(f"Score:         {score}")
    print(f"Rel Distance:  {rel_distance}")
    print(f"Tree Size:     {tree_size}")
    print(f"Raw Distance:  {dist}")