import ast

# create a new subclass of NodeTransformer that 
# overrides the visit_FunctionDef() method.
class FunctionRemover(ast.NodeTransformer):
    def __init__(self, used_functions):
        self.used_functions = used_functions

    def visit_FunctionDef(self, node):
        if node.name not in self.used_functions:
            return None
        return node

def remove_unused_functions(node):
    used_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            used_functions.add(node.func.id)
    transformer = FunctionRemover(used_functions)
    new_tree = transformer.visit(tree)
    return new_tree

# get all global variables and store them in a list
global_vars = set()
def append_global_vars(node):
    if isinstance(node, ast.FunctionDef) or isinstance(node, ast.expr):
        return
    elif isinstance(node, ast.Assign):
        for target in node.targets:
            if isinstance(target, ast.Name):
                global_vars.add(target.id)
    for child in ast.iter_child_nodes(node):
        append_global_vars(child)

# transformer - transform operators
def modify_ast(tree):
    class ModifyNode(ast.NodeTransformer):
        def visit_BinOp(self, node):
            if isinstance(node.op, ast.Add):
                new_op = ast.Sub()
            elif isinstance(node.op, ast.Sub):
                new_op = ast.Add()
            elif isinstance(node.op, ast.Mult):
                new_op = ast.Div()
            elif isinstance(node.op, ast.Div):
                new_op = ast.Mult()
            else:
                new_op = node.op
            
            return ast.BinOp(self.visit(node.left), new_op, self.visit(node.right))
        
        def visit_Compare(self, node):
            if isinstance(node.ops[0], ast.GtE):
                new_op = ast.Lt()
            elif isinstance(node.ops[0], ast.Lt):
                new_op = ast.GtE()
            elif isinstance(node.ops[0], ast.LtE):
                new_op = ast.Gt()
            elif isinstance(node.ops[0], ast.Gt):
                new_op = ast.LtE()
            else:
                new_op = node.ops[0]
            
            return ast.Compare(self.visit(node.left), [new_op], [self.visit(node.comparators[0])])
        
    transformer = ModifyNode()
    return transformer.visit(tree)


#main()
code = input()
code = "\n".join(code.split("\\n"))
tree = ast.parse(code)
append_global_vars(tree)
remove_unused_functions(tree)
new_tree = modify_ast(tree)
print(ast.unparse(new_tree))
for i in global_vars:
    print('print(%s)' % i)

