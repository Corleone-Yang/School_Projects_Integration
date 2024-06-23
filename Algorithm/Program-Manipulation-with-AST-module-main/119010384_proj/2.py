import ast

class FunctionRemover(ast.NodeTransformer):
    def __init__(self, used_functions):
        self.used_functions = used_functions

    def visit_FunctionDef(self, node):
        if node.name not in self.used_functions:
            return None
        return node

def remove_unused_functions(tree):
    used_functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            used_functions.add(node.func.id)
    transformer = FunctionRemover(used_functions)
    new_tree = transformer.visit(tree)
    return new_tree

def getAssign(node):
    start = 0
    end = len(node.body)
    list = []
    for i in range(start, end):
        if isinstance(node.body[i] ,ast.Assign):
            list.append(i)
    return list

def getFunctionDef(node):
    start = 0
    end = len(node.body)
    list = []
    for i in range(start, end):
        if isinstance(node.body[i] ,ast.FunctionDef):
            list.append(i)
    return list

def getExpr(node):
    start = 0
    end = len(node.body)
    list = []
    for i in range(start, end):
        if isinstance(node.body[i] ,ast.Expr):
            list.append(i)
    return list

def collect_Assign_BinOp_ids(node, ids):
    if isinstance(node, ast.BinOp):
        if isinstance(node.left, ast.Name) and isinstance(node.left.ctx, ast.Load):
            ids.append(node.left.id)
        if isinstance(node.right, ast.Name) and isinstance(node.right.ctx, ast.Load):
            ids.append(node.right.id)
        collect_Assign_BinOp_ids(node.left, ids)
        collect_Assign_BinOp_ids(node.right, ids)
    elif isinstance(node, ast.AST):
        for child_node in ast.iter_child_nodes(node):
            collect_Assign_BinOp_ids(child_node, ids)

def slover(node, Assign_list, FunctionDef_list, Expr_list):
    

    count = 0
    globalVars_status = {}
    start = 0
    end = len(Assign_list) + len(FunctionDef_list) + len(Expr_list)
    
    # Assign
    for i in range(start, end):
        if(i in Assign_list):
            Assign_node = node.body[i]

            # Assign, node.value = Constant
            if(isinstance(Assign_node.value, ast.Constant)):
                for target in Assign_node.targets:
                    if(isinstance(target, ast.Name)):
                        globalVars_status[target.id] = 'defined'
            
            # Assign, node.value = Name
            if(isinstance(Assign_node.value, ast.Name)):
                if(globalVars_status.get(Assign_node.value.id) == 'defined'):
                    for target in Assign_node.targets:
                        if(isinstance(target, ast.Name) and globalVars_status.get(target.id) != 'undefined'):
                            globalVars_status[target.id] = 'defined'
                if(Assign_node.value.id not in globalVars_status or globalVars_status.get(Assign_node.value.id) == 'undefined'):
                    for target in Assign_node.targets:
                        if(isinstance(target, ast.Name)):
                            globalVars_status[target.id] = 'undefined'

            # Assign, node.value = BinOp
            if(isinstance(Assign_node.value, ast.BinOp)):
                ids = []
                collect_Assign_BinOp_ids(Assign_node, ids)
                judge = True
                for id in ids:
                    if(id not in globalVars_status or globalVars_status.get(id) == 'undefined'):
                        judge = False
                        count+=1
                if(judge == True):
                    for target in Assign_node.targets:
                        if(isinstance(target, ast.Name)) and globalVars_status.get(target.id) != 'undefined':
                            globalVars_status[target.id] = 'defined'
                elif(judge == False):
                    for target in Assign_node.targets:
                        if(isinstance(target, ast.Name)):
                            globalVars_status[target.id] = 'undefined'

        
        # Expr
        elif(i in Expr_list):
            # get the Expr node
            Expr_node = node.body[i]
            if(isinstance(Expr_node.value, ast.Call)):
                if(isinstance(Expr_node.value.func, ast.Name)):
                    func_id = Expr_node.value.func.id 
            func_position = - 1
            # get the relevant FunctionDef node
            for j in FunctionDef_list:
                if(node.body[j].name == func_id and j > func_position and j < i):
                    func_position = j
            FunctionDef_node = node.body[func_position]
            # get the situation of called function
            VarsList = []
            for arg in Expr_node.value.args:
                if(isinstance(arg, ast.Name)):
                    VarsList.append(arg.id)
                if(isinstance(arg, ast.Constant)):
                    VarsList.append(arg.value)
            for keyword in Expr_node.value.keywords:
                if(isinstance(keyword.value, ast.Name)):
                    VarsList.append(keyword.value.id)
                if(isinstance(keyword.value, ast.Constant)):
                    VarsList.append(keyword.value)
            # get the args in FunctionDef
            FuncList = []
            if(isinstance(FunctionDef_node, ast.FunctionDef)):
                for arg in FunctionDef_node.args.args:
                    FuncList.append(arg.arg)
            # the in function defined vars
            defined_infunVars = set()
            for j in range(0,len(VarsList)):
                if(isinstance(VarsList[j], ast.Constant)):
                    defined_infunVars.add(FuncList[j])
                if(globalVars_status.get(VarsList[j]) == 'defined'):
                    defined_infunVars.add(FuncList[j])
            for i in globalVars_status:
                if(globalVars_status.get(j) == 'defined'):
                    defined_infunVars.add(j)

            for body_node in FunctionDef_node.body:
                # infunction Assign
                if(isinstance(body_node, ast.Assign)):
                    # Assign, node.value = Constant
                    if(isinstance(body_node.value, ast.Constant)):
                        for target in body_node.targets:
                            if isinstance(target, ast.Name):
                                defined_infunVars.add(target.id)
                    # Assign, node.value = Name
                    if(isinstance(body_node.value, ast.Name)):
                        for target in body_node.targets:
                            if(body_node.value.id not in defined_infunVars):
                                count+=1  
                                if(isinstance(target, ast.Name)):
                                    defined_infunVars.remove(target.id)
                            elif(body_node.value.id in defined_infunVars):
                                if(isinstance(target, ast.Name)):
                                    defined_infunVars.add(target.id)
                    # Assign, node.value = BinOp
                    if(isinstance(body_node.value, ast.BinOp)):
                        ids = []
                        collect_Assign_BinOp_ids(body_node, ids)
                        judge = True
                        for id in ids:
                            if(id not in defined_infunVars):
                                judge = False
                                count+=1
                        if(judge == True):
                            for target in body_node.targets:
                                if(isinstance(target, ast.Name)):
                                    defined_infunVars.add(target.id)
                        elif(judge == False):
                            for target in body_node.targets:
                                if(isinstance(target, ast.Name)):
                                    defined_infunVars.discard(target.id)

                # infunction Expr
                elif(isinstance(body_node, ast.Expr)):
                    if(isinstance(body_node.value, ast.Call)):
                        for arg in body_node.value.args:
                            if(isinstance(arg, ast.Name)):
                                if(arg.id not in defined_infunVars):
                                    count+=1
                        for keyword in body_node.value.keywords:
                            if(isinstance(keyword.value, ast.Name)):
                                if(keyword.value.id not in defined_infunVars):
                                    count+=1

    return count

def main():
    code = input()
    code = "\n".join(code.split("\\n"))
    tree = ast.parse(code)
    tree = remove_unused_functions(tree)
    list1 = getAssign(tree)
    list2 = getFunctionDef(tree)
    list3 = getExpr(tree)
    count = slover(tree, list1, list2, list3)
    print(count)


if __name__ == "__main__":
    main()