import ast
import os.path
from mkdocs.autogen import parse_func_string


def get_func_comments(function_definitions):
    for f in function_definitions:
        print('***********function**************')
        print(f.name+" : ")
        print(parse_func_string(ast.get_docstring(f)))


def get_comments_str(file_name):
    with open(file_name) as fd:
        file_contents = fd.read()
    module = ast.parse(file_contents)
    function_definitions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    get_func_comments(function_definitions)

    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for class_def in class_definitions:
        print('-----------class----------------')
        print(class_def.name+" : ")
        print(parse_func_string(ast.get_docstring(class_def)))
        method_definitions = [node for node in class_def.body if isinstance(node, ast.FunctionDef)]
        get_func_comments(method_definitions)

def extract_comments(directory):
    for parent, dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            if os.path.splitext(file_name)[1] == '.py':
                get_comments_str(os.path.join(parent,file_name))


extract_comments('/Users/tao/PycharmProjects/autokeras/tests/google_python_style')

