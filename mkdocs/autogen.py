import ast
import os
import re


def delete_space(parts, start, end):
    if start > end or end >= len(parts):
        return None
    count = 0
    while count < len(parts[start]):
        if parts[start][count] == ' ':
            count += 1
        else:
            break
    return '\n'.join(y for y in [x[count:] for x in parts[start:end + 1] if len(x) > count])


def change_args_to_dict(string):
    if string is None:
        return None
    ans = []
    strings = string.split('\n')
    ind = 1
    start = 0
    while ind <= len(strings):
        if ind < len(strings) and strings[ind].startswith(" "):
            ind += 1
        else:
            if start < ind:
                ans.append('\n'.join(strings[start:ind]))
            start = ind
            ind += 1
    d = {}
    for line in ans:
        if ":" in line and len(line) > 0:
            lines = line.split(":")
            d[lines[0]] = lines[1].strip()
    return d


def remove_next_line(comments):
    for x in comments:
        if comments[x] is not None and '\n' in comments[x]:
            comments[x] = ' '.join(comments[x].split('\n'))
    return comments


def skip_space_line(parts, ind):
    while ind < len(parts):
        if re.match(r'^\s*$', parts[ind]):
            ind += 1
        else:
            break
    return ind


# check if comment is None or len(comment) == 0 return {}
def parse_func_string(comment):
    if comment is None or len(comment) == 0:
        return {}
    comments = {}
    paras = ('Args', 'Attributes', 'Returns', 'Raises')
    comment_parts = ['short_description', 'long_description', 'Args', 'Attributes', 'Returns', 'Raises']
    for x in comment_parts:
        comments[x] = None

    parts = re.split(r'\n', comment)
    ind = 1
    while ind < len(parts):
        if re.match(r'^\s*$', parts[ind]):
            break
        else:
            ind += 1

    comments['short_description'] = '\n'.join(['\n'.join(re.split('\n\s+', x.strip())) for x in parts[0:ind]]).strip(
        ':\n\t ')
    ind = skip_space_line(parts, ind)

    start = ind
    while ind < len(parts):
        if parts[ind].strip().startswith(paras):
            break
        else:
            ind += 1
    long_description = '\n'.join(['\n'.join(re.split('\n\s+', x.strip())) for x in parts[start:ind]]).strip(':\n\t ')
    comments['long_description'] = long_description

    ind = skip_space_line(paras, ind)
    while ind < len(parts):
        if parts[ind].strip().startswith(paras):
            start = ind
            start_with = parts[ind].strip()
            ind += 1
            while ind < len(parts):
                if parts[ind].strip().startswith(paras):
                    break
                else:
                    ind += 1
            part = delete_space(parts, start + 1, ind - 1)
            if start_with.startswith(paras[0]):
                comments[paras[0]] = change_args_to_dict(part)
            elif start_with.startswith(paras[1]):
                comments[paras[1]] = change_args_to_dict(part)
            elif start_with.startswith(paras[2]):
                comments[paras[2]] = part
            elif start_with.startswith(paras[3]):
                comments[paras[3]] = part
            ind = skip_space_line(parts, ind)
        else:
            ind += 1

    remove_next_line(comments)
    return comments


sample_comment = """Fetches rows from a Bigtable.
Hello world.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.



   
    Args:
        big_table: An open Bigtable Table instance.
        keys: A sequence of strings representing the key of each table row
            to fetch.
        other_silly_variable: Another optional variable, that has a much
            longer name than the other args, and which does nothing.

    Raises:
        IOError: An error occurred accessing the bigtable.Table object.


    """


def to_md(comment_dict):
    doc = ''
    if 'short_description' in comment_dict:
        doc += comment_dict['short_description']
        doc += '\n'

    if 'long_description' in comment_dict:
        doc += comment_dict['long_description']
        doc += '\n'

    if 'Args' in comment_dict and comment_dict['Args'] is not None:
        doc += '####Args\n'
        for arg, des in comment_dict['Args'].items():
            doc += '**' + arg + '**: ' + des + '\n\n'

    if 'Attributes' in comment_dict and comment_dict['Attributes'] is not None:
        doc += '####Attributes\n'
        for arg, des in comment_dict['Attributes'].items():
            doc += '**' + arg + '**: ' + des + '\n\n'

    if 'Returns' in comment_dict and comment_dict['Returns'] is not None:
        doc += '####Returns\n'
        doc += comment_dict['Returns']
        doc += '\n'
    return doc


def get_func_comments(function_definitions):
    doc = ''
    for f in function_definitions:
        temp_str = to_md(parse_func_string(ast.get_docstring(f)))
        if temp_str != '':
            doc += '###' + f.name + '\n' + temp_str

    return doc


def get_comments_str(file_name):
    with open(file_name) as fd:
        file_contents = fd.read()
    module = ast.parse(file_contents)
    function_definitions = [node for node in module.body if isinstance(node, ast.FunctionDef)]
    doc = get_func_comments(function_definitions)

    class_definitions = [node for node in module.body if isinstance(node, ast.ClassDef)]
    for class_def in class_definitions:
        temp_str = to_md(parse_func_string(ast.get_docstring(class_def)))
        method_definitions = [node for node in class_def.body if isinstance(node, ast.FunctionDef)]
        temp_str += get_func_comments(method_definitions)
        if temp_str != '':
            doc += '##' + class_def.name + '\n' + temp_str
    return doc


def extract_comments(directory):
    for parent, dir_names, file_names in os.walk(directory):
        for file_name in file_names:
            if os.path.splitext(file_name)[1] == '.py' and file_name != '__init__.py':
                # with open
                doc = get_comments_str(os.path.join(parent, file_name))
                output_file = open(os.path.join('mkdocs/docs', file_name[:-3] + '.md'), 'w')
                output_file.write(doc)
                output_file.close()


extract_comments('autokeras')
