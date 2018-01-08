from numpydoc.docscrape import NumpyDocString
import re


class Photo():
    """
    Array with associated photographic information.


    Parameters
    ----------
    x : type
        Description of parameter `x`.
    y
        Description of parameter `y` (with type not specified)

    Attributes
    ----------
    exposure : float
        Exposure in seconds.

    Methods
    -------
    colorspace(c='rgb')
        Represent the photo in the given colorspace.
    gamma(n=1.0)
        Change the photo's gamma exposure.

    """

    def __init__(x, y):
        print("Snap!")


# doc = NumpyDocString(Photo.__doc__)
# print(doc["Summary"])
# print(doc["Parameters"])
# print(doc["Attributes"])
# print(doc["Methods"])

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
        if re.match(r'^\s*$',parts[ind]):
            ind += 1
        else:
            break
    return ind


# check if comment is None or len(comment) == 0 return {}
def parse_func_string(comment):
    comments = {}
    paras = ('Args','Attributes','Returns','Raises')
    comment_parts = ['short_description', 'long_description', 'Args', 'Attributes', 'Returns', 'Raises']
    for x in comment_parts:
        comments[x] = None

    parts = re.split(r'\n', comment)
    ind = 1
    while ind < len(parts):
        if re.match(r'^\s*$',parts[ind]):
            break
        else:
            ind += 1

    comments['short_description'] ='\n'.join(['\n'.join(re.split('\n\s+', x.strip())) for x in parts[0:ind]]).strip(':\n\t ')
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
            part = delete_space(parts, start+1, ind - 1)
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

comment = """Fetches rows from a Bigtable.
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

# comment = """Summary of class here.
#
#     Longer class information....
#     Longer class information....
#
#     Attributes:
#         likes_spam: A boolean indicating if we like SPAM or not.
#         eggs: An integer count of the eggs we have laid.
#     """
# d = parse_func_string(comment)
# parts = ['short_description','long_description','Args','Attributes','Returns','Raises']
# for part in parts:
#     print(part)
#     print('-------')
#     print(d[part])
#     print('------------')
#     print('------------')
#     print('------------')
