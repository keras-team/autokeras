from mkdocs.autogen import parse_func_string


def test_parse_func_string():
    comment = """Fetches rows from a Bigtable.
Hello world.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.

    Retrieves rows pertaining to the given keys from the Table instance
    represented by big_table.  Silly things may happen if
    other_silly_variable is not None.



    Returns:
        A dict mapping keys to the corresponding table row data
        fetched. Each row is represented as a tuple of strings. For
        example:

        {'Serak': ('Rigel VII', 'Preparer'),
         'Zim': ('Irk', 'Invader'),
         'Lrrr': ('Omicron Persei 8', 'Emperor')}

        If a key from the keys argument is missing from the dictionary,
        then that row was not found in the table.

    Args:
        big_table: An open Bigtable Table instance.
        keys: A sequence of strings representing the key of each table row
            to fetch.
        other_silly_variable: Another optional variable, that has a much
            longer name than the other args, and which does nothing.

    Raises:
        IOError: An error occurred accessing the bigtable.Table object.


    """
    d = parse_func_string(comment)
    parts = ['short_description', 'long_description', 'Args', 'Attributes', 'Returns', 'Raises']
    for part in parts:
        print(part)
        print('-------')
        print(d[part])
        print('------------')
        print('------------')
        print('------------')
