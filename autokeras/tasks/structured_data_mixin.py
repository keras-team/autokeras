import pandas as pd


class StructuredDataMixin(object):

    def check(self, column_names, column_types):
        if column_types:
            for column_type in column_types.values():
                if column_type not in ['categorical', 'numerical']:
                    raise ValueError(
                        'Column_types should be either "categorical" '
                        'or "numerical", but got {name}'.format(name=column_type))
        if column_names and column_types:
            for column_name in column_types:
                if column_name not in column_names:
                    raise ValueError('Column_names and column_types are '
                                     'mismatched. Cannot find column name '
                                     '{name} in the data.'.format(name=column_name))

    def read_for_predict(self, x):
        if isinstance(x, str):
            x = pd.read_csv(x)
            if self._target_col_name in x:
                x.pop(self._target_col_name)
        return x
