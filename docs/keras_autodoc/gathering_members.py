import inspect
from inspect import isclass
from inspect import isfunction
from inspect import isroutine
from typing import List

from .utils import import_object


def get_classes(module, exclude: List[str] = None, return_strings: bool = True):
    """Get all the classes of a module.

    # Arguments

        module: The module to fetch the classes from. If it's a string, it
            should be in the dotted format. `'keras.layers'` for example.
        exclude: The names which will be excluded from the returned list. For
            example, `get_classes('keras.layers', exclude=['Dense', 'Conv2D'])`.
        return_strings: If False, the actual classes will be returned. Note that
            if you use aliases when building your docs, you should use strings.
            This is because the computed signature uses
            `__name__` and `__module__` if you don't provide a string as input.

    # Returns

     A list of strings or a list of classes.
    """
    return _get_all_module_element(module, exclude, return_strings, True)


def get_functions(
    module, exclude: List[str] = None, return_strings: bool = True
):
    """Get all the functions of a module.

    # Arguments

        module: The module to fetch the functions from. If it's a string, it
            should be in the dotted format. `'keras.backend'` for example.
        exclude: The names which will be excluded from the returned list. For
            example, `get_functions('keras.backend', exclude=['max'])`.
        return_strings: If False, the actual functions will be returned. Note
            that if you use aliases when building your docs, you should use
            strings.  This is because the computed signature uses `__name__` and
            `__module__` if you don't provide a string as input.

    # Returns

     A list of strings or a list of functions.
    """
    return _get_all_module_element(module, exclude, return_strings, False)


def get_methods(cls, exclude=None, return_strings=True):
    """Get all the method of a class.

    # Arguments

        cls: The class to fetch the methods from. If it's a
            string, it should be in the dotted format. `'keras.layers.Dense'`
            for example.
        exclude: The names which will be excluded from the returned list. For
            example, `get_methods('keras.Model', exclude=['save'])`.
        return_strings: If False, the actual methods will be returned. Note that
            if you use aliases when building your docs, you should use strings.
            This is because the computed signature uses
            `__name__` and `__module__` if you don't provide a string as input.

    # Returns

     A list of strings or a list of methods.
    """
    if isinstance(cls, str):
        cls_str = cls
        cls = import_object(cls)
    else:
        cls_str = f"{cls.__module__}.{cls.__name__}"
    exclude = exclude or []
    methods = []
    for _, method in inspect.getmembers(cls, predicate=isroutine):
        if method.__name__[0] == "_" or method.__name__ in exclude:
            continue
        if return_strings:
            methods.append(f"{cls_str}.{method.__name__}")
        else:
            methods.append(method)
    return methods


def _get_all_module_element(module, exclude, return_strings, class_):
    if isinstance(module, str):
        module = import_object(module)
    exclude = exclude or []
    module_data = []
    for name in dir(module):
        module_member = getattr(module, name)
        if not (isfunction(module_member) or isclass(module_member)):
            continue
        if name[0] == "_" or name in exclude:
            continue
        if module.__name__ not in module_member.__module__:
            continue
        if module_member in module_data:
            continue
        if class_ and not isclass(module_member):
            continue
        if not class_ and not isfunction(module_member):
            continue
        if return_strings:
            module_data.append(f"{module.__name__}.{name}")
        else:
            module_data.append(module_member)
    module_data.sort(key=id)
    return module_data
