from numpydoc.docscrape import NumpyDocString


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

doc = NumpyDocString(Photo.__doc__)
print(doc["Summary"])
print(doc["Parameters"])
print(doc["Attributes"])
print(doc["Methods"])