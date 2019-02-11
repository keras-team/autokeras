"""
  Test cases for functions in ancillary_utils.py
  -- kandasamy@cs.cmu.edu
"""

# pylint: disable=import-error
# pylint: disable=no-member
# pylint: disable=invalid-name

from ..utils import ancillary_utils
from ..utils.base_test_class import BaseTestClass, execute_tests


class AncillaryUtilsTestCase(BaseTestClass):
  """ Unit test class for ancillary utilities. """

  def setUp(self):
    """ Sets up attributes. """
    # For dictionary
    self.dict_1 = {'a':1, 'b':2, 'c':3}
    self.dict_2 = {'b':2, 'a':1, 'c':3}
    self.dict_3 = {'a':4, 'b':5, 'c':6}
    self.dict_4 = {'d':1, 'e':2, 'f':3}
    # Lists
    self.list_1 = [1, 2, 3, 9, 10]
    self.list_2 = [1, 3, 3, 4, 19]
    self.list_3 = [1, 5, 8, 3]
    self.list_4 = [-x for x in self.list_1]
    self.list_5 = [-x for x in self.list_2]

  def test_compare_dict(self):
    """ Test compare_dict function. """
    self.report('dicts_are_equal')
    assert ancillary_utils.dicts_are_equal(self.dict_1, self.dict_2)
    assert not ancillary_utils.dicts_are_equal(self.dict_1, self.dict_3)
    assert not ancillary_utils.dicts_are_equal(self.dict_1, self.dict_4)

  def test_nondecreasing(self):
    """ Unit tests for is_nondecreasing. """
    assert ancillary_utils.is_nondecreasing(self.list_1)
    assert ancillary_utils.is_nondecreasing(self.list_2)
    assert not ancillary_utils.is_nondecreasing(self.list_3)
    assert not ancillary_utils.is_nondecreasing(self.list_4)
    assert not ancillary_utils.is_nondecreasing(self.list_5)

  def test_nonincreasing(self):
    """ Unit tests for is_nonincreasing. """
    assert not ancillary_utils.is_nonincreasing(self.list_1)
    assert not ancillary_utils.is_nonincreasing(self.list_2)
    assert not ancillary_utils.is_nonincreasing(self.list_3)
    assert ancillary_utils.is_nonincreasing(self.list_4)
    assert ancillary_utils.is_nonincreasing(self.list_5)

  def test_decreasing(self):
    """ Unit tests for is_decreasing. """
    assert not ancillary_utils.is_decreasing(self.list_1)
    assert not ancillary_utils.is_decreasing(self.list_2)
    assert not ancillary_utils.is_decreasing(self.list_3)
    assert ancillary_utils.is_decreasing(self.list_4)
    assert not ancillary_utils.is_decreasing(self.list_5)

  def test_increasing(self):
    """ Unit tests for is_increasing. """
    assert ancillary_utils.is_increasing(self.list_1)
    assert not ancillary_utils.is_increasing(self.list_2)
    assert not ancillary_utils.is_increasing(self.list_3)
    assert not ancillary_utils.is_increasing(self.list_4)
    assert not ancillary_utils.is_increasing(self.list_5)


if __name__ == '__main__':
  execute_tests()

