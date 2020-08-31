

#Test datasets load correctly
#Test dimensiosn of datasets
#Test type of datasets

#Test URL status codes
def test_dataset_url():

    r = requests.get(TRAIN_URL, allow_redirects = True)
    assert(r.status_code == 200)

    r = requests.get(TEST_URL, allow_redirects = True)
    assert(r.status_code == 200)

    r = requests.get(CASP_10_URL, allow_redirects = True)
    assert(r.status_code == 200)

    r = requests.get(CASP_11_URL, allow_redirects = True)
    assert(r.status_code == 200)

def test_get_cullpdb_filtered():
    #test http status code 200 of URL
    #test file extension
    #.assertIsInstance(a, b)
#     if resp.status_code == 200: https://stackoverflow.com/questions/54087303/python-requests-how-to-check-for-200-ok
#     print ('OK!')
# else:
#     print ('Boo!')
    pass

def load_cul6133_filted_test():
    #assert size of each dimension in dataset is correct
    #assert 5278 * all_data = dimension of data
    #assert dimension of training and validation data
    pass

# def test_function():
# #     assert f() == 4
#
# def test_zero_division():
#     with pytest.raises(ZeroDivisionError):
# #         1 / 0
# import unittest
# class TestSum(unittest.TestCase):
#
#     def test_sum(self):
#         self.assertEqual(sum([1, 2, 3]), 6, "Should be 6")
#
#     def test_sum_tuple(self):
#         self.assertEqual(sum((1, 2, 2)), 6, "Should be 6")
