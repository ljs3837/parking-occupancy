import copy
import unittest
import numpy as np
from parking.training.occlusion import occlude_slots

class OcclusionTests(unittest.TestCase):
    def test_partial_deterministic_and_labels_unchanged(self):
        image = np.full((200,200,3),220,np.uint8)
        polygons = [{"points": [[.2,.2],[.8,.2],[.8,.8],[.2,.8]]}]
        saved = copy.deepcopy(polygons)
        result, count = occlude_slots(image,polygons,seed=2)
        again, _ = occlude_slots(image,polygons,seed=2)
        self.assertEqual(count,1)
        self.assertTrue(np.array_equal(result,again))
        self.assertTrue(np.all(image == 220))
        self.assertEqual(polygons,saved)
        fraction = np.any(result != image,axis=2).mean()
        self.assertGreater(fraction,0)
        self.assertLess(fraction,.15)

    def test_empty_labels_leave_image(self):
        image = np.zeros((20,20,3),np.uint8)
        result,count = occlude_slots(image,[])
        self.assertEqual(count,0)
        self.assertTrue(np.array_equal(result,image))
