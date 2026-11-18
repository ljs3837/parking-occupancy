import unittest
import cv2
import numpy as np
from parking.perception.lines import refine_spaces

class LineRefinementTests(unittest.TestCase):
    def detection(self, points=None, label="parking_space"):
        return {"points": points or [[45,45],[155,45],[155,155],[45,155]],
                "xyxy": (45,45,155,155), "label": label, "conf": .8}

    def test_blank_keeps_original(self):
        d = self.detection()
        result = refine_spaces(np.zeros((200,200,3), np.uint8), [d])[0]
        self.assertEqual(result["points"], d["points"])
        self.assertFalse(result["line_refined"])
        self.assertNotIn("line_refined", d)

    def test_painted_rectangle_moves_toward_lines(self):
        image = np.zeros((200,200,3), np.uint8)
        cv2.rectangle(image, (50,50), (150,150), (255,255,255), 2)
        result = refine_spaces(image, [self.detection()])[0]
        self.assertTrue(result["line_refined"])
        self.assertEqual(result["line_supported_edges"], 4)
        self.assertLess(abs(cv2.contourArea(np.float32(result["points"])) - 10000), 2100)
        self.assertEqual(result["conf"], .8)

    def test_interrupted_paint_is_supported(self):
        image = np.zeros((200,200,3),np.uint8)
        cv2.rectangle(image,(50,50),(150,150),(255,255,255),2)
        image[75:120, 40:60] = 0
        result = refine_spaces(image,[self.detection()])[0]
        self.assertTrue(result["line_refined"])
        self.assertGreaterEqual(result["line_supported_edges"], 3)

    def test_vehicle_unchanged(self):
        d = self.detection(label="car")
        self.assertEqual(refine_spaces(np.full((200,200,3),255,np.uint8), [d]), [d])

    def test_perspective_not_forced_to_rectangle(self):
        image = np.zeros((220,220,3), np.uint8)
        polygon = np.int32([[65,35],[140,50],[175,175],[30,170]])
        cv2.polylines(image, [polygon], True, (255,255,255), 2)
        d = self.detection((polygon + np.array([3,2])).tolist())
        result = refine_spaces(image, [d])[0]
        self.assertTrue(result["line_refined"])
        q = np.float32(result["points"])
        lengths = np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        self.assertGreater(lengths.max()/lengths.min(), 1.4)

    def test_axis_aligned_box_without_mask_is_not_guessed(self):
        d = self.detection()
        d["points"] = None
        self.assertEqual(refine_spaces(np.zeros((200,200,3),np.uint8), [d]), [])
        fallback = refine_spaces(np.zeros((200,200,3),np.uint8), [d], strict_quadrilaterals=False)
        self.assertEqual(fallback[0]["points"], None)

if __name__ == "__main__":
    unittest.main()
