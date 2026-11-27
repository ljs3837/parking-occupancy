import unittest
import numpy as np
from parking.training.improved_data import ordered_quad,roi_crop,label_rows
from parking.perception.tiling import merge_candidates,tiled_candidates,polygon_iou

class ImprovedGeometryTests(unittest.TestCase):
    def test_closed_quad_and_corner_format(self):
        p=[[.2,.2],[.8,.2],[.8,.8],[.2,.8],[.2,.2]]
        q=ordered_quad(p)
        self.assertEqual(q.shape,(4,2))
        line=label_rows([{"points":q}],"pose").split()
        self.assertEqual(len(line),17)
        self.assertEqual([float(line[i]) for i in (7,10,13,16)],[2,2,2,2])

    def test_crossing_geometry_rejected(self):
        self.assertIsNone(ordered_quad([[0,0],[1,1],[1,0],[0,1]]))

    def test_roi_coordinate_roundtrip_and_no_source_mutation(self):
        image=np.zeros((200,300,3),np.uint8)
        original=[[.3,.4],[.6,.4],[.6,.7],[.3,.7]]
        roi,polys,bounds=roi_crop(image,[{"points":original}])
        x,y,x2,y2=bounds
        recovered=(np.array(polys[0]["points"])*[x2-x,y2-y]+[x,y])/[300,200]
        np.testing.assert_allclose(recovered,original)
        self.assertLess(roi.shape[1],300)

    def test_polygon_duplicate_suppression_preserves_adjacent_slots(self):
        a={"points":[[0,0],[10,0],[10,20],[0,20]],"xyxy":(0,0,10,20),"conf":.8}
        duplicate={**a,"conf":.5}
        neighbor={**a,"points":[[10,0],[20,0],[20,20],[10,20]],"conf":.7}
        self.assertEqual(len(merge_candidates([a,duplicate,neighbor])),2)
        self.assertEqual(polygon_iou(a["points"],a["points"]),1.)

    def test_tiling_offsets_without_annotations(self):
        frame=np.zeros((200,200,3),np.uint8)
        calls=[]
        def predict(image):
            calls.append(image.shape)
            if image.shape[0]==200:return []
            return [{"xyxy":(20,20,40,40),"points":[[20,20],[40,20],[40,40],[20,40]],"conf":.8}]
        result=tiled_candidates(frame,predict)
        self.assertEqual(len(calls),5)
        self.assertEqual(len(result),4)
        self.assertIn((90,90,110,110),[d["xyxy"] for d in result])

class PolygonMetricTests(unittest.TestCase):
    def test_duplicate_predictions_cannot_match_same_ground_truth(self):
        from parking.training.polygon_metrics import score_polygons,summarize_scores
        quad=[[10,10],[60,10],[60,70],[10,70]]
        score=score_polygons([{"points":quad,"conf":.9},{"points":quad,"conf":.8}],[quad],100,100)
        self.assertEqual((score["tp"],score["fp"],score["fn"]),(1,1,0))
        self.assertEqual(summarize_scores([score])["precision"],.5)

class LiveCornerDecodingTests(unittest.TestCase):
    def test_corner_model_is_decoded_without_segmentation_masks(self):
        from types import SimpleNamespace
        import torch
        from parking.perception.live import LiveDetector
        points=torch.tensor([[[10.,10.],[60.,10.],[60.,70.],[10.,70.]]])
        box=SimpleNamespace(cls=torch.tensor([0.]),xyxy=torch.tensor([[10.,10.,60.,70.]]),conf=torch.tensor([.8]))
        result=SimpleNamespace(boxes=[box],names={0:"parking_space"},masks=None,keypoints=SimpleNamespace(xy=points))
        calls=[]
        def predict(image,**kwargs):
            calls.append(kwargs)
            return [result]
        model=SimpleNamespace(predict=predict)
        detected=LiveDetector()._yolo_boxes(np.zeros((100,100,3),np.uint8),model,"occupancy",imgsz=960)
        self.assertEqual(detected[0]["points"],points[0].tolist())
        self.assertEqual(calls[0]["imgsz"],960)
        self.assertEqual(calls[0]["conf"],0.375)

    def test_sidebar_confidence_reaches_vehicle_and_space_inference(self):
        from types import SimpleNamespace
        import torch
        from parking.perception.live import LiveDetector
        box=SimpleNamespace(cls=torch.tensor([2.]),xyxy=torch.tensor([[10.,10.,60.,70.]]),conf=torch.tensor([.8]))
        result=SimpleNamespace(boxes=[box],names={2:"car"},masks=None,keypoints=None)
        calls=[]
        def predict(image,**kwargs):
            calls.append(kwargs)
            return [result]
        model=SimpleNamespace(predict=predict)
        frame=np.zeros((100,100,3),np.uint8)
        detector=LiveDetector()
        detector.set_inference(conf=0.5)
        detector._vehicle_model=model
        detector._ready=True
        detector.predict(frame)
        self.assertEqual(calls[-1]["conf"],0.5)
        self.assertEqual(calls[-1]["imgsz"],640)
        detector.kind="occupancy"
        detector._space_model=model
        detector.predict_spaces(frame)
        self.assertEqual(calls[-1]["conf"],0.5)
        self.assertEqual(calls[-1]["imgsz"],640)

class EnclosingQuadrilateralTests(unittest.TestCase):
    def test_partial_mask_notch_can_be_regularized_without_right_angle_constraint(self):
        import cv2
        if not hasattr(cv2,"approxPolyN"):
            self.skipTest("Optional enclosing polygon fitter unavailable")
        from parking.perception.lines import refine_spaces
        points=[[30,30],[70,30],[70,110],[130,110],[130,30],[170,30],[170,170],[30,170]]
        detection={"label":"parking_space","conf":.8,"points":points,"xyxy":(30,30,170,170)}
        blank=np.zeros((200,200,3),np.uint8)
        self.assertEqual(refine_spaces(blank,[detection],snap_lines=False),[])
        result=refine_spaces(blank,[detection],snap_lines=False,quad_fit="enclosing")
        self.assertEqual(len(result),1)
        self.assertEqual(len(result[0]["points"]),4)
        self.assertTrue(result[0]["geometry_regularized"])
