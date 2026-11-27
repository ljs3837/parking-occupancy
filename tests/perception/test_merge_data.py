import tempfile,unittest,json
from pathlib import Path
from unittest.mock import patch
import numpy as np,cv2
from parking.training import prepare

class MergeDataTests(unittest.TestCase):
    def test_repetition_happens_after_split(self):
        samples=[{'rel':f'cctv/jongsu/image{i}.jpg'} for i in range(10)]+[{'rel':f'cctv_image/cam/image{i}.jpg'} for i in range(4)]+[{'rel':f'huggingface/image{i}.jpg'} for i in range(5)]+[{'rel':f'kaggle/parking-space-top-view/{i}.jpg'} for i in range(3)]+[{'rel':f'kaggle/mmu-parking-slots/image{i}.jpg'} for i in range(6)]
        rows,n_cctv,n_kaggle=prepare._split_rows(samples,cctv_repeat=3)
        validation=[s['rel'] for split,s in rows if split=='val']
        train=[s['rel'] for split,s in rows if split=='train']
        self.assertEqual(n_cctv,10)
        self.assertEqual(n_kaggle,3)
        self.assertEqual(len(validation),13)
        self.assertTrue(all(name.startswith('cctv/') or name.startswith('kaggle/parking-space-top-view/') for name in validation))
        self.assertTrue(all(name.startswith(('cctv_image/','huggingface/','kaggle/mmu-parking-slots/')) for name in train))
        self.assertFalse(set(validation)&set(train))
        self.assertTrue(all(train.count(name)==1 for name in train))

    def test_augmentation_preserves_validation_and_output_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);run_root=root/'runs';samples=[]
            for i in range(5):
                image=root/f'image{i}.png'
                cv2.imencode('.png',np.full((120,160,3),i*35,np.uint8))[1].tofile(str(image))
                samples.append({'rel':f'cctv/jongsu/image{i}.png','path':image,'polygons':[{'points':[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'label':'parking_space'}]})
            train_image=root/'train.png'
            cv2.imencode('.png',np.full((120,160,3),200,np.uint8))[1].tofile(str(train_image))
            samples.append({'rel':'blackbox/lot/train.png','path':train_image,'polygons':[{'points':[[.2,.2],[.8,.2],[.8,.8],[.2,.8]],'label':'parking_space'}]})
            with patch.object(prepare,'RUNS_DIR',run_root), patch.object(prepare,'occlude_slots',lambda image, polygons, seed=42: (image, 1)):
                output=run_root/'dataset'
                prepare.build_yolo_dataset(lambda _:None,samples=samples,output_dir=output,cctv_repeat=2)
                self.assertEqual(len(list((output/'images/val').glob('*'))),5)
                train_sources={p.stem.split('_')[0] for p in (output/'images/train').glob('*')}
                val_sources={p.stem.split('_')[0] for p in (output/'images/val').glob('*')}
                self.assertFalse(train_sources&val_sources)
                augmented=list((output/'labels/train').glob('*_occluded.txt'))
                self.assertTrue(augmented)
                for label in augmented:
                    base=label.with_name(label.name.replace('_occluded',''))
                    self.assertEqual(label.read_bytes(),base.read_bytes())
                with self.assertRaises(ValueError):
                    prepare.build_yolo_dataset(lambda _:None,samples=samples,output_dir=root,cctv_repeat=1)
                self.assertTrue(samples[0]['path'].is_file())

if __name__=='__main__':unittest.main()
