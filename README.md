# Multiple-Object-Tracking

******************************************************
### To Run Detectron (Currently Using Faster R-CNN) on Video:
(Download model_final.pkl [R-50-C4] at https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md)
```
python Detectron.pytorch/tools/infer_video_stream.py --dataset=coco --cfg=Detectron.pytorch/configs/baselines/e2e_mask_rcnn_R-50-C4_1x.yaml --load_detectron=[path to model_final.pkl] --input_video_path=[input video path] --output_path=[detectron output path]
```
************************************************************

### To Run Greedy Tracker given Detectron Outputs
```
python  src/anomaly_db_bdd_greedy.py --video_path=[input video path] --detections_path=[detectron output path] --video_out_path=[output video path] --anomaly_out_dir=Detectron.pytorch/video_inference/anomalies
```
************************************************************

### To Run DaSiamRPN Tracker given Detectron Outputs (currently only uses first frame bounding boxes)
```
python code/track_multiple/DaSiamRPN_MultiTracker.py --detections=[detectron output path] --video_path=[input video path] --output_folder=[tracks output folder path]
```
