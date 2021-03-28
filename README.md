This is the project repository for our IEEE VR 2021 (Journal Track) paper:
Xianglong Feng, Zeyang Bao, Sheng Wei, LiveObj: Object Semantics-based Viewport Prediction for Live Mobile Virtual Reality Streaming, IEEE Conference on Virtual Reality and 3D User Interfaces (VR), Journal Track, March 2021.
The repository includes the following materials about LiveObj:
### Demo Video:
In each video, the red rectangle represents the actual user view, and the yellow rectangle represents the viewport predicted by the Velocity method for comparison. For the LiveObj method, the blue rectangle represents the detected objects, the green tiles represent the predicted viewport, and the purple tiles represent the ones that are not selected into the predicted viewport.
### Source Code:
Under this folder, there are two source code files:
- “LiveObj.py”: the source code for LiveObj, based on the implementation of YOLOv3.
- “darknet.py”: the source code for darknet, which is the backbone network of YOLOv3.
Please run “LiveObj.py” to execute our live viewport prediction approach.
Before you run the code:
- Please download the latest pre-trained model from https://pjreddie.com/darknet/yolo/.
- Please download the dataset following the paper: Chenglei Wu, Zhihao Tan, Zhi Wang, and Shiqiang Yang. A Dataset for Exploring User Behaviors in VR Spherical Video Streaming. ACM Multimedia Systems Conference (MMSys), pp. 193–198, 2017.