This page contains more information about the two datasets used in the NeurIPS 2019 paper **Multiview Aggregation for Learning Category-Specific Shape Reconstruction**.

![ShapeNetCOCO](xnocs.png)

![ShapeNetPlain](xnocs.png)

# ShapeNetCOCO v1

Download link: [shapenetcoco_v1.zip (172 GB)][1]

ShapeNetCOCO v1 contains data for ShapeNetCore objects in the following categories:
- Cars (02958343) from ShapeNetCore.v2
- Airplanes (02691156) from ShapeNetCore.v1
- Chairs (0301627) from ShapeNetCore.v1

Each ShapeNet model is rendered at 640x480 from 20 different views sampled at a random distance from the object. The background of each RGB rendering (for the first intersection in the depth peeling) is a random image from the COCO dataset. Lights in the scene are placed with respect to the camera for each view, so shading between views is not identical. The data is split into a train and test split by shapes.

Each model directory contains the following files for each view:
- CameraPose.json : contains the camera position and rotation quaternion in world coordinates from the Unity game engine. The ShapeNet model is placed at (0, 0, 0) in world coordinates for every render.
- Color_00.png and Color_01.png : contain the RGB renderings for the first and last intersection in the depth peeling, respectively.
- NOXRayTL_00.png and NOXRayTL_01.png : contain the NOCS maps for the first and last intersection in the depth peeling, respectively.

The camera used to render all images has focal length 617.1 and the camera center (c_x, c_y) is (315, 242).


# ShapeNetPlain v1

Download link: [shapenetplain_v1.zip (5 GB)][2]

ShapeNetPlain v1 contains data for ShapeNetCore objects in the following categories:
- Cars (02958343) from ShapeNetCore.v2
- Airplanes (02691156) from ShapeNetCore.v1
- Chairs (0301627) from ShapeNetCore.v1

Each ShapeNet model is rendered at 640x480 from 5 different views sampled at a fixed distance from the object. The background of each RGB rendering is plain white. Lights in the scene are placed with respect to the camera for each view, so shading between views is not identical. The data is split into a train and test split by shapes.

Each model directory contains the following files for each view:
- CameraPose.json : contains the camera position and rotation quaternion in world coordinates from the Unity game engine.  The ShapeNet model is placed at (0, 0, 0) in world coordinates for every render.
- Color_00.png and Color_01.png : contain the RGB renderings for the first and last intersection in the depth peeling, respectively.
- NOXRayTL_00.png and NOXRayTL_01.png : contain the NOCS maps for the first and last intersection in the depth peeling, respectively.

The camera used to render all images has focal length 617.1 and the camera center (c_x, c_y) is (315, 242).


[1]: http://download.cs.stanford.edu/orion/xnocs/shapenetcoco_v1.zip
[2]: http://download.cs.stanford.edu/orion/xnocs/shapenetplain_v1.zip
