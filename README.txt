mapstitcher
===========

Allows to stitch arbitrary large images.

Installation
------------

In command line run:

  $ python3 -m venv ${PATH_TO_VENV}
  $ source ${PATH_TO_VENV}/bin/activate
  $ pip install --upgrade pip && pip install -r requirements.txt


Running
-------

To run the image stitching process, use the following command:

```bash
python image_stitch_batch.py --list list.txt

list.txt contains the configuration for images, and stitching rows. Ensure it follows the required format detailed below.

Parameters:
--list 			required, File and processing list
--optimization-model 	optional, default='affine', Optimization model homography or affine
--matching-algorithm optional, default='loftr', Matching algorithm loftr or sift
--loftr-model optional, default='outdoor', Model for LOFTR - outdoor or indoor
--flow-alg 		optional, default='raft', Optical Flow algorithm cv or raft
--subsample-flow 	optional, default=2.0, Subsample flow
--vram-size		optional, default=8.0, GPU VRAM size in GB, more can speed-up raft optflow computation 
--max-matches		optional, default=800, Maximum number of matches per image pair (for optimization)
--output		optional, default='result.jp2', Output file
--silent		optional, default=False, Kill any console output

Input List Format
-----------------
The input list file (list.txt) should be structured as follows:

- images
0 path_to_image_0
1 path_to_image_1
2 path_to_image_2
...
N path_to_image_N
- rows
0 1
2 N

where 
'images' section contains an enumerated list of images to be stitched, starting from 0.

'rows' section defines the structure of the image stitching. Each line represents a row of images.

Example Configurations
----------------------
For stitching four images left-to-right in a single row:
- images
0 path/image0.png
1 path/image1.png
2 path/image2.png
3 path/image3.png
- rows
0 1 2 3

For stitching six images with three on the top row and three on the bottom row (assuming overlap right and bottom):
- images
0 path/image0.png
1 path/image1.png
2 path/image2.png
3 path/image3.png
4 path/image4.png
5 path/image5.png
- rows
0 1 2
3 4 5

Notes
-----
Ensure all paths in the images section are correct and point to the respective images.
Each section (images, rows) must start with a - followed by the section name.
The list is case-sensitive, and section names should be written exactly as shown (images, rows).
You can find simple example in test_data directory
