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

list.txt contains the configuration for images, homographies, and stitching rows. Ensure it follows the required format detailed below.

Input List Format
-----------------
The input list file (list.txt) should be structured as follows:

- images
0 path_to_image_0
1 path_to_image_1
2 path_to_image_2
...
N path_to_image_N
- homographies
0 1
0 2
2 N
1 N
- rows
0 1
2 N

where 
'images' section contains an enumerated list of images to be stitched, starting from 0.

'homographies' section lists pairs of images that overlap, for which homographies will be computed. Each pair should appear on a new line.

'rows' section defines the structure of the image stitching. Each line represents a row of images.

Example Configurations
----------------------
For stitching four images left-to-right in a single row:
- images
0 path/image0.png
1 path/image1.png
2 path/image2.png
3 path/image3.png
- homographies
0 1
1 2
2 3
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
- homographies
0 1
1 2
0 3
3 4
1 4
4 5
2 5
- rows
0 1 2
3 4 5

Notes
-----
Ensure all paths in the images section are correct and point to the respective images.
Each section (images, homographies, rows) must start with a - followed by the section name.
The list is case-sensitive, and section names should be written exactly as shown (images, homographies, rows).
You can find simple example in test_data directory