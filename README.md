This project uses computer vision to control the volume on your device.

It calculates the length of the distance between two hand landmarks (4 - end of thumb, 8 - end of index) and maps it to volume using the pycaw library.

This project assumes you have a webcam. If not, you can attach a peripheral camera, and change "cv.VideoCapture(0)" to "cv.VideoCapture(1)".

No configuration is required.
