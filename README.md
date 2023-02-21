# Face Recognition Web App using Flask
## Face Recognition door-lock using Raspberry Pi

This repository contains the code and instructions for building a face recognition system using a Raspberry Pi and a camera module.
The project uses OpenCV, a popular computer vision library, to detect and recognize faces in real-time. The system is designed to learn and recognize faces by capturing images and storing them as reference images. The system can then compare the captured faces with reference images to identify and label individuals.
The code is written in Python and runs on a Raspberry Pi with a connected camera module. The project is compatible with both Raspberry Pi 3 and Raspberry Pi 4.
## Demo


https://user-images.githubusercontent.com/81076373/220266792-0211847d-31cb-43f6-b21f-08b8e57dcebb.mp4


## Requirements

- To run this project, you need:
- Raspberry Pi 3 or Raspberry Pi 4
- Raspberry Pi camera module
- Python 3
- OpenCV (pre-installed on Raspbian)
- Numpy (pre-installed on Raspbian)
## Installation and usage

Clone this repository on your Raspberry Pi by running the following command:
```sh
git clone https://github.com/mahnamnasir/face-recognition-resberry-pi.git
```
Navigate to the repository directory:
```sh
cd face-recognition-resberry-pi
```
Run the following command to install the required packages:
```sh
pip3 install -r requirements.txt
```
Run the face recognition system by running the following command:
```sh
python3 face_recognition.py
```
## Configuration
The face recognition system can be configured by modifying the values in the config.py file. Here are the available configurations:
- model: The face recognition model to use. The default value is hog, which is a faster but less accurate model. You can set it to cnn for a more accurate but slower model.
- tolerance: The tolerance value for face recognition. The default value is 0.6, which is a good balance between accuracy and speed. You can decrease it for higher accuracy but slower speed, or increase it for faster speed but lower accuracy.
- font: The font to use for displaying the name labels. The default value is cv2.FONT_HERSHEY_SIMPLEX.
- font_scale: The font scale to use for displaying the name labels. The default value is 1.
- font_thickness: The font thickness to use for displaying the name labels. The default value is 2.

## Contributing
Contributions are welcome! If you find any issues or have any suggestions, feel free to open an issue or a pull request.
