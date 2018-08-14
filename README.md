# IRB120-ReadWriter
## Introduction

`irb120-reader` is part of a robotics demonstration project which will have an image processing and artificial intelligence module for handwriting recognition, then an [irb120](https://new.abb.com/products/robotics/industrial-robots/irb-120) robotic arm will write it on another surface, such as a whiteboard, glass or paper.

### Description

This repository contains the artificial intelligence module of this project. The main idea is to use deep neural networks like the robot's brain to recognize the characters and give the robot the "ability to read". To do this, the DNN will be deployed on an [NVIDIA Jetson TX2](https://www.nvidia.com/en-us/autonomous-machines/embedded-systems-dev-kits-modules/) device that sends digital signals to the [IRC5](https://new.abb.com/products/robotics/controllers/irc5) robot driver.
