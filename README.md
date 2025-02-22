# Face Morphing Attack Detection Using CNN

This project focuses on detecting face morphing attacks using Convolutional Neural Networks (CNN). Face morphing attacks involve blending two or more facial images to create a composite image that can deceive facial recognition systems. Our CNN model is trained to identify such manipulated images and differentiate them from authentic ones.  

## Features

✅ **User Authentication**: Users can **sign up and log in** using their details, which are securely stored in an **SQLite database**.  

✅ **Image Preprocessing**:  
   - The browsed image is **converted to grayscale**.  
   - It is then **transformed into binary format** for further analysis.

✅ **Morphing Attack Detection**:  
   - The CNN model classifies the image as **morphed or authentic**.

✅ **Runs on Spyder** within **Anaconda Navigator**  


## Technologies Used  
- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow/Keras  
- **Libraries**: OpenCV, NumPy, Matplotlib, Pandas, Scikit-learn, SQLite (for database)  
- **Development Environment**: Spyder (Anaconda Navigator)  
- **Database**: SQLite for user authentication and data storage  

## Installation & Setup  
1. Install Anaconda: [Download Here](https://www.anaconda.com/)  
2. Create and activate a new environment:  
   ```bash
   conda create -n Face_Morphing python=3.8  
   conda activate Face_Morphing  
   ```
3. Install required dependencies:  
   ```bash
   pip install tensorflow
   pip install keras
   pip install opencv-python
   pip install numpy
   pip install matplotlib
   pip install pandas
   pip install scikit-learn
   pip install pillow
   pip install tkvideo
   pip install mediapipe
   pip install gtts
   pip install flask  
   ```
4. Open **Spyder** from Anaconda Navigator and run the script  

## Usage  
- **User Registration & Login**: Users can sign up and log in, with credentials stored in SQLite.  
- **Image Preprocessing**:  
  - Convert image to grayscale  
  - Transform it into binary format for feature extraction  
- **Classification using CNN**:  
  - The model predicts whether the image is morphed or authentic  
  - The outcome is displayed to the user
 
## Future Enhancements  
🚀 Improve model accuracy with advanced architectures  
🔍 Integrate real-time face detection and classification  
📊 Deploy the model as a web or mobile application  
---
