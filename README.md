# Face Morphing Attack Detection Using CNN

This project focuses on detecting face morphing attacks using Convolutional Neural Networks (CNN). Face morphing attacks involve blending two or more facial images to create a composite image that can deceive facial recognition systems. Our CNN model is trained to identify such manipulated images and differentiate them from authentic ones.  

## Features  
✅ Uses a deep learning-based CNN model for high-accuracy morphing attack detection  
✅ Preprocessed datasets to improve model performance  
✅ Trained and tested on real and morphed facial images  
✅ Runs on **Spyder** within **Anaconda Navigator**  

## Technologies Used  
- **Programming Language**: Python  
- **Deep Learning Framework**: TensorFlow/Keras  
- **Libraries**: OpenCV, NumPy, Matplotlib, Pandas, Scikit-learn  
- **Development Environment**: Spyder (Anaconda Navigator)  

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
- Load the dataset containing real and morphed images  
- Train the CNN model using TensorFlow/Keras  
- Test the model on new images to detect morphing attacks  
- Evaluate accuracy and visualize results  

## Future Enhancements  
🚀 Improve model accuracy with advanced architectures  
🔍 Integrate real-time face detection and classification  
📊 Deploy the model as a web or mobile application  

---
