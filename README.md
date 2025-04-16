# Face Recognition Attendance System

A Python-based face recognition system that automates attendance tracking using computer vision and deep learning. The system can recognize enrolled students in real-time and mark their attendance automatically.

## Features

- **Student Enrollment**: Capture and store face samples of students for recognition
- **Real-time Face Recognition**: Identify enrolled students using a trained deep learning model
- **Automatic Attendance Tracking**: Mark attendance automatically when recognized students are detected
- **Data Augmentation**: Enhance training data with rotated and flipped face samples
- **Multi-camera Support**: Automatically detect and use available cameras
- **Attendance History**: Maintain a record of attendance with timestamps
- **User-friendly Interface**: Simple command-line interface for system interaction

## System Requirements

- Python 3.7+
- OpenCV
- TensorFlow 2.x
- NumPy
- Pandas
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/face-recognition-attendance.git
cd face-recognition-attendance
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Project Structure

```
face-recognition-attendance/
├── face_recognition_system.py  # Main system implementation
├── requirements.txt            # Python dependencies
├── face_recognition_model.h5   # Trained model file
├── attendance.csv             # Attendance records
├── enrolled_students.pkl      # Enrolled student data
└── archive (1)/               # Dataset directory
    ├── Dataset.csv           # Dataset metadata
    └── Faces/                # Face images
```

## Usage

1. **Start the System**:
```bash
python face_recognition_system.py
```

2. **Main Menu Options**:
   - Enroll New Student: Register a new student by capturing their face
   - Train Model: Train the face recognition model
   - Start Recognition: Begin real-time face recognition
   - View Enrolled Students: List all enrolled students
   - Exit: Close the system

3. **Enrolling a Student**:
   - Enter student name
   - Position face in camera
   - System will capture 5 face samples
   - Samples are automatically augmented for better recognition

4. **Training the Model**:
   - System uses both enrolled student data and a standard dataset
   - Implements data augmentation for better accuracy
   - Uses a CNN architecture optimized for face recognition
   - Provides training progress and accuracy metrics

5. **Running Recognition**:
   - Opens camera feed
   - Detects faces in real-time
   - Identifies enrolled students
   - Marks attendance automatically
   - Shows confidence scores for recognized faces

## Technical Details

### Face Recognition Model
- Uses a Convolutional Neural Network (CNN)
- Binary classification (enrolled vs. non-enrolled)
- Data augmentation techniques:
  - Horizontal flips
  - Rotation (-15° to +15°)
  - Brightness and contrast adjustments

### Camera Handling
- Supports multiple camera indices (0, 1, 2)
- Automatic camera detection and initialization
- Configurable resolution and frame rate
- Error handling and recovery mechanisms

### Attendance System
- Records date and time of attendance
- Prevents duplicate entries within 5 minutes
- Stores attendance data in CSV format
- Supports multiple attendance records per day

## Troubleshooting

1. **Camera Issues**:
   - Ensure camera is properly connected
   - Check camera permissions
   - Try different camera indices
   - Verify camera is not in use by another application

2. **Recognition Problems**:
   - Ensure good lighting conditions
   - Position face clearly in camera view
   - Retrain model if recognition accuracy is low
   - Check if student is properly enrolled

3. **Model Training Issues**:
   - Ensure sufficient face samples during enrollment
   - Check available system memory
   - Verify dataset files are accessible
   - Monitor training progress for errors

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Uses the Haarcascade classifier for face detection
- Implements transfer learning with pre-trained models
- Utilizes standard face recognition datasets for training 