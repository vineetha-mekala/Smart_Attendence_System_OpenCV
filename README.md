## Smart Attendance System (MTCNN + FaceNet)

This project is an AI-based Smart Attendance System that uses Face Recognition to automatically mark student attendance from classroom images or webcam input.

It detects faces, recognizes students, and stores attendance records in a CSV file.

## Features
Upload classroom images (1–3 images supported)
Capture photo using webcam
Face detection using MTCNN
Face recognition using FaceNet (InceptionResnetV1)
Marks attendance automatically
Avoids duplicate entries for the same day
Saves processed images with annotations
Generates attendance CSV file

## Tech Stack
Python
OpenCV
PyTorch
facenet-pytorch
NumPy
Pandas
Streamlit

## Project Structure
.
├── app.py                  # Streamlit UI
├── system_process.py       # Face detection & recognition
├── attendance_manager.py  # Attendance CSV handling
├── encodings_generator.py # Generate face encodings
├── load_encodings.py      # Load saved encodings
├── students/              # Student images (dataset)
├── processed_images/      # Output images
├── encodings.pkl          # Saved embeddings
├── attendance.csv         # Attendance records

## Setup Instructions
1. Clone the repository
git clone <your-repo-link>
cd <repo-folder>
2. Install dependencies
pip install -r requirements.txt

(If no requirements file, install manually)

pip install streamlit opencv-python numpy pandas torch facenet-pytorch

## Prepare Dataset
Create a folder named students/
Inside it, create subfolders for each student:
students/
   ├── John/
   │     ├── img1.jpg
   │     ├── img2.jpg
   ├── Alice/
         ├── img1.jpg

## Generate Face Encodings

Run this once before starting the app:

python encodings_generator.py

This creates:

encodings.pkl

## Run the Application
streamlit run app.py

## How It Works
User uploads images or captures via webcam
Faces are detected using MTCNN
Face embeddings are generated using FaceNet
Compared with stored encodings using cosine similarity
Recognized names are recorded in attendance.csv
Processed images are saved with bounding boxes

## Output
Processed Images → processed_images/
Attendance File → attendance.csv

## Future Improvements
Real-time video attendance
Database integration (instead of CSV)
UI improvements
Multi-classroom support
Cloud deployment
