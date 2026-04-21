## Smart Attendance System (MTCNN + FaceNet)

This project is an AI-based Smart Attendance System that uses Face Recognition to automatically mark student attendance from classroom images or webcam input.

It detects faces, recognizes students, and stores attendance records in a CSV file.

## Features
1.Upload classroom images (1–3 images supported)
2.Capture photo using webcam
3.Face detection using MTCNN
4.Face recognition using FaceNet (InceptionResnetV1)
5.Marks attendance automatically
6.Avoids duplicate entries for the same day
7.Saves processed images with annotations
8.Generates attendance CSV file

## Tech Stack
1.Python
2.OpenCV
3.PyTorch
4.facenet-pytorch
5.NumPy
6.Pandas
7.Streamlit

## Project Structure
```bash
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
```

## Setup Instructions
1. Clone the repository
```bash
git clone <your-repo-link>
cd <repo-folder>
```
2. Install dependencies
```bash
pip install -r requirements.txt
```
(If no requirements file, install manually)
```bash
pip install streamlit opencv-python numpy pandas torch facenet-pytorch
```

## Prepare Dataset
Create a folder named students/
Inside it, create subfolders for each student:
```bash
students/
   ├── John/
   │     ├── img1.jpg
   │     ├── img2.jpg
   ├── Alice/
         ├── img1.jpg
```

## Generate Face Encodings 

Run this once before starting the app:
```bash
python encodings_generator.py
```
This creates:
```bash
encodings.pkl
```
## Run the Application
```bash
streamlit run app.py
```
## How It Works
1.User uploads images or captures via webcam
2.Faces are detected using MTCNN
3.Face embeddings are generated using FaceNet
4.Compared with stored encodings using cosine similarity
5.Recognized names are recorded in attendance.csv
6.Processed images are saved with bounding boxes

## Output
1.Processed Images → processed_images/
2.Attendance File → attendance.csv

## Future Improvements
1.Real-time video attendance
2.Database integration (instead of CSV)
3.UI improvements
4.Multi-classroom support
5.Cloud deployment
