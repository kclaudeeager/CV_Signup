# Face Recognition System

This project is a face recognition system built using **FastAPI**, **OpenCV**, **face_recognition**, and **YOLOv8**. It allows users to register their faces and log in using either an uploaded image or live webcam detection.

---

## Features

- **Face Detection**: Detect faces in images or live video using YOLOv8.
- **Face Registration**: Register a user's face with augmented images for better recognition.
- **Face Login**: Log in using either:
  - An uploaded image.
  - Live webcam detection.
- **Confidence Threshold**: Adjustable confidence level to reduce false positives.
- **Template Matching**: Fallback mechanism for face matching using OpenCV's template matching.

---

## Requirements

### Python Libraries
Install the required Python libraries using `pip`:

```bash
pip install -r requirements.txt
```

### Additional Tools
- **Git LFS**: Required for handling large files like `face_yolov8n.pt`.

Install Git LFS:

```bash
sudo apt install git-lfs
```

---

## Setup

1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd CV_Signup
   ```

2. **Track Large Files**:
   Ensure `face_yolov8n.pt` is tracked by Git LFS:
   ```bash
   git lfs track "face_yolov8n.pt"
   ```

3. **Run the Application**:
   Start the FastAPI server:
   ```bash
   python main.py
   ```

4. **Access the Application**:
   Open your browser and navigate to:
   ```
   http://127.0.0.1:8000
   ```

---

## Endpoints

### 1. **Home Page**
- **URL**: `/`
- **Method**: `GET`
- **Description**: Displays the face recognition system's interface.

### 2. **Detect Faces**
- **URL**: `/detect`
- **Method**: `POST`
- **Description**: Detects faces in an uploaded image.
- **Request Body**: `file` (image file).

### 3. **Register User**
- **URL**: `/signup`
- **Method**: `POST`
- **Description**: Registers a user's face.
- **Request Body**:
  - `file` (image file).
  - `name` (user's name).

### 4. **Login**
- **URL**: `/login`
- **Method**: `POST`
- **Description**: Logs in a user by matching their face.
- **Request Body**: `file` (image file).

---

## File Structure

```
CV_Signup/
├── main.py                # FastAPI application
├── templates/
│   └── index.html         # Frontend HTML
├── static/
│   └── style.css          # Frontend CSS
├── face_yolov8n.pt        # YOLOv8 model file
├── face_dataset/          # Directory for storing user face images
├── face_features.db       # SQLite database for storing user data
├── .gitignore             # Git ignore file
└── README.md              # Project documentation
```

---

## Notes

- **Confidence Threshold**: The confidence threshold for face matching is set to `0.4` in the `login` endpoint. Adjust this value in `main.py` to reduce false positives.
- **Template Matching**: If face recognition fails, the system uses OpenCV's template matching as a fallback.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
