from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import cv2
import numpy as np
from ultralytics import YOLO
import sqlite3
import os
import face_recognition
# import imgaug.augmenters as iaa
import albumentations as A

from operator import itemgetter
import logging

logging.basicConfig(level=logging.INFO)

app = FastAPI()
app.mount("/static", StaticFiles(directory="templates"), name="static")
templates = Jinja2Templates(directory="templates")

# Face detection model
model = YOLO('face_yolov8n.pt')  # Specific face detection model

# Database and storage setup
DB_PATH = "face_features.db"
FACE_STORAGE_DIR = "face_dataset"
os.makedirs(FACE_STORAGE_DIR, exist_ok=True)

# Create database if not exists
if not os.path.exists(DB_PATH):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY, 
            name TEXT, 
            face_encodings BLOB,
            face_images_dir TEXT
        )
    """)
    conn.commit()
    conn.close()

# Image augmentation pipeline
# augmentation_pipeline = iaa.Sequential([
#     iaa.Fliplr(0.5),  # 50% horizontal flips
#     iaa.Affine(
#         rotate=(-10, 10),  # Rotation between -10 and 10 degrees
#         scale=(0.8, 1.2),  # Scale between 80% and 120%
#     ),
#     iaa.GaussianBlur(sigma=(0, 1.0)),  # Slight gaussian blur
#     iaa.AdditiveGaussianNoise(scale=(0, 0.05*255))  # Add some noise
# ])
def create_augmentation_pipeline():
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Affine(
            scale={'x': (0.8, 1.2), 'y': (0.8, 1.2)},
            translate_percent={'x': (-0.1, 0.1), 'y': (-0.1, 0.1)},
            rotate=[-10, 10],
            p=0.5
        ),
        A.GaussNoise(p=0.2),
        A.OneOf([
            A.MotionBlur(p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        A.ShiftScaleRotate(
            shift_limit=0.1, 
            scale_limit=0.1, 
            rotate_limit=15, 
            p=0.5
        ),
    ])

# Augmentation function
def augment_image(image, pipeline):
    """
    Augment an image using the given augmentation pipeline
    
    Args:
        image (numpy.ndarray): Input image
        pipeline (A.Compose): Augmentation pipeline
    
    Returns:
        list: List of augmented images
    """
    augmented_images = []
    
    # Original image
    augmented_images.append(image)
    
    # Generate 4 additional augmented images
    for _ in range(4):
        augmented = pipeline(image=image)['image']
        augmented_images.append(augmented)
    
    return augmented_images

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/save-frame", response_class=HTMLResponse)
async def save_frame(request: Request, file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Save the frame
    cv2.imwrite("static/frame.jpg", frame)
    
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Detect faces using YOLO
    results = model(frame)
    faces = []
    for result in results:
        for box in result.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            faces.append([x1, y1, x2, y2])
    
    return {"faces": faces}

@app.post("/signup")
async def signup(file: UploadFile = File(...), name: str = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    if not face_encodings:
        raise HTTPException(status_code=400, detail="No face detected")
    
    # Create user-specific directory
    user_face_dir = os.path.join(FACE_STORAGE_DIR, name)
    os.makedirs(user_face_dir, exist_ok=True)
    
    # Generate multiple augmented images
    augmented_encodings = []
    for i in range(5):  # Generate 5 augmented images
        # Original and augmented images
        images_to_process = [frame]
        
        # Apply augmentations
        aug_pipeline = create_augmentation_pipeline()
        augmented_images = augment_image(frame, aug_pipeline)
        images_to_process.extend(augmented_images)  # Use extend instead of append
        
        for j, img in enumerate(images_to_process):
            # Convert to RGB
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Compute face encoding
            face_enc = face_recognition.face_encodings(rgb_img, face_locations)[0]
            augmented_encodings.append(face_enc)
            
            # Save image
            cv2.imwrite(os.path.join(user_face_dir, f'face_{i}_{j}.jpg'), img)
    
    # Convert encodings to bytes for storage
    combined_encodings = np.array(augmented_encodings)
    
    # Store in database
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        INSERT INTO users (name, face_encodings, face_images_dir) 
        VALUES (?, ?, ?)
    """, (name, combined_encodings.tobytes(), user_face_dir))
    conn.commit()
    conn.close()
    
    return {"status": "User registered successfully"}


@app.post("/login")
async def login(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    # Convert to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    if not face_encodings:
        raise HTTPException(status_code=400, detail="No face detected")
    
    # Get all stored users
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, face_encodings FROM users")
    users = c.fetchall()
    conn.close()
    
    # Compare face with stored faces
    best_match = None
    best_distance = float("inf")
    for name, stored_encodings_bytes in users:
        # Convert stored bytes back to numpy array
        stored_encodings = np.frombuffer(stored_encodings_bytes, dtype=np.float64).reshape(-1, 128)
        
        # Calculate distances
        distances = face_recognition.face_distance(stored_encodings, face_encodings[0])
        min_distance = min(distances)
        
        # Check if this is the best match
        if min_distance < best_distance and min_distance < 0.4:  # Adjust threshold as needed
            best_distance = min_distance
            best_match = name
        logging.info(f"User: {name}, Distance: {min_distance}")
    
    logging.info(f"Best match: {best_match}, Distance: {best_distance}")
    if best_match:
        return {"status": "Login successful", "user": best_match, "confidence": 1 - best_distance}
    
    raise HTTPException(status_code=401, detail="No matching face found")
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)