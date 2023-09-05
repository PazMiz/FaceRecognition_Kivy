import cv2
import numpy as np
import sqlite3
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
import face_recognition

class FaceRecognitionApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Error: Camera not found or cannot be opened.")
            return

        self.img = Image()
        self.name_label = Label(text="Name: Unknown", font_size=20)
        layout = BoxLayout(orientation="vertical")
        layout.add_widget(self.img)
        layout.add_widget(self.name_label)
        self.capture_button = Button(text="Capture Photo")
        self.capture_button.bind(on_press=self.capture_photo)
        layout.add_widget(self.capture_button)
        self.register_with_photo_button = Button(text="Register with Photo")
        self.register_with_photo_button.bind(on_press=self.register_with_photo)
        layout.add_widget(self.register_with_photo_button)
        self.login_button = Button(text="Login with Face")
        self.login_button.bind(on_press=self.login_with_face)
        layout.add_widget(self.login_button)
        self.db_connection = sqlite3.connect("face_recognition.db")
        self.create_table_if_not_exists()
        self.registered_name = None
        self.registered_face_encoding = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.captured_photo = None
        return layout

    def create_table_if_not_exists(self):
        cursor = self.db_connection.cursor()
        cursor.execute('''CREATE TABLE IF NOT EXISTS faces
                          (name TEXT, encoding BLOB, photo BLOB)''')
        self.db_connection.commit()

    def capture_photo(self, instance):
        ret, frame = self.capture.read()
        if ret:
            self.captured_photo = frame
            self.display_image(frame)
            print("Photo captured")

    def display_image(self, frame):
        buf = cv2.flip(frame, 0).tobytes()
        texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.img.texture = texture

    def register_with_photo(self, instance):
        if self.captured_photo is not None:
            detected_face = self.detect_face(self.captured_photo)
            if detected_face is not None:
                name = "paz"
                self.name_label.text = f"Name: {name}"
                self.save_face_to_db(name, detected_face)
                print(f"Face registered as '{name}'")
                self.registered_name = name
                self.registered_face_encoding = self.encode_face(detected_face)
            else:
                self.name_label.text = "Name: Unknown"
                print("Face not recognized. Please try again.")
        else:
            self.name_label.text = "Name: Unknown"
            print("Capture a photo first.")

    def login_with_face(self, instance):
        if self.captured_photo is not None:
            detected_face = self.detect_face(self.captured_photo)
            if detected_face is not None and self.registered_name is not None and self.registered_face_encoding is not None:
                detected_face_rgb = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)  # Convert to RGB
                if self.match_face(detected_face_rgb, self.registered_face_encoding):
                    self.name_label.text = f"Welcome, {self.registered_name}!"
                    print(f"Welcome, {self.registered_name}!")
                    self.display_image(detected_face_rgb)  # Display the recognized face
                else:
                    self.name_label.text = "Name: Unknown"
                    print("Face not recognized. Please try again.")
            else:
                self.name_label.text = "Name: Unknown"
                print("No registered face found.")
        else:
            self.name_label.text = "Name: Unknown"
            print("Capture a photo first.")

    def detect_face(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            x, y, w, h = faces[0]
            detected_face = gray[y:y + h, x:x + w]
            return detected_face
        return None

    def encode_face(self, detected_face):
        # Convert the detected face to RGB format
        detected_face_rgb = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
        
        # Encode the detected face using face_recognition library
        detected_face_encoding = face_recognition.face_encodings(detected_face_rgb)
        
        if len(detected_face_encoding) > 0:
            # Convert the encoding to a string
            encoding_str = detected_face_encoding[0].tostring()
            return encoding_str
        
        return None

    def match_face(self, detected_face, registered_face_encoding):
        # Convert the detected face to an encoding using face_recognition library
        detected_face_encoding = face_recognition.face_encodings(detected_face)
        
        if len(detected_face_encoding) > 0:
            # Calculate the distance using numpy's linalg.norm function
            distance = np.linalg.norm(np.frombuffer(registered_face_encoding, dtype=np.float64) - detected_face_encoding[0])
            
            # Define a threshold for considering it a match (you can adjust this threshold as needed)
            threshold = 0.6  # Adjust this threshold as needed
            
            # Check if the distance is below the threshold
            if distance < threshold:
                return True
        
        return False

    def save_face_to_db(self, name, detected_face):
        encoding_str = self.encode_face(detected_face)
        if encoding_str is not None:
            cursor = self.db_connection.cursor()
            cursor.execute("INSERT INTO faces VALUES (?, ?, ?)", (name, sqlite3.Binary(encoding_str), sqlite3.Binary(detected_face.tobytes())))
            self.db_connection.commit()

    def on_stop(self):
        self.capture.release()
        self.db_connection.close()

if __name__ == "__main__":
    FaceRecognitionApp().run()
