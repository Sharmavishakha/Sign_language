# from flask import Flask, render_template, Response, request, jsonify
# import cv2
# import mediapipe as mp
# import pickle
# import numpy as np

# app = Flask(__name__)

# # Load the model
# model_dict = pickle.load(open(r"C:\Users\chetna\OneDrive\Desktop\signTotext\model_sign.p", 'rb'))
# model = model_dict['model']

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# # Label dictionary for predictions
# labels_dict = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
#                'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
#                'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
#                'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}

# # Variables to hold words and sentences
# current_word = ""
# sentence = ""

# @app.route('/')
# def index():
#     return render_template('index.html')

# def generate_frames():
#     global current_word
#     cap = cv2.VideoCapture(0)

#     while True:
#         success, frame = cap.read()
#         if not success:
#             break

#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(frame_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Extract hand landmarks
#                 data_aux = []
#                 for i in range(len(hand_landmarks.landmark)):
#                     x = hand_landmarks.landmark[i].x
#                     y = hand_landmarks.landmark[i].y
#                     data_aux.append(x)
#                     data_aux.append(y)

#                 # Make predictions
#                 predicted_character = ""
#                 if len(data_aux) == 42:  # One hand
#                     prediction = model.predict([np.asarray(data_aux)])
#                     predicted_character = prediction[0] if isinstance(prediction[0], str) else labels_dict[int(prediction[0])]

#                 if predicted_character:
#                     current_word += predicted_character

#         # Display current word and sentence on the frame
#         cv2.putText(frame, f'Word: {current_word}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#         cv2.putText(frame, f'Sentence: {sentence}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

#         # Encode frame for display on the web
#         ret, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

#     cap.release()

# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# @app.route('/add_word', methods=['POST'])
# def add_word():
#     global sentence, current_word
#     sentence += current_word + " "
#     current_word = ""
#     return jsonify({"sentence": sentence})

# @app.route('/clear_sentence', methods=['POST'])
# def clear_sentence():
#     global sentence, current_word
#     sentence = ""
#     current_word = ""
#     return jsonify({"sentence": sentence})

# if __name__ == '__main__':
#     app.run(debug=True)


from flask import Flask, render_template, Response, request, jsonify
import cv2
import mediapipe as mp
import pickle
import numpy as np

app = Flask(__name__)

# Load the model
model_dict = pickle.load(open(r"C:\Users\chetna\OneDrive\Desktop\signTotext\model_sign.p", 'rb'))
model = model_dict['model']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.5)

# Label dictionary for predictions
labels_dict = {'A': 'A', 'B': 'B', 'C': 'C', 'D': 'D', 'E': 'E', 'F': 'F', 'G': 'G',
               'H': 'H', 'I': 'I', 'J': 'J', 'K': 'K', 'L': 'L', 'M': 'M', 'N': 'N',
               'O': 'O', 'P': 'P', 'Q': 'Q', 'R': 'R', 'S': 'S', 'T': 'T', 'U': 'U',
               'V': 'V', 'W': 'W', 'X': 'X', 'Y': 'Y', 'Z': 'Z'}

# Variables to hold words and sentences
current_word = ""
sentence = ""
frame_counter = 0  # Frame counter for detection delay
DETECTION_DELAY = 20  # Number of frames to wait between detections

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    global current_word, frame_counter
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        # Increment frame counter
        frame_counter += 1

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Only detect after delay
                if frame_counter >= DETECTION_DELAY:
                    # Reset frame counter
                    frame_counter = 0

                    # Extract hand landmarks
                    data_aux = []
                    for i in range(len(hand_landmarks.landmark)):
                        x = hand_landmarks.landmark[i].x
                        y = hand_landmarks.landmark[i].y
                        data_aux.append(x)
                        data_aux.append(y)

                    # Make predictions
                    predicted_character = ""
                    if len(data_aux) == 42:  # One hand
                        prediction = model.predict([np.asarray(data_aux)])
                        predicted_character = prediction[0] if isinstance(prediction[0], str) else labels_dict[int(prediction[0])]

                    if predicted_character:
                        current_word += predicted_character

        # Display current word and sentence on the frame
        cv2.putText(frame, f'Word: {current_word}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(frame, f'Sentence: {sentence}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Encode frame for display on the web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_word', methods=['POST'])
def add_word():
    global sentence, current_word
    sentence += current_word + " "
    current_word = ""
    return jsonify({"sentence": sentence})

@app.route('/clear_sentence', methods=['POST'])
def clear_sentence():
    global sentence, current_word
    sentence = ""
    current_word = ""
    return jsonify({"sentence": sentence})

if __name__ == '__main__':
    app.run(debug=True)
