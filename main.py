
from flask import Flask, render_template, Response
import argparse
import cv2
from ultralytics import YOLOv10
from threading import Thread
import speech_recognition as sr
import numpy as np
from g4f.client import Client
import time
from g4f.client import Client
from g4f.Provider.GeminiPro import GeminiPro
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import pyttsx3
import asyncio
import sys

if sys.platform:
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

app = Flask(__name__)
temp = "This is the first iteration of the response so ignore my instructions on repetition."



def dir_scene(img_bytes):
    imgClient = Client(
        api_key="AIzaSyD7gGcC_grUkZx5Ww_N1h_RkeHlj95U6RM",
        provider=GeminiPro
    )
    response = imgClient.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "I am a blind person that needs to know details of this image with defining features such as specific objects, their color, and distance from my position. You are a helpful agent that will give a vivid and detailed description of the situation. If there are any people and their faces visible try to guess their emotion based on their face. Please provide an educated guess on what actions are happening in this scene as well as a guess on what may happen next."}],
        image=img_bytes
    )

    return response.choices[0].message.content


def parse_arguments() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="YOLOv8 live")
  parser.add_argument(
      "--webcam-resolution",
      default=[1280, 720],
      nargs=2,
      type=int
  )
  parser.add_argument(
      "--horizontal-fov",
      default=70.0,
      type=float,
      help="Horizontal field of view of the webcam in degrees"
  )
  args = parser.parse_args()
  return args




def get_object_color(frame, bbox):
  x1, y1, x2, y2 = bbox
  object_region = frame[int(y1):int(y2), int(x1):int(x2)]
  mean_color = cv2.mean(object_region)[:3]
  return mean_color




def color_to_description(color):
  color = np.array(color)
  if np.all(color < [50, 50, 50]):
      return "very dark"
  elif np.all(color < [100, 100, 100]):
      return "dark"
  elif np.all(color < [150, 150, 150]):
      return "medium"
  elif np.all(color < [200, 200, 200]):
      return "light"
  else:
      return "very light"




def calculate_angle(position, fov, frame_size):
  center = frame_size / 2
  relative_position = position - center
  angle = (relative_position / center) * (fov / 2)
  return angle




def describe_position(center_x, center_y, frame_width, frame_height):
  horizontal_pos = "center"
  vertical_pos = "center"
  if center_x < frame_width / 3:
    horizontal_pos = "left"
  elif center_x > 2 * frame_width / 3:
    horizontal_pos = "right"
  if center_y < frame_height / 3:
    vertical_pos = "top"
  elif center_y > 2 * frame_height / 3:
    vertical_pos = "bottom"
  return f"{vertical_pos} {horizontal_pos}"




def size_description(width, height, frame_width, frame_height):
  object_area = width * height
  frame_area = frame_width * frame_height
  size_ratio = object_area / frame_area
  if size_ratio < 0.05:
      return "small"
  elif size_ratio < 0.2:
      return "medium"
  else:
      return "large"




def extract_data(frame, results, model, h_fov, frame_width, frame_height):
    object_descriptions = []
    class_counts = {}

    for result in results:
        if result.boxes.xyxy.shape[0] == 0:
            continue

        for i in range(result.boxes.xyxy.shape[0]):
            bbox = result.boxes.xyxy[i].cpu().numpy()
            confidence = result.boxes.conf[i].cpu().numpy()
            class_id = result.boxes.cls[i].cpu().numpy()
            class_name = model.names[int(class_id)]

            mean_color = get_object_color(frame, bbox)
            color_description = color_to_description(mean_color)
            object_width = bbox[2] - bbox[0]
            object_height = bbox[3] - bbox[1]
            size_desc = size_description(object_width, object_height, frame_width, frame_height)
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            h_angle = calculate_angle(center_x, h_fov, frame_width)
            v_angle = calculate_angle(center_y, h_fov * (frame_height / frame_width), frame_height)

            direction = describe_position(center_x, center_y, frame_width, frame_height)
            description = (f"I see a {size_desc} {class_name} at the {direction}. "
                           f"The color of the object is {color_description}. It is positioned at an angle of {h_angle:.2f} degrees horizontally and "
                           f"{v_angle:.2f} degrees vertically.")
            object_descriptions.append(description)

            if class_name in class_counts:
                class_counts[class_name] += 1
            else:
                class_counts[class_name] = 1

    scene_summary = "Here's what I see: " + ", ".join([f"{count} {name}(s)" for name, count in class_counts.items()])
    return object_descriptions, scene_summary

def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = ""

        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print("Exception: " + str(e))

    return said.lower()

def find_obj(obj, results, model, cap):
    for result in results:
        for i in range(result.boxes.xyxy.shape[0]):
            class_id = result.boxes.cls[i].cpu().numpy()
            class_name = model.names[int(class_id)]
            if class_name.lower() == obj.lower():
                bbox = result.boxes.xyxy[i].cpu().numpy()
                center_x = (bbox[0] + bbox[2]) / 2
                center_y = (bbox[1] + bbox[3]) / 2
                position_description = describe_position(center_x, center_y, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
                speak(position_description)
    return None

def generate_scene_description(data_log, dir_log):
  global temp
  client = Client()

  scene_description_prompt = (f"Here is a data log of all the objects detected in the past 30 seconds:+ "+"\n".join(data_log) + "\n"
                              f"Here is a scene log of broder details including actions and predections about the scene as well as descriprions of objects in it:+ " + "\n".join(dir_log) + "\n"
                              f" You are a helpful assistant that will take the data log and the scene log and output a breif but descriptive response. Weight the content of the data log 40% and the scene log 60% in your response. I am a blind person that needs to know the basics of the enviroment and what the current scene entails. Please describe the scene in a natural, breif, but all encapsulating manner." + "\n"
                              f"Also, to be less repetetive dont repeat the same information from last 30 seconds. The overall response can be similar but avoid repeating the same information. The response you put from the last 30 seconds is here: \n"+temp)
  response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": scene_description_prompt}]
  )
  temp = response.choices[0].message.content
  return temp
def speak(text):
    engine = pyttsx3.init()
    engine.say(text) 
    engine.runAndWait()

def gptDirectory(text, results, model, cap, img_bytes):
   client = Client()

   scene_description_prompt = ("the input is: " + text + "\n" + "You are a directory assistant that will follow the following instructions word for word."+"\n"+
                               "If the input is asking to find an object for example(asking where is ___ or help me find ___), ONLY output verbatim exactly as follows: \"find\" + the object that is trying to be found."+"\n"+
                               "If the input is a question (who, what, when, where, why) about the scene for examplem asking about an object or the scene. ONLY output verbatim exactly as follows: \"question\" + the question asked." +"\n"+
                               "If the input states there is an emergency and the input is calling out for help, ONLY output verbatim exactly as follows: \"help\"." +"\n"+
                               "If the input does not match any of the above, be a helpful assistant and try to assist their inquiry."+"\n")
   response = client.chat.completions.create(
      model="gpt-4o",
      messages=[{"role": "user", "content": scene_description_prompt}]
   )
   output = response.choices[0].message.content
   s = output.split(" ")
   for i in s:
        if (i == "find"):
            s.remove("find")
            find_obj(' '.join(s), results, model, cap)
        elif (i == "question"):
            s.remove("question")
            question(' '.join(s), img_bytes)
        elif (i == "help"):
            emergency_contact()
            speak("Email sent, help should be arriving soon")
def question( quest,img_bytes):
    imgClient = Client(
        api_key="AIzaSyD7gGcC_grUkZx5Ww_N1h_RkeHlj95U6RM",
        provider=GeminiPro
    )
    response = imgClient.chat.completions.create(
        model="gemini-1.5-flash",
        messages=[{"role": "user", "content": "I am a blind person that has a question about this image as follows:" + quest}],
        image=img_bytes 
    )

    speak(response.choices[0].message.content)

def emergency_contact():
    message = "I need help! Please come to me quick! my location is: " # we can modify this as however we want
    speak("Who should the emergency email be sent to?")
    email = get_audio()
    s = smtplib.SMTP('smtp.gmail.com', 587)
    s.starttls()
    s.login("sender_email_id", "sender_email_id_password") #We have to put our credentials in here (kinda risky)
    s.sendmail("sender_email_id", email, message) #We have to put our credentials in here (kinda risky) other side note: we can add a location feature just have to add to message.
    s.quit()
def main():
    speak("Hello I am VisuAI. I ill be your new eyes. If you have any questions, or want me to find an object, or have an emergency, just say hey vision.")
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution
    h_fov = args.horizontal_fov

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    model = YOLOv10.from_pretrained('jameslahm/yolov10x')

    last_data_log_time = time.time()
    last_dir_log_time = time.time()
    last_update_time = time.time()
    data_log = ""
    dir_log = ""
    wake= "hey vision"
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, agnostic_nms=True)

        if results:
            object_descriptions, scene_summary = extract_data(frame, results, model, h_fov, frame_width, frame_height)
            detected_objects = "Here is the scene summary:" + scene_summary + \
                               "Here is a more detailed description of the objects mentioned:".join(object_descriptions)

            current_time = time.time()

            if current_time - last_data_log_time >= 1:
                data_log += f"{time.strftime('%H:%M:%S', time.localtime())}: {detected_objects}\n"
                last_data_log_time = current_time

            if current_time - last_dir_log_time >= 10:
                success, img_encoded = cv2.imencode('.jpg', frame)
                if success:
                    img_bytes = img_encoded.tobytes()

                    dir_description = dir_scene(img_bytes)
                    dir_log += f"{time.strftime('%H:%M:%S', time.localtime())}: {dir_description}\n"
                    last_dir_log_time = current_time

            if current_time - last_update_time >= 75:
                scene_description = generate_scene_description(data_log, dir_log)
                speak(scene_description)
                print(scene_description)
                last_update_time = current_time
                data_log = ""
                dir_log = ""

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        speak("speak now")
        text = get_audio()
        

        if text.count(wake) > 0: # we can try to make this async
           speak("I am ready")
           text = get_audio()
           gptDirectory(text, results, model, cap, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()




@app.route('/video_feed')
def video_feed():
    render_template('camera.html')
    return Response(main(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/camera')
def camera():
    return render_template('camera.html')

if __name__ == '__main__':
    app.run(debug=True)