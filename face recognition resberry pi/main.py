###################Libraries###############################################################
# from datetime import datetime
import datetime
from flask import Flask , render_template, redirect, url_for ,request, Response 
from flask_bootstrap import Bootstrap
from dominate.tags import img
from train_model import tarin_model
#import face-recognition
import cv2
import os
from flask import Flask,send_from_directory ,url_for ,request
from flask_uploads import UploadSet ,IMAGES , configure_uploads
from flask_wtf import FlaskForm 
from flask_wtf.file import FileField, FileRequired,FileAllowed
from wtforms import SubmitField
import shutil
from imutils.video import VideoStream
from imutils.video import FPS
#import RPi.GPIO as GPIO
import imutils
import requests
import pickle
import time
################################################################################################
# Define flask app
app = Flask(__name__)

app.config['SECRET_KEY'] = 'hello'
app.config["UPLOADED_PHOTOS_DEST"] = "uploads"

photos = UploadSet("photos",IMAGES)
configure_uploads(app,photos)

class UploadForm(FlaskForm):
    photo = FileField(
        validators=[
            FileAllowed(photos,'Only images are allowed'),
            FileRequired('File Feild should not be empty')
        ]
    )
    submit = SubmitField('Upload')
@app.route("/uploads/<filename>")
def get_file(filename):
    return send_from_directory(app.config["UPLOADED_PHOTOS_DEST"],filename)

############################################################################################
################# Home page ################################################################
############################################################################################

# upload image
@app.route("/home",methods=['POST','GET'])
def upload_image():
    form_data = request.form
    name =form_data.get("username")
    global names
    names = str(name)
    dir = "dataset/"+ names
    os.mkdir(dir)
    form = UploadForm()
    if form.validate_on_submit():
        global filename
        filename = photos.save(form.photo.data)
        img_name = "dataset/"+ name +"/image_{}.jpg".format(0)
        src = 'uploads'
        src = f"{src}/{os.listdir(src)[0]}"
        shutil.move(src,dir)
        file_url = url_for('get_file',filename=filename)
    else:
        file_url =None
    return render_template("dataset.html" , form=form,file_url=file_url,form_data=form_data)


############################################################################################3
################ train model ###############################################################3
############################################################################################3 
@app.route("/train",methods=["POST",'GET'])
def train():
    return render_template("training.html")
@app.route("/model")
def train_medel():
    return tarin_model()

 
###########################################################################################
######################### Login form ######################################################
###########################################################################################
@app.route('/login',methods=['GET','POST'])
def login():
    error = None
    if request.method == 'POST':
        if request.form['username'] != 'admin' or request.form['password'] != 'admin':
            error = 'Invalid Credentials. Please try again'
        else:
            return redirect(url_for('upload_image'))
    return render_template("login.html",error=error)


################################################################################################
########################### Face Recognition ####################################################
#################################################################################################3
currentname = 'unknown'
# Function for gen_Frames
#Determine faces from encodings.pickle file model created from train_model.py
encodingsP = "encodings.pickle"
#use this xml file
cascade = "haarcascade_frontalface_default.xml"
# load the known faces and embeddings along with OpenCV's Haar
# cascade for face detection
print("[INFO] loading encodings + facevis detector...")
data = pickle.loads(open(encodingsP, "rb").read())
detector = cv2.CascadeClassifier(cascade)
def gen():
    currentname = "unknown"
    vs = VideoStream(src=0).start()
    #vs = VideoStream(usePiCamera=True).start()
    time.sleep(2.0)
    # start the FPS counter
    fps = FPS().start()

    while True:
        
        
        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()
        frame = imutils.resize(frame, width=500)
        # convert the input frame from (1) BGR to grayscale (for face
        # detection) and (2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
            minNeighbors=5, minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
            # OpenCV returns bounding box coordinates in (x, y, w, h) order
        # but we need them in (top, right, bottom, left) order, so we
        # need to do a bit of reordering
        boxes = [(y, x + w, y + h, x) for (x, y, w, h) in rects]

        # compute the facial embeddings for each face bounding box
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

            # loop over the facial embeddings
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings
            matches = face_recognition.compare_faces(data["encodings"],
                encoding)
            name = "Unknown"

            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
    # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                name = max(counts, key=counts.get)
                
                #If someone in your dataset is identified, print their name on the screen
            if currentname != name:
                #Take a picture to send in the email
                img_name = "image.jpg"
                cv2.imwrite(img_name, frame)
                now = datetime.datetime.now()
                nows =str(now)
                print ("Current date and time : ")
                print (now.strftime("%Y-%m-%d %H:%M:%S"))
                print('Taking a picture.')
                        
					#Now send me an email to let me know who is at the door
                request = send_message(nows)
                print ('Status Code: '+format(request.status_code)) #200 status code means email sent successfully
                currentname = name
                print(currentname)
                
            # update the list of names
            names.append(name)    
            
            
        # loop over the recognized faces
        allowed = "door accessed"
        dallowed = "door not accessed"
        dir = "dataset"
        allowed_people = os.listdir(dir)
        coordinates = (50,50)
        fontScale = 1
        color = (255,0,255)
        thickness = 2
                    
        for ((top, right, bottom, left), name) in zip(boxes, names):
            # draw the predicted face name on the image - color is in BGR
            cv2.rectangle(frame, (left, top), (right, bottom),
                (0, 255, 225), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                .8, (0, 255, 255), 2)
            if currentname in allowed_people:
                cv2.putText(frame,allowed,coordinates,cv2.FONT_HERSHEY_SIMPLEX,fontScale,(0,255,0),thickness)
            else:
                cv2.putText(frame,dallowed,coordinates,cv2.FONT_HERSHEY_SIMPLEX,fontScale,(0,0,255),thickness)
                
                   
         #update the FPS counter
            fps.update()

        # stop the timer and display FPS information
        fps.stop()
        print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))



        
        cv2.imwrite('t.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + open('t.jpg', 'rb').read() + b'\r\n')

# door lock message
@app.route('/')
def message():
    return render_template('facedetect.html',currentname=currentname)

# Face detection
@app.route('/facedetect',methods =['GET'])
def facedetect():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


#####################################################################################
##################### Timer Clock Setup##############################################
#####################################################################################
@app.route('/time',methods=['POST','GET'])
def timer():
    form_data = request.form
    global timer
    timer =form_data.get("timer")
    print(timer)
    return render_template('display.html',form_data=form_data)
    
@app.route('/alarm',methods=['POST','GET'])
def alarm():
    str(timer)
    times = datetime.datetime.strptime(timer, '%H:%M')
    print (times.hour, times.minute)
    alarmH = times.hour
    alarmM = times.minute
    while(1 == 1):
        if(alarmH == datetime.datetime.now().hour and
            alarmM == datetime.datetime.now().minute) :
            print("Open Door")
            break
    return render_template('display.html')
#####################################################################################
############################ Email Sending###########################################
#####################################################################################
#function for setting up emails
def send_message(now):
    return requests.post(
        "https://api.mailgun.net/v3/sandboxf887c61707d640a8bad771cee322cb0a.mailgun.org/messages",
        auth=("api", "df0e4dfbd3a0a4a88ee184d339545e27-78651cec-3fa2b72f"),
        files = [("attachment", ("image.jpg", open("image.jpg", "rb").read()))],
        data={"from": 'mailgun@sandboxf887c61707d640a8bad771cee322cb0a.mailgun.org',
            "to": ["m7jalmousa@gmail.com"],
            "subject": "You have a visitor",
            "html": "<html>" + now + " Date and Time for the UNKNOWN entry. </html>"})

#App Run
if __name__ == '__main__':
   app.run(debug=True,use_reloader=True)

