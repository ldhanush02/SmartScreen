from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
from ultralytics import YOLO
import os
import cv2

app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RUN_FOLDER'] = 'runs'

app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(file_path)
        # file.save(file_path)
        base_path=os.path.dirname(__file__)
        file_path1=os.path.join(base_path,'uploads',filename)
        file.save(file_path1)
        # global imgpath
        # predict_img.imgpath=filename
        img=cv2.imread(file_path1)
        print(file_path1)
        # frame=cv2.imencode('.png',cv2.UMat(img))[1].tobytes()
        # image=Image.open(io.BytesIO(frame))
        yolo=YOLO('best.pt')
        detections=yolo.predict(file_path1,save=True)
        folderpath='runs/detect'
        subfolders=[f for f in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath,f))]
        latestsubfolder=max(subfolders,key = lambda x: os.path.getctime(os.path.join(folderpath,x)) )
        directory=folderpath+"\\"+latestsubfolder
        files=os.listdir(directory)
        latest_file=files[0]
        filename1=os.path.join(folderpath,latestsubfolder,latest_file)
        return render_template('upload_result.html', filename=filename,filepath=filename1,latestsubfolder=latestsubfolder)

    return redirect(request.url)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/runs/detect/<latestsubfolder>/<filename>')
def run_file(latestsubfolder,filename):
    folderpath='runs/detect'
    subfolders=[f for f in os.listdir(folderpath) if os.path.isdir(os.path.join(folderpath,f))]
    latestsubfolder=max(subfolders,key = lambda x: os.path.getctime(os.path.join(folderpath,x)) )
    directory=folderpath+"/"+latestsubfolder
    files=os.listdir(directory)
    latest_file=secure_filename(files[0])
    filepath=os.path.join(folderpath,latestsubfolder,latest_file)
    return send_from_directory(os.path.join(app.config['RUN_FOLDER'],'detect',latestsubfolder),filename,mimetype='image/png')

if __name__ == '__main__':
    # Ensure the 'uploads' folder exists
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    app.run(debug=True)
