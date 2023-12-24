from flask import Flask, render_template, Response, request, jsonify
import base64
import numpy as np
import cv2
import json
import virtual_trial

app = Flask(__name__)

@app.route('/tryon/<file_path>', methods=['POST', 'GET'])
def tryon(file_path):
    file_path = file_path.replace(',', '/')
    return render_template("checkout.html", file_path=file_path)

@app.route('/upload_frame', methods=['POST'])
def process_frame_upload():
    file = request.files['frame']
    json_data = request.form['json_data']
    data = json.loads(json_data)
    file_path = data['file_path']
    
    filestr = file.read()
    npimg = np.frombuffer(filestr, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    processed_frame = virtual_trial.process_frame(frame, file_path)
    if processed_frame is not None:
        ret, jpeg = cv2.imencode('.jpg', processed_frame)
        if ret:
            return jsonify({'image': base64.b64encode(jpeg.tobytes()).decode('utf-8')})
        else:
            return jsonify({'error': 'Failed to process frame'}), 500
    else:
        return jsonify({'error': 'No frame processed'}), 500
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=5001)
