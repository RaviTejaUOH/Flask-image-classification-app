from flask import Flask, render_template
from werkzeug.utils import secure_filename
from flask import request
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input, decode_predictions
#import cv2
from keras.models import model_from_json
from keras.preprocessing.image import image
import numpy as np
import os

app = Flask(__name__)

json_file = open('model_json.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights('my_model_weights.h5')
loaded_model._make_predict_function()




def predict(filepath):
	#read image
	#img = cv2.imread(filepath)
	#resize image
	#img_resize = cv2.resize(img, (224,224))

	img = image.load_img(filepath, target_size=(224, 224))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	preds = loaded_model.predict(x)
	preds = decode_predictions(preds, top=3)[0]
	os.remove(filepath)
	return preds


@app.route('/')
def index():
	return render_template("index.html")


@app.route('/', methods=['POST'])
def upload_file():
	f = request.files['file']
	basepath = os.path.dirname(__file__)
	print(basepath)
	file_path = os.path.join(basepath, secure_filename(f.filename))
	f.save(file_path)
	preds = predict(file_path)
	return preds[0][1]



if __name__ == "__main__":
	app.run(debug=True,port=os.environ.get('PORT'))
