# importing modules
import os
import json
from flask import Flask, request
from flask import Flask, redirect, url_for, request, render_template, jsonify
from flask_restplus import Api, Resource
from flasgger import Swagger
from flasgger.utils import swag_from

from werkzeug.utils import secure_filename
from flasgger import LazyString, LazyJSONEncoder


import torch
import numpy as np 
import argparse
import pickle 
import os
from torchvision import transforms 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from PIL import Image


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
encoder_path = 'models/encoder-5-3000.pkl'
decoder_path = 'models/decoder-5-3000.pkl'
vocab_path = 'data/vocab.pkl'

embed_size = 256
hidden_size = 512
num_layers = 1

def load_image(image_path, transform=None):
    image = Image.open(image_path)
    image = image.resize([224, 224], Image.LANCZOS)
    
    if transform is not None:
        image = transform(image).unsqueeze(0)
    
    return image

def infer(file_path):
    # Image preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.485, 0.456, 0.406), 
                             (0.229, 0.224, 0.225))])
    
    # Load vocabulary wrapper
    with open(vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    # Build models
    encoder = EncoderCNN(embed_size).eval()  # eval mode (batchnorm uses moving mean/variance)
    decoder = DecoderRNN(embed_size, hidden_size, len(vocab), num_layers)
    encoder = encoder.to(device)
    decoder = decoder.to(device)

    # Load the trained model parameters
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu'))
    decoder.load_state_dict(torch.load(decoder_path, map_location='cpu'))

    # Prepare an image
    image = load_image(file_path, transform)
    image_tensor = image.to(device)
    
    # Generate an caption from the image
    feature = encoder(image_tensor)
    sampled_ids = decoder.sample(feature)
    sampled_ids = sampled_ids[0].cpu().numpy()          # (1, max_seq_length) -> (max_seq_length)
    
    # Convert word_ids to words
    sampled_caption = []
    for word_id in sampled_ids:
        word = vocab.idx2word[word_id]
        sampled_caption.append(word)
        if word == '<end>':
            break
    sentence = ' '.join(sampled_caption)
    return sentence




# Create a Flask instance
app = Flask(__name__)
app.config["SWAGGER"] = {"title": "Swagger-UI", "uiversion": 2}


# swagger configuration
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,  # all in
            "model_filter": lambda tag: True,  # all in
        }
    ],
    "static_url_path": "/flasgger_static",
    # "static_folder": "static",  # must be set by user
    "swagger_ui": True,
    "specs_route": "/swagger/",
}

template = dict(
    swaggerUiPrefix=LazyString(lambda: request.environ.get("HTTP_X_SCRIPT_NAME", ""))
)

app.json_encoder = LazyJSONEncoder
swagger = Swagger(app, config=swagger_config, template=template)


PWD = os.path.dirname(os.path.realpath(__file__))


# Flask leverages the function decorator for the endpoint registration.
# Register a function to handle POST in '/inference' URI.
@app.route('/predict', methods=['POST'])
@swag_from("swagger_config.yml")
def upload():
    """
    Serve the inference request
    Pass the image in multipart/form-data to the inference function
    """
    if request.method == 'POST':

      # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        return infer(file_path)
    return ''

# Start the web server
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')
if __name__ == '__main__':
    app.run()

