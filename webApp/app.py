from flask import Flask, request, render_template
from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from google.cloud import storage
import os
app = Flask(__name__)
                 
   
class LLMModel:
    def __init__(self, model_path: str):
        self.tokenizer = BartTokenizer.from_pretrained(model_path)
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.model.eval()
        if torch.cuda.is_available():
            self.model.to('cuda')

    def predict(self, input_text: str):
        inputs = self.tokenizer.encode(input_text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = inputs.to('cuda')
        outputs = self.model.generate(inputs, max_length=400, early_stopping=True,do_sample=True,
                                             num_beams=10,temperature=0.8,top_k=50,top_p=0.95,no_repeat_ngram_size=3,length_penalty=1.0,repetition_penalty=1.1)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

def download_model(storage_client,bucket_name, source_blob_prefix, destination_dir):
    # Downloads a model from GCS to a local directory.
    bucket = storage_client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=source_blob_prefix)
    for blob in blobs:
        destination_file_name = os.path.join(destination_dir, blob.name[len(source_blob_prefix):])
        os.makedirs(os.path.dirname(destination_file_name), exist_ok=True)
        blob.download_to_filename(destination_file_name)
        print(f"Downloaded {blob.name} to {destination_file_name}.")

    print("model downloaded successfully")


# Now you can initialize the GCP client
#storage_client = storage.Client()
credentials_file="int-infra-training-gcp-b4ede48008c9.json"
#storage_client = storage.Client.from_service_account_json(credentials_file)
bucket_name = 'my-bucket-int-infra-training-gcp'
source_blob_prefix = 'bart-finetuned/'

# Directory to store the downloaded model files
destination_dir = 'bart-finetuned'


# Download the model from GCS
download_model(storage_client,bucket_name, source_blob_prefix, destination_dir)

# Load the model using the LLMModel class
fine_tuned_model = LLMModel(destination_dir) 
original_model = LLMModel('facebook/bart-large')

@app.route("/", methods=['GET', 'POST'])
def query_model():
    answers = {}
    if request.method == 'POST':
        input_text = request.form.get('input_text')
        if input_text:
            answers['fine_tuned'] = fine_tuned_model.predict(input_text)
            answers['original'] = original_model.predict(input_text)
    return render_template("query_page.html", answers=answers)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
