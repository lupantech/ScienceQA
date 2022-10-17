import os
import json
import torch
import warnings

from tqdm import tqdm
from PIL import Image
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer

warnings.filterwarnings("ignore")


def predict_caption(image_paths):
    images = []
    for image_path in image_paths:
        i_image = Image.open(image_path)
        if i_image.mode != "RGB":
            i_image = i_image.convert(mode="RGB")

        images.append(i_image)

    pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    output_ids = model.generate(pixel_values, **gen_kwargs)

    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    preds = [pred.strip() for pred in preds]
    return preds


if __name__ == "__main__":
    ## Arguments
    input_path = "../data/scienceqa"
    output_path = "../data"
    output_name = "captions_user.json"

    model_name = "nlpconnect/vit-gpt2-image-captioning"
    feature_extractor_name = "nlpconnect/vit-gpt2-image-captioning"
    tokenizer_name = "nlpconnect/vit-gpt2-image-captioning"

    max_length = 64
    num_beams = 4
    gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

    ## Read data
    problems = json.load(open(os.path.join(input_path, 'problems.json')))
    pids = [pid for pid in list(problems.keys()) if problems[pid]['image'] == 'image.png']

    print("number of images: ", len(pids))
    print(pids[:10])

    ## Prepare the model
    # url = "https://huggingface.co/sachin/vit2distilgpt2"
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    feature_extractor = ViTFeatureExtractor.from_pretrained(feature_extractor_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ## Generate image captions
    captions = {}

    print(f"Generating captions!")
    for pid in tqdm(pids):
        image_file = os.path.join(input_path, 'images', problems[pid]['split'], pid, 'image.png')
        try:
            caption = predict_caption([image_file])[0]
            captions[pid] = caption.capitalize() + '.'
        except Exception as e:
            print(image_file)
            print(e)

    ## Save the captions
    output_file = os.path.join(output_path, output_name)
    os.makedirs(output_path, exist_ok=True)
    print(f"Saved to {output_file}")

    results = {
        "model": model_name,
        "feature_extractor": feature_extractor_name,
        "tokenizer": tokenizer_name,
        "max_length": max_length,
        "num_beams": num_beams,
        "captions": captions
    }

    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, separators=(',', ': '))
