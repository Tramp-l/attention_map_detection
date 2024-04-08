import torch

def extract_attention_weights(model,input_images):

    model.eval()

    with torch.no_grad():

        outputs = model(input_images, output_attentions=True)
        attention_weights = outputs.attentions

    attention_weights_layer_6 = attention_weights[5]
    attention_weights_layer_12 = attention_weights[11]

    return attention_weights_layer_6, attention_weights_layer_12