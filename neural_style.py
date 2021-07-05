# import numpy, tensorflow and matplotlib 
import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
from numba import vectorize
  
# import VGG 19 model and keras Model API 
from tensorflow.python.keras.applications.vgg19 import VGG19, preprocess_input 
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array 
from tensorflow.python.keras.models import Model 


content_path = r'D:\\duke\\project\\neural_style_transfer\\content\\1.jpg'
style_path = r'D:\\duke\\project\\neural_style_transfer\\style\\5.jpg'

# this function download the VGG model and initiliase it 
model = VGG19( 
    include_top=False, 
    weights='imagenet'
) 
# set training to False 
model.trainable = False
# Print details of different layers 
  
model.summary()

def load_and_process_image(image_path): 
    img = load_img(image_path) 
    # convert image to array 
    img = img_to_array(img) 
    img = preprocess_input(img) 
    img = np.expand_dims(img, axis=0) 
    return img

def deprocess(img): 
    # perform the inverse of the pre processing step 
    img[:, :, 0] += 103.939
    img[:, :, 1] += 116.779
    img[:, :, 2] += 123.68
    # convert RGB to BGR 
    img = img[:, :, ::-1] 
  
    img = np.clip(img, 0, 255).astype('uint8') 
    return img 
  
  
def display_image(image): 
    # remove one dimension if image has 4 dimension 
    if len(image.shape) == 4: 
        img = np.squeeze(image, axis=0) 
  
    img = deprocess(img) 
  
    plt.grid(False) 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(img) 
    return

def save_image(image,i): 
    # remove one dimension if image has 4 dimension 
    if len(image.shape) == 4: 
        img = np.squeeze(image, axis=0) 
  
    img = deprocess(img) 
  
    plt.grid(False) 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(img)
    plt.savefig(r'D:\\duke\\project\\neural_style_transfer\\output3\\' + str(i) +'.png')

    return

content_img = load_and_process_image(content_path) 
display_image(content_img) 
  
# load style image 
style_img = load_and_process_image(style_path) 
display_image(style_img) 

# define content model 
content_layer = 'block5_conv2'
content_model = Model( 
    inputs=model.input, 
    outputs=model.get_layer(content_layer).output 
) 
content_model.summary() 

style_layers = [ 
    'block1_conv1', 
    'block3_conv1', 
    'block5_conv1'
] 
style_models = [Model(inputs=model.input, 
                      outputs=model.get_layer(layer).output) for layer in style_layers] 

                      # gram matrix 
def gram_matrix(A): 
    channels = int(A.shape[-1]) 
    a = tf.reshape(A, [-1, channels]) 
    n = tf.shape(a)[0] 
    gram = tf.matmul(a, a, transpose_a=True) 
    return gram / tf.cast(n, tf.float32) 
  
  
weight_of_layer = 1. / len(style_models) 

# style loss 
def style_cost(style, generated): 
    J_style = 0
  
    for style_model in style_models: 
        a_S = style_model(style) 
        a_G = style_model(generated) 
        GS = gram_matrix(a_S) 
        GG = gram_matrix(a_G) 
        current_cost = tf.reduce_mean(tf.square(GS - GG)) 
        J_style += current_cost * weight_of_layer 
  
    return J_style,a_G

# Content loss 
def content_cost(content, generated,a_G): 
    a_C = content_model(content) 
    loss = tf.reduce_mean(tf.square(a_C - a_G)) 
    return loss

# training function 
generated_images = [] 
  
def training_loop(content_path, style_path, iterations=50, a=10, b=1000): 
    # load content and style images from their repsective path 
    content = load_and_process_image(content_path) 
    style = load_and_process_image(style_path) 
    generated = tf.Variable(content, dtype=tf.float32) 
  
    opt = tf.keras.optimizers.Adam(learning_rate=7) 
  
    best_cost = float('Inf')
    best_image = None
    for i in range(iterations): 
        
        with tf.GradientTape() as tape: 
            J_style,a_G = style_cost(style, generated)
            J_content = content_cost(content, generated,a_G)  
            J_total = a * J_content + b * J_style 
  
        grads = tape.gradient(J_total, generated) 
        opt.apply_gradients([(grads, generated)]) 
  
        if J_total < best_cost: 
            best_cost = J_total 
            best_image = generated.numpy() 
  
        print("Iteration :{}".format(i)) 
        print('Total Loss {:e}.'.format(J_total)) 
        generated_images.append(generated.numpy()) 
        save_image(generated_images[i],i)
  
    return best_image


# Train the model and get best image 

final_img = training_loop(content_path, style_path) 