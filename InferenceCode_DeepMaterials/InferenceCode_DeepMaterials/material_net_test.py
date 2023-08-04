from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
import numpy as np
import argparse
import json
import glob
import random
import collections
import math
import time
from lxml import etree
from random import shuffle

#!!!!!!!!!!!!!!If running TF v > 2.0 uncomment those lines (also remove the tensorflow import on line 5):!!!!!!!!!!!
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()



#Implementation based on https://github.com/affinelayer/pix2pix-tensorflow and modified.

#For research only, not for commercial use. Do not distribute
#Data was generated using Substance Designer and Substance Share library : https://share.allegorithmic.com

#Please contact valentin.deschaintre@inria.fr for any question (inria and Optis for Ansys collaboration).

#Source code tested for tensorflow version 1.4.0
class inputMaterial:
    def __init__(self, name, lightPower, lightXPos, lightYPos, lightZPos, camXPos, camYPos, camZPos, uvscale, uoffset, voffset, rotation, identifier, path):
        self.substanceName = name
        self.lightPower = lightPower
        self.lightXPos = lightXPos
        self.lightYPos = lightYPos
        self.lightZPos = lightZPos
        self.camXPos = camXPos
        self.camYPos = camYPos
        self.camZPos = camZPos
        self.uvscale = uvscale
        self.uoffset = uoffset
        self.voffset = voffset
        self.rotation = rotation
        self.identifier = identifier
        self.path = path

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", help="path to xml file, folder or image (defined by --imageFormat) containing information images")
parser.add_argument("--mode", required=True, choices=["test", "export", "eval"])
parser.add_argument("--output_dir", required=True, help="where to put output files")
parser.add_argument("--seed", type=int)
parser.add_argument("--checkpoint", required=True, default=None, help="directory with checkpoint to resume training from or use for testing")

parser.add_argument("--testMode", type=str, default="auto", choices=["auto", "xml", "folder", "image"], help="Which loss to use instead of the L1 loss")
parser.add_argument("--imageFormat", type=str, default="png", choices=["jpg", "png", "jpeg", "JPG", "JPEG", "PNG"], help="Which format have the input files")
parser.add_argument("--aspect_ratio", type=float, default=1.0, help="aspect ratio of output images (width/height)")
parser.add_argument("--batch_size", type=int, default=1, help="number of images in batch")
parser.add_argument("--which_direction", type=str, default="AtoB", choices=["AtoB", "BtoA"])
parser.add_argument("--ngf", type=int, default=64, help="number of generator filters in first conv layer")
parser.add_argument("--ndf", type=int, default=64, help="number of discriminator filters in first conv layer")
parser.add_argument("--scale_size", type=int, default=288, help="scale images to this size before cropping to 256x256")
parser.add_argument("--flip", dest="flip", action="store_true", help="flip images horizontally")
parser.add_argument("--no_flip", dest="flip", action="store_false", help="don't flip images horizontally")
parser.set_defaults(flip=True)
parser.add_argument("--nbTargets", type=int, default=1, help="Number of images to output")
parser.add_argument("--depthFactor", type=int, default=0, help="Factor for the capacity of the network")
parser.add_argument("--loss", type=str, default="l1", choices=["l1", "specuRough", "render", "flatMean", "l2", "renderL2"], help="Which loss to use instead of the L1 loss")
parser.add_argument("--useLog", dest="useLog", action="store_true", help="Use the log for input")
parser.set_defaults(useLog=False)
parser.add_argument("--logOutputAlbedos", dest="logOutputAlbedos", action="store_true", help="Log the output albedos ? ?")
parser.set_defaults(logOutputAlbedos=False)

a = parser.parse_args()

EPS = 1e-12
CROP_SIZE = 256

if a.testMode == "auto":
    if a.input_dir.lower().endswith(".xml"):
        a.testMode = "xml";
    elif os.path.isdir(a.input_dir):
        a.testMode = "folder";
    else:
        a.testMode = "image";

Examples = collections.namedtuple("Examples", "iterator, paths, inputs, targets, count, steps_per_epoch")
Model = collections.namedtuple("Model", "outputs")

if a.depthFactor == 0:
    a.depthFactor = a.nbTargets

def preprocess(image):
    with tf.name_scope("preprocess"):
        # [0, 1] => [-1, 1]
        return image * 2 - 1

def deprocess(image):
    with tf.name_scope("deprocess"):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2

def int_shape(x):
    return list(map(int, x.get_shape()))

def conv(batch_input, out_channels, stride):
    with tf.variable_scope("conv"):
        in_channels = batch_input.get_shape()[3]
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
        #     => [batch, out_height, out_width, out_channels]
        padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
        conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
        return conv

def lrelu(x, a):
    with tf.name_scope("lrelu"):
        # adding these together creates the leak part and linear part
        # then cancels them out by subtracting/adding an absolute value term
        # leak: a*x/2 - a*abs(x)/2
        # linear: x/2 + abs(x)/2

        # this block looks like it has 2 inputs on the graph unless we do this
        x = tf.identity(x)
        return (0.5 * (1 + a)) * x + (0.5 * (1 - a)) * tf.abs(x)

def instancenorm(input):
    with tf.variable_scope("instancenorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[3]
        offset = tf.get_variable("offset", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [1, 1, 1, channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=[1, 2], keep_dims=True)
        #[batchsize ,1,1, channelNb]
        variance_epsilon = 1e-5
        #Batch normalization function does the mean substraction then divide by the standard deviation (to normalize it). It finally multiply by scale and adds offset.
        #normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        #For instanceNorm we do it ourselves :
        normalized = (((input - mean) / tf.sqrt(variance + variance_epsilon)) * scale) + offset
        return normalized, mean, variance

def deconv(batch_input, out_channels):
   with tf.variable_scope("deconv"):
        in_height, in_width, in_channels = [int(batch_input.get_shape()[1]), int(batch_input.get_shape()[2]), int(batch_input.get_shape()[3])]
        #filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        filter1 = tf.get_variable("filter1", [4, 4, out_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))

        # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
        #     => [batch, out_height, out_width, out_channels]
        resized_images = tf.image.resize_images(batch_input, [in_height * 2, in_width * 2], method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        conv = tf.nn.conv2d(resized_images, filter, [1, 1, 1, 1], padding="SAME")
        conv = tf.nn.conv2d(conv, filter1, [1, 1, 1, 1], padding="SAME")

        #conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
        return conv

def createMaterialTable(examplesDict, shuffleImages):
    materialsList = []
    pathsList = []
    flatPathsList = []
    examplesDictKeys = examplesDict.keys()
    examplesDictKeys = sorted(examplesDict)
    
    for substanceName in examplesDictKeys:
        for variationKey, variationList in examplesDict[substanceName].items():
            materialsList.append(variationList)
            tmpPathList = []
            if a.mode == "test":
                for variation in variationList:
                    tmpPathList.append(variation.path)
            else:
                if len(variationList) > 1:
                    randomChoices = np.random.choice(variationList, 2, replace = False)
                    tmpPathList.append(randomChoices[0].path)
                    tmpPathList.append(randomChoices[1].path)
                else:
                    tmpPathList.append(variationList[0].path)
            pathsList.append(tmpPathList)
    if shuffleImages == True:
        shuffle(pathsList)

    for elem in pathsList:
        flatPathsList.extend(elem)
    return flatPathsList

    #Simply return the path in an array.
def readInputImage(inputPath):
    return [inputPath]
    
    #Returns all paths to images in the targeted directory matching the required image format.
def readInputFolder(input_dir, shuffle):
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
        
    pathList = glob.glob(os.path.join(input_dir, "*." + a.imageFormat))
    pathList = sorted(pathList);
    
    if shuffle:
        pathList = shuffle(pathList)
    return pathList
    
    #Reads all information in an xml and returns a list of the paths to input images.
def readInputXML(inputPath, shuffle):
    exampleDict = {}
    pathDict = {}
    tree = etree.parse(inputPath)
    for elem in tree.findall('.//item'):
        imagePath = elem.find('image').text
        if not (imagePath is None) and os.path.exists(imagePath):
            lightPower = elem.find('lightPower').text
            lightXPos = elem.find('lightXPos').text
            lightYPos = elem.find('lightYPos').text
            lightZPos = elem.find('lightZPos').text
            camXPos = elem.find('camXPos').text
            camYPos = elem.find('camYPos').text
            camZPos = elem.find('camZPos').text
            uvscale = elem.find('uvscale').text
            uoffset = elem.find('uoffset').text
            voffset = elem.find('voffset').text
            rotation = elem.find('rotation').text
            identifier = elem.find('identifier').text
            
            substanceName = imagePath.split("/")[-1]
            if(substanceName.split('.')[0].isdigit()):
                substanceName = '%04d' % int(substanceName.split('.')[0])
            substanceNumber = 0
            imageSplitsemi = imagePath.split(";")
            if len(imageSplitsemi) > 1:                    
                substanceName = imageSplitsemi[1]
                substanceNumber = imageSplitsemi[2].split(".")[0]

            material = inputMaterial(substanceName, lightPower, lightXPos, lightYPos, lightZPos, camXPos, camYPos, camZPos, uvscale, uoffset, voffset, rotation, identifier, imagePath)
            idkey = str(substanceNumber) +";"+ identifier.rsplit(";", 1)[0]
            
            if not (substanceName in exampleDict) :
                exampleDict[substanceName] = {idkey : [material]}
                pathDict[imagePath] = material

            else:
                if not (idkey in exampleDict[substanceName]):
                    exampleDict[substanceName][idkey] = [material]
                    pathDict[imagePath] = material

                else:
                    exampleDict[substanceName][idkey].append(material)
    flatPathList = createMaterialTable(exampleDict, shuffle)
    return flatPathList

def _parse_function(filename):
    image_string = tf.io.read_file(filename)

    raw_input = tf.image.decode_image(image_string)
    raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)
   
    assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message="image does not have 3 channels")
    with tf.control_dependencies([assertion]):
        raw_input = tf.identity(raw_input)
        raw_input.set_shape([None, None, 3])
        images=[]
        input = raw_input
        #add black images as targets if we just want to evaluate input images with no targets.
        if a.mode == "eval":
            shape = tf.shape(input)
            black = tf.zeros([shape[0], shape[1]  * a.nbTargets, shape[2]], dtype=tf.float32)
            input = tf.concat([input, black], axis=1)
        width = tf.shape(input)[1] # [height, width, channels]
        imageWidth = width // (a.nbTargets + 1)

        for imageId in range(a.nbTargets + 1):
            beginning = imageId * imageWidth
            end = (imageId+1) * imageWidth
            images.append(input[:,beginning:end,:])

    if a.which_direction == "AtoB":
        inputs, targets = [images[0], images[1:]]
    elif a.which_direction == "BtoA":
        inputs, targets = [images[-1], images[:-1]]
    else:
        raise Exception("invalid direction")
    if a.useLog:
        inputs = logTensor(inputs)
    inputs = preprocess(inputs)
    targetsTmp = []
    for target in targets:
        targetsTmp.append(preprocess(target))
    targets = targetsTmp
    # synchronize seed for image operations so that we do the same operations to both
    # input and output images 
    
    def transform(image):
        r = image
        if a.flip:
            r = tf.image.random_flip_left_right(r, seed=seed)

        # area produces a nice downscaling, but does nearest neighbor for upscaling
        # assume we're going to be doing downscaling here
        r = tf.image.resize(r, [a.scale_size, a.scale_size], method=tf.image.ResizeMethod.AREA)
        return r

    with tf.name_scope("input_images"):
        input_images = transform(inputs)

    with tf.name_scope("target_images"):
        target_images = []
        for target in targets:
            target_images.append(transform(target))
    
    return filename, input_images, target_images    

def load_examples(input_dir, shuffleValue):
    test_queue = tf.constant([" "])
    if input_dir is None or not os.path.exists(input_dir):
        raise Exception("input_dir does not exist")
    flatPathList = []
    if a.testMode == "xml":
        flatPathList = readInputXML(input_dir, shuffleValue)
    elif a.testMode == "folder":
        flatPathList = readInputFolder(input_dir, shuffleValue)
    elif a.testMode == "image":
        flatPathList = readInputImage(input_dir)
    
        
    if len(flatPathList) == 0:
        raise Exception("input_dir contains no image files")

    with tf.name_scope("load_images"):
        filenamesTensor = tf.constant(flatPathList) 
        dataset = tf.data.Dataset.from_tensor_slices(filenamesTensor)
        dataset = dataset.map(_parse_function, num_parallel_calls=1)
        dataset = dataset.repeat()
        batched_dataset = dataset.batch(a.batch_size)

    for paths_batch, inputs_batch, targets_batch in batched_dataset:
        if a.scale_size > CROP_SIZE:
            xyCropping = tf.random_uniform([2], 0, a.scale_size - CROP_SIZE, dtype=tf.int32)
            inputs_batch = inputs_batch[:, xyCropping[0]: xyCropping[0] + CROP_SIZE,
                           xyCropping[1]: xyCropping[1] + CROP_SIZE, :]
            targets_batch = targets_batch[:, :, xyCropping[0]: xyCropping[0] + CROP_SIZE,
                            xyCropping[1]: xyCropping[1] + CROP_SIZE, :]

    steps_per_epoch = int(math.floor(len(flatPathList) / a.batch_size))
    print("Steps per epoch : " + str(steps_per_epoch))
    #[batchsize, nbMaps, 256,256,3] Do the reshape by hand and probably concat it on third axis so we are sure of the reshape.
    targets_batch_concat = targets_batch[:,0]
    for imageId in range(1, a.nbTargets):
        targets_batch_concat = tf.concat([targets_batch_concat, targets_batch[:,imageId]], axis = -1)
    
    targets_batch = targets_batch_concat
    return Examples(
        iterator=iterator,
        paths=paths_batch,
        inputs=inputs_batch,
        targets=targets_batch,
        count=len(flatPathList),
        steps_per_epoch=steps_per_epoch,
    )
    
#input is of shape [batch, X]. Returns the outputs of the layer
def fullyConnected(input, outputDim, useBias, layerName = "layer", initMultiplyer = 1.0):
    with tf.variable_scope("fully_connected"):
        batchSize = tf.shape(input)[0];
        inputChannels = int(input.get_shape()[-1])
        weights = tf.get_variable("weight", [inputChannels, outputDim ], dtype=tf.float32, initializer=tf.random_normal_initializer(0, initMultiplyer * tf.sqrt(1.0/float(inputChannels))))
        weightsTiled = tf.tile(tf.expand_dims(weights, axis = 0), [batchSize, 1,1])
        squeezedInput = input
        
        if (len(input.get_shape()) > 3) :
            squeezedInput = tf.squeeze(squeezedInput, [1])
            squeezedInput = tf.squeeze(squeezedInput, [1])

        outputs = tf.matmul(tf.expand_dims(squeezedInput, axis = 1), weightsTiled)
        outputs = tf.squeeze(outputs, [1])
        if(useBias):
            bias = tf.get_variable("bias", [outputDim], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.002))
            outputs = outputs + tf.expand_dims(bias, axis = 0)
            
        return outputs

def GlobalToGenerator(inputs, channels):
    with tf.variable_scope("GlobalToGenerator1"):
        fc1 = fullyConnected(inputs, channels, False, "fullyConnected_global_to_unet" ,0.01)
    return tf.expand_dims(tf.expand_dims(fc1, axis = 1), axis=1)
    
def logTensor(tensor):
    return (tf.math.log(tf.add(tensor, 0.01)) - tf.math.log(0.01)) / (tf.math.log(1.01) - tf.math.log(0.01))

def create_generator(generator_inputs, generator_outputs_channels):
    layers = []
    #Input here should be [batch, 256,256,3]
    inputMean, inputVariance = tf.nn.moments(generator_inputs, axes=[1, 2], keep_dims=False)
    globalNetworkInput = inputMean
    globalNetworkOutputs = []
    with tf.variable_scope("globalNetwork_fc_1"):    
        globalNetwork_fc_1 = fullyConnected(globalNetworkInput, a.ngf * 2, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
        globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc_1))
        
    #encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
    with tf.variable_scope("encoder_1"):
        output = conv(generator_inputs, a.ngf * a.depthFactor , stride=2)
        layers.append(output)
    #Default ngf is 64
    layer_specs = [
        a.ngf * 2 * a.depthFactor, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
        a.ngf * 4 * a.depthFactor, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
        a.ngf * 8 * a.depthFactor, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
        a.ngf * 8 * a.depthFactor, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
        a.ngf * 8 * a.depthFactor, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
        a.ngf * 8 * a.depthFactor, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
        #a.ngf * 8 * a.depthFactor, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
    ]
    
    for layerCount, out_channels in enumerate(layer_specs):
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            convolved = conv(rectified, out_channels, stride=2)
            #here mean and variance will be [batch, 1, 1, out_channels]
            outputs, mean, variance = instancenorm(convolved)
            
            outputs = outputs + GlobalToGenerator(globalNetworkOutputs[-1], out_channels)
            with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):  
                nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)  
                globalNetwork_fc = ""
                if layerCount + 1 < len(layer_specs) - 1:
                    globalNetwork_fc = fullyConnected(nextGlobalInput, layer_specs[layerCount + 1], True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                else : 
                    globalNetwork_fc = fullyConnected(nextGlobalInput, layer_specs[layerCount], True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
    
                globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))
            layers.append(outputs)

    with tf.variable_scope("encoder_8"):
        rectified = lrelu(layers[-1], 0.2)
         # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
        convolved = conv(rectified, a.ngf * 8 * a.depthFactor, stride=2)
        convolved = convolved  + GlobalToGenerator(globalNetworkOutputs[-1], a.ngf * 8 * a.depthFactor)
        
        with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):  
            mean, variance = tf.nn.moments(convolved, axes=[1, 2], keep_dims=True)
            nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
            globalNetwork_fc = fullyConnected(nextGlobalInput, a.ngf * 8 * a.depthFactor, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
            globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))  
                      
        layers.append(convolved)
    #default nfg = 64
    layer_specs = [
        (a.ngf * 8 * a.depthFactor, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
        (a.ngf * 8 * a.depthFactor, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
        (a.ngf * 8 * a.depthFactor, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
        (a.ngf * 8 * a.depthFactor, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
        (a.ngf * 4 * a.depthFactor, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
        (a.ngf * 2 * a.depthFactor, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
        (a.ngf * a.depthFactor, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
    ]

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = lrelu(input, 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            output = deconv(rectified, out_channels)
            output, mean, variance = instancenorm(output)
            output = output + GlobalToGenerator(globalNetworkOutputs[-1], out_channels)
            with tf.variable_scope("globalNetwork_fc_%d" % (len(globalNetworkOutputs) + 1)):    
                nextGlobalInput = tf.concat([tf.expand_dims(tf.expand_dims(globalNetworkOutputs[-1], axis = 1), axis=1), mean], axis = -1)
                globalNetwork_fc = fullyConnected(nextGlobalInput, out_channels, True, "globalNetworkLayer" + str(len(globalNetworkOutputs) + 1))
                globalNetworkOutputs.append(tf.nn.selu(globalNetwork_fc))
            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
    with tf.variable_scope("decoder_1"):
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = lrelu(input, 0.2)
        output = deconv(rectified, generator_outputs_channels)
        output = output + GlobalToGenerator(globalNetworkOutputs[-1], generator_outputs_channels)
        output = tf.tanh(output)
        layers.append(output)

    return layers[-1]

def create_model(inputs, reuse_bool = False):
    with tf.variable_scope("generator", reuse=reuse_bool) as scope:
        out_channels = 9 
        outputs = create_generator(inputs, out_channels) 
        
    partialOutputedNormals = outputs[:,:,:,0:2]
    outputedDiffuse = outputs[:,:,:,2:5]
    outputedRoughness = outputs[:,:,:,5]
    outputedSpecular = outputs[:,:,:,6:9]
    normalShape = tf.shape(partialOutputedNormals)
    newShape = [normalShape[0], normalShape[1], normalShape[2], 1]

    tmpNormals = tf.ones(newShape, tf.float32)
    
    normNormals = tf_Normalize(tf.concat([partialOutputedNormals, tmpNormals], axis = -1))
    outputedRoughnessExpanded = tf.expand_dims(outputedRoughness, axis = -1)
    reconstructedOutputs =  tf.concat([normNormals, outputedDiffuse, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedRoughnessExpanded, outputedSpecular], axis=-1)

    return Model(
        outputs=reconstructedOutputs
    )
    
def save_loss_value(values):
    averaged = np.mean(values)
    with open(os.path.join(a.output_dir, "losses.txt"), "a") as f:
            f.write(str(averaged) + "\n")
                    
def save_images(fetches, output_dir = a.output_dir, step=None):
    image_dir = os.path.join(output_dir, "images")
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    filesets = []
    for i, in_path in enumerate(fetches["paths"]):
        name, _ = os.path.splitext(os.path.basename(in_path.decode("utf8")))
        fileset = {"name": name, "step": step}
        #fetch inputs
        kind = "inputs"
        filename = name + "-" + kind + ".png"
        if step is not None:
            filename = "%08d-%s" % (step, filename)
        fileset[kind] = filename
        out_path = os.path.join(image_dir, filename)
        contents = fetches[kind][i]
        with open(out_path, "wb") as f:
            f.write(contents)
        #fetch outputs and targets
        for kind in ["outputs", "targets"]:
            for idImage in range(a.nbTargets):
                filename = name + "-" + kind + "-" + str(idImage) + "-.png"
                if step is not None:
                    filename = "%08d-%s" % (step, filename)
                filetsetKey = kind + str(idImage)
                fileset[filetsetKey] = filename
                out_path = os.path.join(image_dir, filename)
                contents = fetches[kind][i * a.nbTargets + idImage]
                with open(out_path, "wb") as f:
                    f.write(contents)
        filesets.append(fileset)
    return filesets


def append_index(filesets, output_dir = a.output_dir, step=False):
    index_path = os.path.join(output_dir, "index.html")
    if os.path.exists(index_path):
        index = open(index_path, "a")
    else:
        mapnames = ["normals", "diffuse", "roughness", "log(specular)"]
        index = open(index_path, "w")
        index.write("<html><body><table><tr>")
        if step:
            index.write("<th>step</th>")
        index.write("<th>name</th><th>log(input)</th>")
        for idImage in range(a.nbTargets):
            index.write("<th>" + str(mapnames[idImage]) + "</th>")
        index.write("</tr>")            

    for fileset in filesets:
        index.write("<tr>")

        if step:
            index.write("<td>%d</td>" % fileset["step"])
        index.write("<td>%s targets</td>" % fileset["name"])
        if a.mode != "eval" : 

            for kind in ["inputs", "targets"]:
                if kind == "inputs":
                    index.write("<td><img src='images/%s'></td>" % fileset[kind])
                elif kind == "targets":
                    for idImage in range(a.nbTargets):
                        filetsetKey = kind + str(idImage)
                        index.write("<td><img src='images/%s'></td>" % fileset[filetsetKey])
            index.write("</tr>")
            index.write("<tr>")

        if step:
            index.write("<td></td>")
        index.write("<td>outputs</td>")
        for kind in ["inputs", "outputs"]:
            if kind == "inputs":
                index.write("<td><img src='images/%s'></td>" % fileset[kind])
            elif kind=="outputs":
                for idImage in range(a.nbTargets):
                    filetsetKey = kind + str(idImage)
                    index.write("<td><img src='images/%s'></td>" % fileset[filetsetKey])
        index.write("</tr>")
    
    return index_path

def runTestFromTrain(currentStep, evalExamples, max_steps, display_fetches_test, sess):
    max_steps = min(evalExamples.steps_per_epoch, max_steps)
    for step in range(max_steps):
        try:
            results_test = sess.run(display_fetches_test)
            test_output_dir = a.output_dir + "/testStep"+str(currentStep)
            filesets = save_images(results_test, test_output_dir)
            index_path = append_index(filesets, test_output_dir)
        except tf.errors.OutOfRangeError:
            print("Error in the runTestFromTrain of OutOfRangeError")
            continue;    
    print("Wrote index at", index_path)
    
def main():

    if a.seed is None:
        a.seed = random.randint(0, 2**31 - 1)

    tf.random.set_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)

    if not os.path.exists(a.output_dir):
        os.makedirs(a.output_dir)

    if a.mode == "test" or a.mode == "export" or a.mode == "eval" :
        if a.checkpoint is None:
            raise Exception("checkpoint required for test, export or eval mode")

        # load some options from the checkpoint
        options = {"which_direction", "ngf", "ndf", "nbTargets", "depthFactor", "loss", "useLog"}
        with open(os.path.join(a.checkpoint, "options.json")) as f:
            for key, val in json.loads(f.read()).items():
                if key in options:
                    print("Loaded", key, "=", val)
                    setattr(a, key, val)
        # disable these features in test mode
        a.scale_size = CROP_SIZE
        a.flip = False

    for k, v in a._get_kwargs():
        print(k, "=", v)

    with open(os.path.join(a.output_dir, "options.json"), "w") as f:
        f.write(json.dumps(vars(a), sort_keys=True, indent=4))

    examples = load_examples(a.input_dir, a.mode == "train")
    print(a.mode + " set size : %d" % examples.count)

    # inputs and targets are [batch_size, height, width, channels]
    model = create_model(examples.inputs, False)
    tmpTargets = examples.targets
    
    # undo colorization splitting on images that we use for display/output    
    inputs = deprocess(examples.inputs)
    targets = deprocess(tmpTargets)
    outputs = deprocess(model.outputs)
    
        
    def convert(image, squeeze=False):
        if a.aspect_ratio != 1.0:
            # upscale to correct aspect ratio
            size = [CROP_SIZE, int(round(CROP_SIZE * a.aspect_ratio))]
            image = tf.image.resize_images(image, size=size, method=tf.image.ResizeMethod.BICUBIC)

        if squeeze:
            def tempLog(imageValue):                    
                imageValue= tf.log(imageValue + 0.01)
                imageValue = imageValue - tf.reduce_min(imageValue)
                imageValue = imageValue / tf.reduce_max(imageValue)
                return imageValue
            image = [tempLog(imageVal) for imageVal in image]
        #imageUint = tf.clip_by_value(image, 0.0, 1.0)
        #imageUint = imageUint * 65535.0
        #imageUint16 = tf.cast(imageUint, tf.uint16)
        #return imageUint16
        return tf.image.convert_image_dtype(image, dtype=tf.uint16, saturate=True)

    with tf.name_scope("transform_images"):
        targets_list = tf.split(targets, a.nbTargets, axis=3)#4 * [batch, 256,256,3]
        if a.logOutputAlbedos:
            targets_list[-1] = logTensor(targets_list[-1])
            targets_list[1] = logTensor(targets_list[1])        
        
        targets = tf.stack(targets_list, axis = 1) #[batch, 4,256,256,3]
        shape = tf.shape(targets)
        newShape = tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0)

        targets_reshaped = tf.reshape(targets, newShape)
        outputs_list = tf.split(outputs, a.nbTargets, axis=3)#4 * [batch, 256,256,3]
        if a.logOutputAlbedos:
            outputs_list[-1] = logTensor(outputs_list[-1])
            outputs_list[1] = logTensor(outputs_list[1])
        
        outputs = tf.stack(outputs_list, axis = 1) #[batch, 4,256,256,3]
        shape = tf.shape(outputs)
        newShape = tf.concat([[shape[0] * shape[1]], shape[2:]], axis=0)
        outputs_reshaped = tf.reshape(outputs, newShape)

    # reverse any processing on images so they can be written to disk or displayed to user
    with tf.name_scope("convert_inputs"):
        converted_inputs = convert(inputs)
    with tf.name_scope("convert_targets"):
        converted_targets = convert(targets_reshaped)
    with tf.name_scope("convert_outputs"):
        converted_outputs = convert(outputs_reshaped)
    with tf.name_scope("encode_images"):
        display_fetches = {
            "paths": examples.paths,
            "inputs": tf.map_fn(tf.image.encode_png, converted_inputs, dtype=tf.string, name="input_pngs"),
            "targets": tf.map_fn(tf.image.encode_png, converted_targets, dtype=tf.string, name="target_pngs"),
            "outputs": tf.map_fn(tf.image.encode_png, converted_outputs, dtype=tf.string, name="output_pngs"),
        }

    with tf.name_scope("parameter_count"):
        parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) for v in tf.trainable_variables()])

    saver = tf.train.Saver(max_to_keep=1)

    logdir = None
    sv = tf.train.Supervisor(logdir=logdir, save_summaries_secs=0, saver=None)
    with sv.managed_session() as sess:
        print("Parameter_count =", sess.run(parameter_count))
        if a.checkpoint is None:
            print("Checkpoint is required, this is test only")
            return
        if a.checkpoint is not None:
            print("Loading model from checkpoint : " + a.checkpoint)
            checkpoint = tf.train.latest_checkpoint(a.checkpoint)
            saver.restore(sess, checkpoint)

        max_steps = 2**32
        
        sess.run(examples.iterator.initializer)
        if a.mode == "test" or a.mode == "eval":
            # testing at most, process the test data once
            max_steps = min(examples.steps_per_epoch, max_steps)
            for step in range(max_steps):
                try:
                    results = sess.run(display_fetches)
                    filesets = save_images(results)
                    for i, f in enumerate(filesets):
                        print("Evaluated image", f["name"])
                    index_path = append_index(filesets)
                except tf.errors.OutOfRangeError :
                    print("Testing fails in OutOfRangeError, seems that images couldn't be found ?")
                    continue;
    
# Normalizes a tensor troughout the Channels dimension (BatchSize, Width, Height, Channels)
# Keeps 4th dimension to 1. Output will be (BatchSize, Width, Height, 1).
def tf_Normalize(tensor):
    Length = tf.sqrt(tf.reduce_sum(tf.square(tensor), axis = -1, keep_dims=True))
    return tf.div(tensor, Length)
    
main()
