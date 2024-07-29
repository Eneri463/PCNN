import tensorflow.compat.v1 as tf
import numpy as np
from PIL import Image
from tensorflow import keras
from random import randint
import cv2
import os
from imutils import paths
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16

tf.disable_eager_execution()

batchSize = 6
imageHeight = 256
imageWidth = 256
imageChannels = 3
epochs = 30
pp = 70000 # количество изображений, которые будут использоваться для обучения/тестирования
pathM = 'data/masks'
pathMT = 'data/masks_test'
pathI = 'data/CelebA'


# -------------------------------------------------------------------
# класс, создающий сеть с частичными свёртками и обучающий её
class PCNN:
    
    # загрузка заданных слоёв предобученой модели vgg-16
    def vgg16Layers(self, layerNames):
        
        vgg = VGG16(include_top=False, weights='imagenet')
        vgg.trainable = False

        outputs = [vgg.get_layer(name).output for name in layerNames]

        model = tf.keras.Model([vgg.input], outputs)
        
        return model
    
    # вычисление матрицы Грама
    def gramMatrix(self, inputTensor):
        
        result = tf.linalg.einsum('bijc,bijd->bcd', inputTensor, inputTensor)
        inputShape = tf.shape(inputTensor)
        numLocations = tf.cast(inputShape[1]*inputShape[2], tf.float32)
        
        return result/(numLocations)
    
    # функция потери L1
    def L1(self, x):
        
        if len(x.get_shape()) == 4:
            return tf.reduce_mean(tf.reduce_sum(tf.math.abs(x), [1, 2, 3]))
        else: 
            return tf.reduce_mean(tf.reduce_sum(tf.math.abs(x), [1, 2]))
    
    # вычисление общей функции потерь
    def lossFunction(self, mask, yTrue, yOut):
    
        lValid = self.L1(mask*(yOut-yTrue))
        lHole = self.L1((1-mask)*(yOut-yTrue))
        
        
        yСomp = (mask) * yTrue + (1-mask) * yOut
        
        compOut = self.vgg(yСomp)
        trueOut = self.vgg(yTrue)
        outOut = self.vgg(yOut)
        
        lPerceptual = 0
        lStyleOut = 0
        lStyleComp = 0
        
        for psiOut, psiComp, psiTrue in zip(outOut, compOut, trueOut):
            
            lPerceptual += self.L1(psiOut-psiTrue) + self.L1(psiComp-psiTrue)
            
            gramOut = self.gramMatrix(psiOut)
            gramTrue = self.gramMatrix(psiTrue)
            gramComp = self.gramMatrix(psiComp)
            
            lStyleOut += self.L1(gramOut-gramTrue)
            lStyleComp += self.L1(gramComp-gramTrue)
        
        lTv = tf.reduce_mean(tf.image.total_variation(yСomp))
        
        lTotal = lValid + 6*lHole+0.05*lPerceptual+120*(lStyleOut+lStyleComp)+0.1*lTv
        
        return lTotal
        
    # конструктор
    def __init__(self, trainData, testData, epochs, batchSize, imageHeight = 227, imageWidth = 227, imageChannels = 3):
        
        tf.reset_default_graph()
        
        # ----- гиперпараметры обучения
        self.epochs = epochs                   # количество эпох
        self.batchSize = batchSize             # размер одного батча
        self.imageHeight = imageHeight         # высота изображения
        self.imageWidth = imageWidth           # ширина изображения
        self.imageChannels = imageChannels     # количество каналов в изображении
        
        # ----- объекты и данные, используемые при обучении
        
        self.vgg = self.vgg16Layers(['block1_pool','block2_pool','block3_pool'])
        
        # данные для обучения
        self.trainData = trainData
        self.testData = testData
        # исходное изображение
        self.images =  tf.placeholder(tf.float32, [self.batchSize, self.imageHeight, self.imageWidth, self.imageChannels])
        # исходное изображение с повреждённой областью внутри
        self.damagedInputs = tf.placeholder(tf.float32, [self.batchSize, self.imageHeight, self.imageWidth, self.imageChannels])
        # маска восстанавливаемой области
        self.masks = tf.placeholder(tf.float32, [self.batchSize, self.imageHeight, self.imageWidth, self.imageChannels])
        # генератор PCNN
        generator = GEN("PCNN")
        
        self.outputs = generator(self.damagedInputs, self.masks)
        
        self.loss = self.lossFunction(self.masks, self.images, self.outputs)
        
        self.genOptimizer = tf.train.AdamOptimizer(2e-4).minimize(self.loss, var_list=generator.get_var())
        
        self.costGen = tf.summary.scalar("Loss", self.loss)
        self.merged = tf.summary.merge_all()
        self.writerTest = tf.summary.FileWriter("./logs/test")
        self.writerTrain = tf.summary.FileWriter("./logs/train")
        
        self.sess = tf.Session()
        
        self.sess.run(tf.global_variables_initializer())
        
        self.saver = tf.train.Saver()
        
        
    # -------------------------------------------------------------- 
    # обучение
    def train(self, i=0, whenSave = 1):
        
        tf.reset_default_graph()
        
        self.writerTrain.add_graph(self.sess.graph)
        self.writerTest.add_graph(self.sess.graph)
        
        for epoch in range(self.epochs):
            
            im = []
            im2 = []
            
            # ----- шаг обучения
            for numberBatch in range(len(self.trainData)):
                
                originalImgs, damagedImgs, masks = self.trainData[numberBatch]
                
                self.sess.run(self.genOptimizer, feed_dict={self.images: originalImgs, self.damagedInputs: damagedImgs, self.masks: masks})
            
            # ----- вывод промежуточных результатов:
            originalImgs2, damagedImgs2, masks2 = self.testData[0]

            summaryTrain, resLossTrain = self.sess.run([self.merged, self.loss], feed_dict={self.images: originalImgs, self.damagedInputs: damagedImgs, self.masks: masks})

            summaryTest, resLossTest = self.sess.run([self.merged, self.loss], feed_dict={self.images: originalImgs2, self.damagedInputs: damagedImgs2, self.masks: masks2})

            self.writerTrain.add_summary(summaryTrain, i)
            self.writerTest.add_summary(summaryTest, i)

            print("Итерация " + str(i) + ", loss = " + str(resLossTrain) + ", loss Test = " + str(resLossTest) + ".")

            # ----- сохраняем параметры нейросети каждые whenSave эпох
            if (epoch + 1) % whenSave == 0:

                resImage = self.sess.run([self.outputs], feed_dict={self.damagedInputs: damagedImgs2, self.masks: masks2})
                
                Image.fromarray(np.uint8(resImage[0][0]*255)).save("./Results//" + str(i) + ".jpg")
                Image.fromarray(np.uint8(damagedImgs2[0]*255)).save("./Results//" + str(i) + "_1.jpg")
                self.saver.save(self.sess, "./save_para//para.ckpt")

            self.trainData.on_epoch_end()
            self.testData.on_epoch_end()
            i=i+1

    # -------------------------------------------------------------- 
    # восстановление данных
    def restoreModel(self, pathMeta, path):

        self.saver = tf.train.import_meta_graph(pathMeta)
        self.saver.restore(self.sess, tf.train.latest_checkpoint(path))
    
    # -------------------------------------------------------------- 
    # использование готовой модели для восстановления изображения:
    def useModel(self, image, mask):
        
        resImage = self.sess.run([self.outputs], feed_dict={self.damagedInputs: image, self.masks: mask})
        Image.fromarray(np.uint8(resImage[0][0]*255)).save("./Results.jpg")
        print("Результат сохранен")
        
    # -------------------------------------------------------------- 
    # использование готовой модели для восстановления изображения:
    def useModel(self, image, mask):
        
        resImage = self.sess.run([self.outputs], feed_dict={self.damagedInputs: image, self.masks: mask})
        Image.fromarray(np.uint8(resImage[0][0]*255)).save("./Results.jpg")
        print("Результат сохранен")
    
# -------------------------------------------------------------------
# реализация слоёв нейронной сети 

# ----- слой кодировщика
def encoderLayers(name, inputs, masks, filters, kSize, batchNorm = True, biasUse = True):
    
    with tf.variable_scope(name + 'en'):
    
        outputsImgs, outputsMasks = pConv2D(name, inputs, masks, filters, kSize, stride=2, biasUse = biasUse)
        
        if biasUse:
            b = tf.get_variable("b",shape=[filters], initializer=tf.constant_initializer(0.))
            outputsImgs = tf.nn.bias_add(outputsImgs, b)

        outputsImgs = keras.layers.ReLU()(outputsImgs)

        if batchNorm:
            outputsImgs = tf.layers.batch_normalization(outputsImgs)
    
    return outputsImgs, outputsMasks

# ----- слой декодировщика
def decoderLayers(name, inputs, masks, concatIm, concatMask, filters, kSize, activationFunc = True, batchNorm = True, biasUse = True):
    
    with tf.variable_scope(name + 'de'):
        
        outputsImgs = upSampling("up" + name, inputs, size=(2,2))
        outputsMasks = upSampling("up" + name + "Masks", masks, size=(2,2))

        outputsImgs = keras.layers.concatenate(inputs = [concatIm, outputsImgs], axis=-1)
        outputsMasks = keras.layers.concatenate(inputs=[concatMask, outputsMasks], axis=-1)

        outputsImgs, outputsMasks = pConv2D(name, outputsImgs, outputsMasks, filters, kSize, stride=1, biasUse = biasUse)
        
        if biasUse:
            b = tf.get_variable("b",shape=[filters], initializer=tf.constant_initializer(0.))
            outputsImgs = tf.nn.bias_add(outputsImgs, b)
        
        if activationFunc:
            outputsImgs = keras.layers.LeakyReLU(0.2)(outputsImgs)

        if batchNorm:
            outputsImgs = tf.layers.batch_normalization(outputsImgs)
    
    return outputsImgs, outputsMasks


# ----- частичная свёртка
def pConv2D(name, inputs, masks, filters, kSize, stride, biasUse = True):
    
    
    padding = [[0, 0],[int((kSize - 1) / 2), int((kSize - 1) / 2)],[int((kSize - 1) / 2), int((kSize - 1) / 2)],[0, 0]]

    # дополняем маски и карты признаков нулями с помощью padding
    pdMasks = tf.pad(masks, padding, "CONSTANT")
    pdImages = tf.pad(inputs, padding, "CONSTANT")


    outputImages = tf.layers.conv2d(inputs = pdImages*pdMasks,filters=filters,kernel_size=kSize,strides=stride,use_bias=False,name='features')

    outputMasks = tf.layers.conv2d(inputs = pdMasks,filters=filters,kernel_size=kSize,strides=stride,kernel_initializer=tf.ones_initializer,use_bias=False,name='masks')

    maskRatio = (kSize*kSize*inputs.shape._dims[3]._value) / (outputMasks + 1e-5)
    outputMasks = tf.clip_by_value(outputMasks, 0.0, 1.0)
    maskRatio = maskRatio*outputMasks
    outputImages = maskRatio*outputImages
        
    return  outputImages, outputMasks


# ----- повышающая дискритизация
def upSampling(name, inputs, size):
    
    with tf.variable_scope(name):
        
        outputImages = keras.layers.UpSampling2D(size=size)(inputs)
        
    return  outputImages


# -------------------------------------------------------------------
# класс генератора
class GEN:
    
    def __init__(self, name):
        
        self.name = name
    
    def __call__(self, inputs, masks):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            # ----- блок кодировщика
            conv1, mask1 = encoderLayers('Pconv1', inputs, masks, filters = 64, kSize = 7, batchNorm = False)
            conv2, mask2 = encoderLayers('Pconv2', conv1, mask1, filters = 128, kSize = 5)
            conv3, mask3 = encoderLayers('Pconv3', conv2, mask2, filters = 256, kSize = 5)
            conv4, mask4 = encoderLayers('Pconv4', conv3, mask3, filters = 512, kSize = 3)
            conv5, mask5 = encoderLayers('Pconv5', conv4, mask4, filters = 512, kSize = 3)
            conv6, mask6 = encoderLayers('Pconv6', conv5, mask5, filters = 512, kSize = 3)
            conv7, mask7 = encoderLayers('Pconv7', conv6, mask6, filters = 512, kSize = 3)
            # ----- блок декодировщика (decoder)
            dconv1, dmask1 = decoderLayers('Pconv8', conv7, mask7, conv6, mask6, 512, kSize=3)
            dconv2, dmask2 = decoderLayers('Pconv9', dconv1, dmask1, conv5, mask5, 512, kSize=3)
            dconv3, dmask3 = decoderLayers('Pconv10', dconv2, dmask2, conv4, mask4, 512, kSize=3)
            dconv4, dmask4 = decoderLayers('Pconv11', dconv3, dmask3, conv3, mask3, 256, kSize=3)
            dconv5, dmask5 = decoderLayers('Pconv12', dconv4, dmask4, conv2, mask2, 128, kSize=3)
            dconv6, dmask6 = decoderLayers('Pconv13', dconv5, dmask5, conv1, mask1, 64, kSize=3)
            dconv7, dmask7 = decoderLayers('Pconv14', dconv6, dmask6, inputs, masks, 3, kSize=3, activationFunc = False, batchNorm = False)
            
            
            return keras.layers.Activation('sigmoid')(dconv7)

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


# -------------------------------------------------------------------
# реализация слоёв нейронной сети 

# ----- слой кодировщика
def encoderLayers(name, inputs, masks, filters, kSize, batchNorm = True, biasUse = True):
    
    with tf.variable_scope(name + 'en'):
    
        outputsImgs, outputsMasks = pConv2D(name, inputs, masks, filters, kSize, stride=2, biasUse = biasUse)
        
        if biasUse:
            b = tf.get_variable("b",shape=[filters], initializer=tf.constant_initializer(0.))
            outputsImgs = tf.nn.bias_add(outputsImgs, b)

        outputsImgs = keras.layers.ReLU()(outputsImgs)

        if batchNorm:
            outputsImgs = tf.layers.batch_normalization(outputsImgs)
    
    return outputsImgs, outputsMasks

# ----- слой декодировщика
def decoderLayers(name, inputs, masks, concatIm, concatMask, filters, kSize, activationFunc = True, batchNorm = True, biasUse = True):
    
    with tf.variable_scope(name + 'de'):
        
        outputsImgs = upSampling("up" + name, inputs, size=(2,2))
        outputsMasks = upSampling("up" + name + "Masks", masks, size=(2,2))

        outputsImgs = keras.layers.concatenate(inputs = [concatIm, outputsImgs], axis=-1)
        outputsMasks = keras.layers.concatenate(inputs=[concatMask, outputsMasks], axis=-1)

        outputsImgs, outputsMasks = pConv2D(name, outputsImgs, outputsMasks, filters, kSize, stride=1, biasUse = biasUse)
        
        if biasUse:
            b = tf.get_variable("b",shape=[filters], initializer=tf.constant_initializer(0.))
            outputsImgs = tf.nn.bias_add(outputsImgs, b)
        
        if activationFunc:
            outputsImgs = keras.layers.LeakyReLU(0.2)(outputsImgs)

        if batchNorm:
            outputsImgs = tf.layers.batch_normalization(outputsImgs)
    
    return outputsImgs, outputsMasks


# ----- частичная свёртка
def pConv2D(name, inputs, masks, filters, kSize, stride, biasUse = True):
    
    
    padding = [[0, 0],[int((kSize - 1) / 2), int((kSize - 1) / 2)],[int((kSize - 1) / 2), int((kSize - 1) / 2)],[0, 0]]

    # дополняем маски и карты признаков нулями с помощью padding
    pdMasks = tf.pad(masks, padding, "CONSTANT")
    pdImages = tf.pad(inputs, padding, "CONSTANT")


    outputImages = tf.layers.conv2d(inputs = pdImages*pdMasks,filters=filters,kernel_size=kSize,strides=stride,use_bias=False,name='features')

    outputMasks = tf.layers.conv2d(inputs = pdMasks,filters=filters,kernel_size=kSize,strides=stride,kernel_initializer=tf.ones_initializer,use_bias=False,name='masks')

    maskRatio = (kSize*kSize*inputs.shape._dims[3]._value) / (outputMasks + 1e-5)
    outputMasks = tf.clip_by_value(outputMasks, 0.0, 1.0)
    maskRatio = maskRatio*outputMasks
    outputImages = maskRatio*outputImages
        
    return  outputImages, outputMasks


# ----- повышающая дискритизация
def upSampling(name, inputs, size):
    
    with tf.variable_scope(name):
        
        outputImages = keras.layers.UpSampling2D(size=size)(inputs)
        
    return  outputImages


# -------------------------------------------------------------------
# класс генератора
class GEN:
    
    def __init__(self, name):
        
        self.name = name
    
    def __call__(self, inputs, masks):
        
        with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
            
            # ----- блок кодировщика
            conv1, mask1 = encoderLayers('Pconv1', inputs, masks, filters = 64, kSize = 7, batchNorm = False)
            conv2, mask2 = encoderLayers('Pconv2', conv1, mask1, filters = 128, kSize = 5)
            conv3, mask3 = encoderLayers('Pconv3', conv2, mask2, filters = 256, kSize = 5)
            conv4, mask4 = encoderLayers('Pconv4', conv3, mask3, filters = 512, kSize = 3)
            conv5, mask5 = encoderLayers('Pconv5', conv4, mask4, filters = 512, kSize = 3)
            conv6, mask6 = encoderLayers('Pconv6', conv5, mask5, filters = 512, kSize = 3)
            conv7, mask7 = encoderLayers('Pconv7', conv6, mask6, filters = 512, kSize = 3)
            # ----- блок декодировщика (decoder)
            dconv1, dmask1 = decoderLayers('Pconv8', conv7, mask7, conv6, mask6, 512, kSize=3)
            dconv2, dmask2 = decoderLayers('Pconv9', dconv1, dmask1, conv5, mask5, 512, kSize=3)
            dconv3, dmask3 = decoderLayers('Pconv10', dconv2, dmask2, conv4, mask4, 512, kSize=3)
            dconv4, dmask4 = decoderLayers('Pconv11', dconv3, dmask3, conv3, mask3, 256, kSize=3)
            dconv5, dmask5 = decoderLayers('Pconv12', dconv4, dmask4, conv2, mask2, 128, kSize=3)
            dconv6, dmask6 = decoderLayers('Pconv13', dconv5, dmask5, conv1, mask1, 64, kSize=3)
            dconv7, dmask7 = decoderLayers('Pconv14', dconv6, dmask6, inputs, masks, 3, kSize=3, activationFunc = False, batchNorm = False)
            
            
            return keras.layers.Activation('sigmoid')(dconv7)

    def get_var(self):
        return  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.name)


# класс, генерирующий тренировочные данные
class createAugment():
    
    # --
    # инициализация объекта класса
    def __init__(self, imgs, masks, batch_size=10, dim=(128, 128), n_channels=3):
        self.batch_size = batch_size  # размер батча
        self.images = imgs            # исходное изображение
        self.masks = masks            # маски изображений
        self.dim = dim                # размер изображения
        self.n_channels = n_channels  # количество каналов
        self.on_epoch_end()           # генерация набора батчей
    
    # --
    # результат: возможных батчей за эпоху
    def __len__(self):
        return int(np.floor(len(self.images) / self.batch_size))
    
    # --
    # результат: взятие батча с заданным номером (индексом)
    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        imageOrig, imageMasked, imageMasks = self.data_generation(indexes)
        return imageOrig, imageMasked, imageMasks
    
    # --
    # функция, повторяющаяся в конце каждой эпохи
    # результат: новая совокупность индексов изображений для очередного батча
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.images))
        np.random.shuffle(self.indexes)
    
    # --
    # результат: батч данных, включающий в себя 
    # маскированное изображение и часть изображения под маской
    def data_generation(self, idxs):
        
        imageMasked = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # маскированное изображения
        imageMasks = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # маски
        imageOrig = np.empty((self.batch_size, self.dim[0], self.dim[1], self.n_channels)) # изображение под маской

        for i, idx in enumerate(idxs):
            
            image, masked_image, image_masks = self.createMask(self.images[idx].copy())
            imageMasked[i,] = masked_image/255
            imageOrig[i,] = image/255
            imageMasks[i,] = image_masks/255
            
        return imageOrig, imageMasked, imageMasks
    
    # --
    # поворот изображения
    def imageRotate(self, img):
        
        angle = np.random.randint(1, 359)
        M = cv2.getRotationMatrix2D((self.dim[0]//2, self.dim[1]//2), 45, 1.0)
        rotated = cv2.warpAffine(img, M, self.dim)
        
        return rotated
    
    # --
    # уменьшение изначальной маски
    def imageResize(self, img):

        background = np.full((self.dim[0], self.dim[1], self.n_channels), 0, np.uint8)

        widthNew = np.random.randint(int(self.dim[0]/2), self.dim[0])
        heightNew = np.random.randint(int(self.dim[1]/2), self.dim[1])
        
        imgNew = cv2.resize(img, (widthNew, heightNew))

        xNew = np.random.randint(0, self.dim[0] - widthNew)
        yNew = np.random.randint(0, self.dim[1] - heightNew)

        background[yNew:yNew+heightNew,xNew:xNew+widthNew] = imgNew

        return background
    
    # --
    # добавляем дополнительные элементы
    def imageDetails(self, mask):
        
        # генерируем количество линий-повреждений на рисунке
        n_line = np.random.randint(1, 5)
        
        # рисуем линии
        for i in range(n_line):
            
            # генерируем первую точку линии
            x_start = np.random.randint(1, self.dim[0])
            y_start = np.random.randint(1, self.dim[1])
            
            # генерируем вторую точку линии
            x_finish = np.random.randint(1, self.dim[0])
            y_finish = np.random.randint(1, self.dim[1])
            
            # определяем толщину линии
            point = np.random.randint(1, 5)
            
            # рисуем линию между сгенерированными точками
            cv2.line(mask, (x_start, y_start), (x_finish, y_finish), (255,255,255), point)
        
        return mask
    
    # --
    # маскированного изображения и изображения под маской
    def createMask(self, image):
        
        randNumberOfMask = np.random.randint(0, len(self.masks)-1)
        isRotate = np.random.randint(1, 10)
        isResize = np.random.randint(1, 10)
        
        mask = (self.masks[randNumberOfMask].copy())
        
        if isResize%2 == 0:
            mask = self.imageResize(mask)
        
        if isRotate%2 == 0:
            mask = self.imageRotate(mask)
          
        
        mask = self.imageDetails(mask)
        
        mask2 = (mask//255)*255
        
        imageMasked = cv2.bitwise_and(image, cv2.bitwise_not(mask2)) + mask2
        
        return image, imageMasked, cv2.bitwise_not(mask2)

imagePaths = os.listdir(pathI)
masksPaths = os.listdir(pathM)
masksPathsTest = os.listdir(pathMT)
images = np.empty((pp, imageHeight, imageWidth, imageChannels), dtype='uint8')
masks = np.empty((len(masksPaths), imageHeight, imageWidth, imageChannels), dtype='uint8')
masksTest = np.empty((len(masksPathsTest), imageHeight, imageWidth, imageChannels), dtype='uint8')

imagePaths = imagePaths[20000+1:]

i = 0

for path in imagePaths:
    img = Image.open(os.path.join(pathI, path))
    img = img.resize((imageHeight,imageWidth))
    images[i] = tf.keras.preprocessing.image.img_to_array(img)
    i = i+1
    if i == pp:
        break

i = 0
for path in masksPaths:
    img = Image.open(os.path.join(pathM, path))
    img = img.resize((imageHeight,imageWidth))
    masks[i] = tf.keras.preprocessing.image.img_to_array(img)
    i = i+1  

i = 0    
for path in masksPathsTest:
    img = Image.open(os.path.join(pathMT, path))
    img = img.resize((imageHeight,imageWidth))
    masksTest[i] = tf.keras.preprocessing.image.img_to_array(img)
    i = i+1
    
trainData = createAugment(images[0:int(pp*0.9)], masks, batchSize, dim = [imageHeight, imageWidth])
testData = createAugment(images[int(pp*0.9):], masksTest, batchSize, dim = [imageHeight, imageWidth])

network = PCNN(trainData, testData, epochs, batchSize, imageHeight, imageWidth, imageChannels)

network.train(271)