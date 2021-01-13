import tensorflow as tf # 2.0.0
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from PIL import Image
import os
import json
import matplotlib.pyplot as plt
import argparse
import numpy as np
import time
import shutil

from coco import COCO
from train import train_step
from evaluate import evaluate
from model import mechAttention, CNN_Encoder, RNN_Decoder
from loss import loss_function
from utils import (load_image, save_InceptionV3_features,
                   calc_max_length, load_InceptionV3_features,
                   plot_loss, plot_attention)

EPOCHS = 10
BATCH_SIZE = 64
BUFFER_SIZE = 10000
EMBEDDING_DIM = 256
UNITS = 512

def main(COCO):
    parser = argparse.ArgumentParser(description="Image Caption")
    parser.add_argument('--annotations_dir', type=str, default="annotations/", help='annotations directory')
    parser.add_argument('--images_dir', type=str, default="train2014/")
    parser.add_argument('--cache_inception_features', type=bool, default=False)
    parser.add_argument('--train_examples', type=int, default=0) #riju
    parser.add_argument('--test_image', type=str, default="NA") #riju
    parser.add_argument('--rmv_data', type=bool, default=True)
    
    EPOCHS = 10
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    EMBEDDING_DIM = 256
    UNITS = 512
    
    

    args = parser.parse_args()
    
    # predict and plot the test image
    test_image_path = args.test_image
    delData = args.rmv_data
    test_image_YN = False
    if not test_image_path == "NA": 
        EPOCHS = 1
        test_image_YN = True
    # folders to load/save COCO data (annotations and images)
    annotation_folder = args.annotations_dir
    image_folder = args.images_dir

    # flag to download InceptionV3 features
    cache_inception_features = args.cache_inception_features

    print("----------------------------")
    print("STARTED IMAGE CAPTIONING PIPELINE")
    print("----------------------------")

    print()
    print("EPOCHS", EPOCHS)
    print("BATCH SIZE:", BATCH_SIZE)

    print()
    print("Available files in dir", os.listdir())

    print()
    print("Using GPU card:", tf.config.list_physical_devices('GPU'))

    print()
    print("Image and annotation folders")
    print(image_folder)
    print(annotation_folder)
    print("Cache image features")
    print(cache_inception_features)

    COCO_annotations = COCO.download_annotations()
    with open(COCO_annotations, "r") as f:
        annotations = json.load(f)

    # this returns path to images
    if not cache_inception_features:
        os.mkdir("train2014")
    COCO_PATH = COCO.download_images()
    print()
    print("COCO PATH:", COCO_PATH)
    
    # limit size of training (30,000 images)
    # store captions and image names
    all_captions = []
    all_img_names = []

    for annotation in annotations["annotations"]:
        # generate captions using start/end of sentence
        caption = "<start> " + annotation["caption"] + " <end>"
        image_id = annotation["image_id"]
        if cache_inception_features:
            full_coco_image_path = COCO_PATH + "COCO_train2014_" + "%012d.jpg" % (image_id)
        else:
            full_coco_image_path = COCO_PATH.replace("/train2014/","/extracted_v3Features/") + "COCO_train2014_" + "%012d.jpg" % (image_id)

        all_img_names.append(full_coco_image_path)
        all_captions.append(caption)

    # shuffle images/captions and set random state
    train_captions, train_img_names = shuffle(all_captions,
                                       all_img_names,
                                       random_state=19)

    print("**********************************")
    print("Resulting array with all captions ({})".format(len(all_captions)))
    all_captions

    # select first n captions from shuffled dataset
    #num_examples = 30000
    num_examples = args.train_examples
    if num_examples != 0: 
        train_captions = train_captions[:num_examples]
        train_img_names = train_img_names[:num_examples]

        print()
        print("num examples (total images):", num_examples)

    print("********************************")
    print("Size of downsampled training set")
    print("Captions: {}".format(len(train_captions)))
    
    # Process images using InceptionV3 model
    print()
    print("[-STATUS-] Instantiating InceptionV3 model")
    image_model = tf.keras.applications.InceptionV3(include_top=False, weights="imagenet")

    print()
    print("[-STATUS-] CNN network -STATUS-rmation")
    print("inputs of model:", image_model.input)
    print("shape of last hidden layer:", image_model.layers[-1].output)

    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output

    image_features_extract_model = tf.keras.Model(new_input, hidden_layer)

    # save feature vectors obtained with InceptionV3
    encode_train = sorted(set(train_img_names)) # get unique images (paths)
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(10)
   
    if cache_inception_features:
        
        # delete the original dataset to save space
        dir_name = "./extracted_v3Features/"
        
        if os.path.exists(dir_name):
            test = os.listdir(dir_name)
            for item in test:
                os.remove(os.path.join(dir_name, item))
            os.rmdir("extracted_v3Features")
        
        os.mkdir("extracted_v3Features")
        print()
        print("[-STATUS-] Caching InceptionV3 features")
        # this fetches 25950 features for the given initial 30,000
        for img, path in image_dataset:
            print("extracting image feature:", path)
            batch_features = image_features_extract_model(img)
            batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))

            for bf, p in zip(batch_features, path):
                print("### PATH ###")
                # print(p) #riju
                path_of_feature = p.numpy().decode("utf-8")
                # print(path_of_feature) #riju
                # print("### PATH AFTER REPLACE ###")
                path_of_feature = path_of_feature.replace("/train2014/","/extracted_v3Features/")
                print(path_of_feature)
                np.save(path_of_feature, bf.numpy())
                
        
    else:
        print()
        print("[-STATUS-] Fetching computed features from PC")
    print("Extraction of InceptionV3 features complete")
    
    all_img_names = []
    for i in train_img_names:
        print("### PATH OF TRAIN IMAGES ###")
        print(i)
        i = i.replace("/train2014/","/extracted_v3Features/")
        print(i)
        all_img_names.append(i)
    
    train_img_names = all_img_names
    #print(train_img_names)
    #exit()                                                 # uncom to proceed beyond generating inceptionV3 features 
    # pre-process and tokenize captions
    # tokenize by space, limit vocabulary up to 5000 words, create word-to-index and index-to-word mapping
    # pad all sequences
    # Choose the top 5000 words from the vocabulary
    print()
    print("[-STATUS-] Creating keras tokenizer")
    #top_k = 5000
    top_k = 20000
    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                      oov_token="<unk>",
                                                      filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    tokenizer.fit_on_texts(train_captions)
    train_seqs = tokenizer.texts_to_sequences(train_captions)
    
    # print("### TRAIN CAP ###")
    # print(train_captions)
    # print("### TRAIN SEQ ###")
    # print(train_seqs)

    # # compare train_seqs[0:6] and train_captions[0:6]
    #
    # pad each vector to the max length of captions
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, maxlen=calc_max_length(train_seqs), padding="post")

    # Calculates the max_length, which is used to store the attention weights
    max_length = calc_max_length(train_seqs)

    # Create training and validation sets using an 80-20 split
    img_name_train, img_name_val, cap_train, cap_val = train_test_split(train_img_names,
                                                                        cap_vector,
                                                                        test_size=0.2,
                                                                        random_state=0) 
    
    
    # ekhane paste kor
    # print("### IMAGE NAME VAL ###")
    # print(len(img_name_val))
    # print("### CAP TRAIN")
    # print(cap_train)
    
    
    if cache_inception_features:
        for img in img_name_val:
            shutil.copy(img.replace("/extracted_v3Features/","/train2014/"), './extracted_v3Features/')
    
    
    # delete the original dataset to save space
    if delData:
        dir_name = "./train2014/"
        test = os.listdir(dir_name)

        for item in test:
            if item.endswith(".jpg"):
                os.remove(os.path.join(dir_name, item))
        os.rmdir("train2014")
    
    
    # create tensor flow dataset for training
    vocab_size = top_k + 1
    num_steps = len(img_name_train) // BATCH_SIZE
    print("Vocabulary size:", vocab_size)
    print("Num steps:", num_steps)

    # Vector from InceptionV3 is (64, 2048)
    # these two variables represent that vector shape
    attention_features_shape = 64
    features_shape = 2048

    print()
    print("[-STATUS-] Creating tf dataset")
    dataset = tf.data.Dataset.from_tensor_slices((img_name_train, cap_train))

    # use load method to load the numpy files in parallel
    # this loads image and caption
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
                          load_InceptionV3_features, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # shuffle data
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # define encoder / decoder
    encoder = CNN_Encoder(EMBEDDING_DIM)
    decoder = RNN_Decoder(EMBEDDING_DIM, UNITS, vocab_size)

    # define optimizer and loss function
    optimizer = tf.keras.optimizers.Adam()
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy()

    # create checkpoint to resume/load model
    checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=encoder,
                               decoder=decoder,
                               optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
        start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
        # restoring the latest checkpoint in checkpoint_path
        ckpt.restore(ckpt_manager.latest_checkpoint)

    print()
    print("[-STATUS-] Started training model")

    # to avoid resetting the loss plot array
    loss_plot = []

    for epoch in range(start_epoch, EPOCHS):
        print()
        print("Epoch:", epoch + 1)
        start = time.time()
        total_loss = 0
        for (batch, (img_tensor, target)) in enumerate(dataset):
            #print(img_tensor)
            batch_loss, t_loss = train_step(img_tensor, target,
                                            encoder, decoder,
                                            tokenizer, optimizer)
            # print("batch", batch)
            # print("batch_loss", batch_loss)
            # print("epoch loss", t_loss)

            total_loss += t_loss
            if batch % 1 == 0:
                print ('Epoch {} Batch {} Loss {:.4f}'.format(
                  epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))
        # storing the epoch end loss value to plot later
        loss_plot.append(total_loss / num_steps)

        if epoch % 5 == 0:
            ckpt_manager.save()

        print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
        print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    # save loss plot
    plot_loss(loss_plot, save_plot=True)

    print()
    print("[-STATUS-] Testing Caption generation")

    #img_list=[] #riju
    
    # for img in img_name_val:
    #     img_list.append(img+'.npy')
    #     print(img+'.npy')
    
    #img_name_val=img_list
    
    # generate captions on the validation set
    for i in range(30):
        if test_image_YN:
            break
        # generate random id
        rid = np.random.randint(0, len(img_name_val))
        #print(img_name_val)
        # fetch corresponding image and caption from id
        image = img_name_val[rid]
        print("### VAL IMG PATH ###")
        print(image) #riju
        #image = image.replace("COCO_extracted_v3Features","COCO_train2014")
        real_caption = ' '.join([tokenizer.index_word[i] for i in cap_val[rid] if i not in [0]])

        # evaluate image (result is the caption)
        result, attention_plot = evaluate(image, max_length, attention_features_shape, encoder, decoder,
                                          load_image, image_features_extract_model, tokenizer)

        print()
        print("----------------------------")
        print("Image id:", i+1)
        print ("Real Caption:", real_caption)
        print ("Predicted Caption:", " ".join(result))
        plot_attention(image, result, attention_plot, save_plot=True, index_image=i)
        
    if test_image_YN:
        
        result, attention_plot = evaluate(test_image_path,max_length, attention_features_shape, encoder, decoder,
                                          load_image, image_features_extract_model, tokenizer)
        print ('Prediction Caption:', ' '.join(result))
        print(test_image_path)
        plot_attention(test_image_path, result, attention_plot, save_plot=True, index_image=108)

    print("---------------")
    print("|FINISHED MAIN|")
    print("---------------")

if __name__ == "__main__":
    main(COCO = COCO("/annotations/", '/train2014/'))
    print("Hello World!!!")
