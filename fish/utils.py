def setup_env():
  import os, sys, boto3
  import pandas as pd
  from botocore import UNSIGNED
  from botocore.client import Config  
 
  print("Getting required libraries...")
  os.system("rm -rf sample_data")
  os.system("git clone https://github.com/fizyr/keras-retinanet.git")
  os.system("pip install keras-retinanet")
  os.system("pip install keras-resnet")
  sys.path.append("keras-retinanet")
  
  s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
  
  print("Downloading model from S3...")
  s3.download_file("fisheries-deep-learning-models", "resnet50_csv_converted_08.h5", "weights.h5")
    
  print("Downloading meta data from S3...")
  s3.download_file("fisheries-deep-learning-models", "classes.csv", "classes.csv")

  print("Downloads complete!")  


def load_data():
  """Helper function to load RetinaNet and class info"""
  
  import pandas as pd
  from keras_retinanet import models
  from IPython.display import clear_output
  
  # get model
  model = models.load_model("weights.h5", backbone_name='resnet50')

  # get classes info
  classes = pd.read_csv(f"classes.csv", header=None)
  classes = {v:k for k,v in zip(classes.loc[:,0].values, classes.loc[:,1].values)}
  
  clear_output()
  return model, classes


def process_image():
         
  import os, cv2
  import numpy as np
  import matplotlib.pyplot as plt

  from google.colab import files
  from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
  from keras_retinanet.utils.visualization import draw_box
  from IPython.display import Image, display

  # save uploads to storage
  uploaded = files.upload()
  if len(uploaded) == 0:
    return
  else:
    for file_name, v in uploaded.items():
      break

  # load image
  image = read_image_bgr(file_name)
  draw = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)

  # preprocess image  
  X = preprocess_image(image)
  X, scale = resize_image(X)
  X = np.expand_dims(X, axis=0)

  # make predictions
  print("Making predictions...")
  boxes, scores, labels = model.predict_on_batch(X)
  boxes, scores, labels = boxes[0], scores[0], labels[0]
  boxes, scores, labels = boxes[labels>0], scores[labels>0], labels[labels>0]
  boxes /= scale    

  if len(labels) == 0:
    print("\nTop predictions: No fish detected")

  else:  

    # print top predictions
    print("\nTop predictions:")
    count = 1
    seen_list = []
    for i in range(min(10, len(labels))):
      box, score, label = boxes[i], scores[i], labels[i]
      if label not in seen_list:
        print(f"{count}. {classes[label]} - {100*score:>0.1f}%")
        seen_list.append(label)
        count += 1

    # get top prediction
    box, score, label = boxes[0], scores[0], labels[0]

    # draw box
    draw_box(draw, box.astype(int), color=[0, 0, 1])

  # plot image
  print("\nDisplaying Image...")
  fig = plt.figure()
  fig.patch.set_facecolor('xkcd:mint green')  
  plt.imshow(draw)
  plt.axis('off')
  plt.savefig(file_name, dpi=1000, pad_inches=0.0)
  plt.close()
  display(Image(file_name, height=X.shape[1]+100, width=X.shape[2]))
  os.remove(file_name)