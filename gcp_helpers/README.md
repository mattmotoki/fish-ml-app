
# [Google Colab Fish Detection App](https://colab.research.google.com/drive/1kdHLEmP8eYzRhJBCEqR9jSg7qk6E5iwn#scrollTo=n1eRvv6d16s-)

## Example Code
Run the following cells in a Google Colab notebook.

### 1. Clone this repo
```
import os
os.system("rm -rf fish-ml-app-utils")
os.system("git clone https://github.com/mattmotoki/fish-ml-app.git")

import sys
sys.path.append("fish-ml-app-utils")
from fish.utils import setup_env, load_data, process_image
```

### 2. Setup the environment 
```
setup_env()
```

### 3. Load the model and class information from S3
```
model, classes = load_data()
```

### 4. Ask the user to upload an image, then process it
```
process_image()
```
