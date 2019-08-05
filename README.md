
# [Google Colab Fish Detection App](https://colab.research.google.com/drive/1kdHLEmP8eYzRhJBCEqR9jSg7qk6E5iwn#scrollTo=n1eRvv6d16s-)

## Example Code
Run the following cells in a Google Colab notebook.

### 1. Clone this repo
```
!git clone https://github.com/mattmotoki/fish-ml-app-utils.git
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
