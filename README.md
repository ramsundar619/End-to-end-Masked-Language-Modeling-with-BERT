# End to end Masked Language Modeling with BERT
## Introduction
This Illustration shows how to implement a Masked Language Model (MLM) with BERT and fine-tune it on the IMDB Reviews dataset.
## Table of Contents
- Introduction
- Installation
- Loading Dataset
- Getting Started
- Methodology being used for BERT using Keras and Tensorflow library
- Input
- Parameters used for libraries
- Output
## Installation
**Create a conda virtual environment and activate it**
```
conda create --name tf24
conda activate tf24
```
**Install Dependencies**
```
conda install tensorflow==2.4.1=gpu_py38h8a7d6ce_0
conda install pandas
conda install numpy==1.19.5
```
## Loading Dataset
```
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```
## Getting Started
**For Training**
```
python3 train.py
```
**For Testing**
```
python3 test.py
```
## Methodology being used for BERT using Keras and Tensorflow library
- pd.DataFrame() - Two-dimensional, size-mutable, potentially heterogeneous tabular data
- tf.strings.lower() - converts text to lower-case
- tf.strings.regex_replace() - Replace elements of input matching regex pattern with rewrite, Here input=lowercase, pattern="<br />", rewrite= " "
- tf.keras.layers.experimental.preprocessing.TextVectorization() - A preprocessing layer which maps text features to integer sequences.
- tf.keras.layers.experimental.preprocessing.TextVectorization.adapt() - call `adapt` on the text-only dataset to create the vocabulary. texts - input
- tf.keras.layers.experimental.preprocessing.TextVectorization.get_vocabulary() - Returns the current vocabulary of the layer. It contains word tokens, padding token ('') and OOV token ('UNK')
- np.random.rand() - Create an array of the given shape and populate it with random samples from a uniform distribution
- np.ones() - Return a new array of given shape and type, filled with ones
- np.random.randint() - Return random integers from low (inclusive) to high (exclusive)
- tf.data.Dataset.from_tensor_slices() - we can get the slices of an array in the form of objects - loading dataset
- tf.keras.layers.MultiHeadAttention() - This layer first projects query, key and value, the query and key tensors are dot-producted and scaled. These are softmaxed to obtain attention probabilities. The value tensors are then interpolated by these probabilities, then concatenated back to a single tensor
- tf.keras.sequential() - regular densely-connected NN layer
- tf.keras.layers.Dropout() - randomly sets input units to 0 with a frequency of rate at each step during training time
- tf.keras.layers.LayerNormalization() - Normalize the activations of the previous layer for each given example in a batch independently
- tf.keras.losses.SparseCategoricalCrossentropy - Computes the crossentropy loss between the labels and predictions
- tf.keras.losses.Reduction.NONE - No additional reduction is applied to the output of the wrapped loss function
- tf.keras.metrics.Mean() - Computes the (weighted) mean of the given values
- tf.keras.Model() - Model groups layers into an object with training and inference features
- tf.keras.layers.Input - Input layer
- tf.keras.layers.Embedding - Turns positive integers (indexes) into dense vectors of fixed size
- tf.range - Creates a sequence of numbers
- tf.keras.layers.Dense - Fully connected layer
- tf.keras.Model.compile() - Configures the model for training.
  - Parameters used
  - Optimizer - An optimizer is one of the two arguments required for compiling a Keras model.
- tf.keras.optimizers.Adam - Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and second-order                                    moments.
- tf.keras.model.get_layer() - Retrieves a layer based on either its name (unique) or index.
- tf.saved_model.save() - Exports a tf.Module (and subclasses) obj to SavedModel format.
- tf.keras.models.load_model() - Loads the saved model.
- tf.keras.layers.MaxPooling1D() - Max pooling operation for 1D spatial data.
## Input
Here we are using [aclIMDB Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz) as an input to the model for finetuning once it's self-supervised with mlm modelling
## Parameters used
- MAX_LEN = 256       - Fixed length of each input sample in tokens after tokenization
- BATCH_SIZE = 32     - Batch size
- LR = 0.001          - Learning Rate
- VOCAB_SIZE = 30000  - Maximum number of words in vocabulary - Fixed size of vocabulary for bert
- EMBED_DIM = 128     - Word embedding vector size
- NUM_HEAD = 8        - Number of attention heads (BERT)
- FF_DIM = 128        - Feedforward dimension (BERT)
- NUM_LAYERS = 1      - Number of BERT module layers
- buffer_size=tf.data.AUTOTUNE - This allows later elements to be prepared while the current element is being processed.
- filters - The dimensionality of the output space (i.e. the number of output filters in the convolution).
- kernel_size - An integer or tuple/list of 2 integers, specifying the height and width of the 2D convolution window.
- strides - An integer or tuple/list of 2 integers, specifying the strides of the convolution along the height and width. 
- padding - one of "valid" or "same" (case-insensitive). "valid" means no padding. "same" results in padding with zeros evenly to the left/right or up/down of the input. When padding="same" and strides=1, the output has the same size as the input. 
- "relu" - Rectified Linear Unit Function which returns element-wise max(x, 0).
- "softmax" - Softmax converts a vector of values to a probability distribution.The elements of the output vector are in range (0, 1) and sum to 1.
- rate - Float between 0 and 1. Fraction of the input units to drop. 
## Ouput
After fine-tuning, the model shows a test accuracy of 84%
