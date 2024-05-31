# Netformer Library

Our library is designed to facilitate research in the field of Neural Architecture Search (NAS).

## Features

### Random Model Generator
To create a dataset of models, we developed a configurable random model generator. These models are trained on random regression tasks.

### Model Tokenization
To train a transformer on these models, we tokenized the trained models.

### Mixed Loss Based on Neuron Sensitivity

During training, we use a mixed loss function based on Neuron Sensitivity. The concept of Neuron Sensitivity is introduced in the paper "[SeReNe: Sensitivity based Regularization of
Neurons for Structured Sparsity in Neural Networks](https://arxiv.org/pdf/2102.03773)".


Our combined loss function aims to effectively guide the training process by considering multiple aspects of the generated models. It consists of two parts:

1. Vertex Type and Number Prediction (CrossEntropy Loss)
2. Edge Weight Prediction (Mean Squared Error with Neuron Sensitivity)

**Vertex Type and Number Prediction (CrossEntropy Loss)**
In this part, we focus on predicting the type of vertex, its number, and its parent's number. We use CrossEntropyLoss for this purpose, as it is well-suited for multiclass classification tasks.

For each vertex, we predict its type (e.g., Linear, Tanh, ReLU), its number within the graph, and the number of its parent within the graph structure. By using CrossEntropyLoss, we aim to minimize the discrepancy between the predicted and true values for these attributes.

**Edge Weight Prediction** (Mean Squared Error with Neuron Sensitivity)
In this part, we focus on predicting the weights of the edges between vertices. We use Mean Squared Error (MSE) as the loss function for this regression task. However, to improve the robustness of the training process and to better handle outliers among the predicted models, we incorporate the concept of neuron sensitivity.

Neuron sensitivity provides insight into the importance of each neuron in the model's architecture. We multiply the MSE loss by the sensitivity of each neuron before summing them up. This effectively gives more weight to the edges connected to neurons that are deemed more sensitive, thereby emphasizing their impact on the overall architecture.

### Metrics

We use various metrics to evaluate the generated models:

1. **Cosine Distance**: We calculate the cosine distance between the generated model and the initial model, excluding the last layer. This helps in understanding how close the generated architecture is to the initial one in terms of structure and functionality.
2. **Edge Prediction Accuracy**: This metric evaluates how accurately the edges (connections between neurons) of the generated model match the edges of the initial model.

### Meta-Feature Concatenation

#### Selected Meta-Features
We carefully select the following meta-features from the [pymse](https://pymse.readthedocs.io/en/latest/) library, considering them to be highly significant for describing the model:

- **inst_to_attr**: Represents the ratio of instances to attributes.
- **nr_class**: Indicates the total number of classes present in the model.
- **nr_attr**: Denotes the overall number of attributes in the model.
- **attr_to_inst**: Reflects the ratio of attributes to instances.
- **skewness**: Measures the skewness of the model's data distribution.
- **kurtosis**: Quantifies the kurtosis of the model's data distribution.
- **cor**: Illustrates the correlation between different attributes of the model.
- **cov**: Represents the covariance between attributes of the model.
- **attr_conc**: Indicates the concentration of attributes within the model.
- **class_conc**: Highlights the concentration of classes within the model.
- **sparsity**: Describes the sparsity level of the model.
- **gravity**: Represents the gravity measure of the model.
- **class_ent**: Reflects the entropy of classes within the model.
- **attr_ent**: Denotes the entropy of attributes within the model.
- **mut_inf**: Measures the mutual information of the model.
- **eq_num_attr**: Indicates if the model has an equal number of attributes.
- **ns_ratio**: Represents the ratio of neuron sensitivity within the model.
- **f1, f2**: Functions 1 and 2 that describe specific characteristics of the model.
- **tree_depth**: Represents the depth of the model's tree structure.
- **leaves_branch**: Indicates the number of leaves per branch in the model.
- **nodes_per_attr**: Denotes the number of nodes per attribute in the model.
- **leaves_per_class**: Indicates the number of leaves per class in the model.

By augmenting each token of the model with these meta-features, we enhance the model's understanding of its structure, characteristics, and properties. This augmentation contributes to improving the model's ability to comprehend and process data, leading to more accurate and high-quality predictions.

You can get it by simply calling `meta_features = get_meta_features(parameters, target)` with the appropriate parameters and target arrays for each dataset.

## Training Process

1. **Encoder**: The transformer's encoder processes the tokenized models to produce embeddings.
2. **Embedding**: These embeddings are concatenated with meta-features.
3. **Variational Part**: The concatenated embeddings are passed through the variational part of the transformer.
4. **Decoder**: The decoder translates the variational embeddings back into tokens.
5. **Reverse Conversion**: Tokens are converted back into models and evaluated based on the predicted edges metric.



## Inference

We take an embedding of the normal distribution. Then we run it through the transformer and get the prediction.

## Installation

Provide steps to install your library, including any dependencies.

```bash
pip install -r requirements.txt
```

## Run example
```bash
bash lib/generator/install_datasets.sh
cp example/* .
python train_loop_example.py
python inference_example.py
```


