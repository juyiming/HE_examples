# Visualization examples
Visualization examples for hierarchical explanations of HE_loo and HE_lime on SST-2 and MNLI datasets.

# Implementation
The implementation of HE_loo and HE_lime.

The implementations are based on the Huggingface’s transformer model (https://github.com/huggingface/transformers) and the official code repository of LIME (https://github.com/marcotcr/lime).

Among them, the lime-text.py and lime_base in the LIME library have been rewritten. After installing LIME, they need to be replaced with rewritten files.

# Visualization
In visualization.py, the prediction labels of examples to be visualized need to be loaded as the variable: labels.
