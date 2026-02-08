# shapley_masking

Is the new BPNet-lite working as well as the BPNet from original paper
1. I suspect that BPNet-lite is working better because:
    - BPNet model from paper only uses 1,000 bp input to predict 1,000 bp output; BPNet-lite model uses 2,114 bp input to predict 1,000 bp output; longer sequence context results in better performance near the edge of the of the 1,000 bp output
    - [not so sure about this reason]: BPNet model from paper does not sample negative peaks while BPNet-lite model samples negatives; negative examples might provide more training data
2. To provide evidence for this, we will can just evaluate both models on a test set. We used the same test chromosomes. To make in favor of BPNet model from paper we will use their test set from the paper [What is their test set]
3. Create code to replicate their metrics on their test set [maybe pearson residual or correlation; this will confirm we are using their test set]
4. see if loss is lower for BPNet-lite model than BPNet model using BPNet model test set [make sure that the count scaling hyperparameter in the loss is the same for both models]
