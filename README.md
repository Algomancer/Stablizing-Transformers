# Stablizing Transformers

This project offers an open-source implimentation of [σReparam](https://arxiv.org/pdf/2303.06296.pdf), a technique designed to enhance Transformer training stability by addressing attention entropy collapse. The approach reparametrizes linear layers with spectral normalization and a learned scalar, showcasing improved robustness across various tasks without the need for specific hyperparameters without warmup and adaptive optimizers.

σReparam + pre-LN 💯


### Summary
Tasks: Image classification, self-supervised learning, machine translation, automatic speech recognition, and language modeling.

Paper Findings:

1. Observed entropy collapse in various baseline models.
2. Image Classification: σReparam simplifies ViT training, removing several components, and reduces training time by 16%.
3. Self-Supervised Learning: σReparam enhances SimCLR training stability and robustness.
4. Machine Translation: σReparam stabilizes deep post-LN architectures (up to 100L-100L encoder-decoder layers).
5. Speech Recognition: σReparam stabilizes post-LN Transformer training, simplifying the recipe.
6. Language Modeling: σReparam works well with causal Transformer architectures, matching state-of-the-art performance without post-LN.
