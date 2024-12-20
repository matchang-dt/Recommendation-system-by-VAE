# CSE258 Recommender Systems Final Assignment
My recommender system employs Denoised Variational Auto Encoder. The input and output are based on user-item interaction matrix. In training phase, some of 1-elements in input matrix are masked as 0. The VAE is trained to reproduce the denoised matrix.

# Requirement
- scikit-learn==1.5.2
- torch==2.5.1
