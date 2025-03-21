# Recommendation System by VAE
This project was for CSE258 Recommender Systems Final Assignment. My recommender system if for [Steam Video Game and Bundle Data](https://cseweb.ucsd.edu/~jmcauley/datasets.html#steam_data) It employs Denoised Variational Auto Encoder. The input and output of the DVAE are based on user-item interaction matrix. In the training phase, some of 1-elements in input matrix are masked as 0. The VAE is trained to reproduce the denoised (demasked) matrix. In the inference phase, it takes a user-item combination and looks up them in the denoised matrix. In "vae_genre_toprec.py" the recommender determine if the input user item combination is in the top n% of recommendations of the user.

# Requirement
- scikit-learn==1.5.2
- torch==2.5.1
