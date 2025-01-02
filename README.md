## Description

Stegonagraphy is a technique that hides information in images through slight manipulations of pixel values. This project uses convolutional neural networks (CNNS) to detect if steganography is present in an image, and if so, which technique was used. The model can detect four kinds of steganography: Least Significant Bit (LSB), Pixel Value Differencing (PVD), Fast Fourier Transform (FFT), and Discrete Cosine Transform (DCT).

LSB (Least Significant Bit) is the simplest technique of steganography where data is hidden in the least significant bit of each pixel value. PVD (Pixel Value Difference) is achieved by determining the amount of information that can be hidden based on the difference in value of a pixel and its adjacent pixels. The FFT technique uses the Fast Fourier Transform algorithm to transform the image from the spatial domain to the frequency domain. Then, information can be embedded in the frequency domain and then turned back into an RGB image. The DCT (Discrete Cosine Transform) technique is similar to FFT but uses exclusively cosine waves whereas FFT uses both sine and cosine waves.

![LSB Visualizer](https://arobs.com/wp-content/uploads/2023/07/Least-Significant-Bit-Steganography.jpg.webp)

We chose to use EfficientNetV2 as our model since it is a powerful and lightweight model well suited to classifying images. The dataset we used was a combination of different Kaggle datasets focusing on images stegonagraphically altered LSB, PVD, FFT, and DCT, as well as images with no steganographic alterations.

Accuracy Results Based on Dataset Trained On
| Dataset |  DCT  |  FFT  |  LSB  |  PVD  | All Steganography |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| Accuracy  | 87.6%  | 87.1% | 98.5% |  98.75% | 66.29% |

F1 Score From Individual Dataset Training
| Dataset |  DCT  |  FFT  |  LSB  |  PVD  |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| Steganography Class F1 | 0.72 | 0.75 | 0.99 |  0.99 |
| Clean Class F1 | 0.46  | 0.91 | 0.99 | 0.99 |

F1 Score From Cumulative Dataset Training
| Class |  DCT  |  FFT  |  LSB  |  PVD  | Clean |
|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
| F1  | 0.67  | 0.58  | 0.58  |  0.5  | 0.93  |



## Poetry Instructions
To set up poetry locally run
```bash
poetry install
```
To add packages run
```bash
poetry add numpy
```
To run a file run
```bash
poetry run python main.py
```

## Datasets
- https://www.kaggle.com/datasets/petrdufek/stego-pvd-dataset
- https://www.kaggle.com/datasets/diegozanchett/digital-steganography
- https://www.kaggle.com/datasets/marcozuppelli/stegoimagesdataset
- https://www.kaggle.com/datasets/utkarshsaxenadn/landscape-recognition-image-dataset-12k-images
