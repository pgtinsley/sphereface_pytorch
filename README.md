## FOR OGE

Clone this repository somewhere to your account on hutchentoot.

Change directories with 'cd' into sphereface\_pytorch.

Create a conda environment on hutchentoot using the following command:
```
conda env create -f sphereface_pytorch.yml
```

That should probably take a bit of time to set up.

After that, make sure all of your images are in one folder. They can have the extensions: .png, .jpg, .jpeg.
We can probably change that to add more, but that's what supported as of now.

Once those are all in the same folder, you can run the following command to detect faces, extract features, and put them in a pkl file.

```python
python oge.py --dir <path/to/folder> --embeddings <path/to/pkl/file>
#OR
python oge.py -d <path/to/folder> -e <path/to/pkl/file>

#Example
python oge.py --dir ./data/test --embeddings test.pkl
python oge.py -d ./data/test -e test.pkl
```

The long part will be the beginning, which detects the faces and saves them to 160x160 images.
The next part is quick, and resizes the 160x160 images to 96x112 images.
The last part is feature extraction, which has actually been rather quick the past couple of times I've tried it... which is nice!

If you have any problems running this, lmk! 

Best,
PT



# SphereFace
A PyTorch Implementation of SphereFace.
The code can be trained on CASIA-Webface and the best accuracy on LFW is **99.22%**.

[SphereFace: Deep Hypersphere Embedding for Face Recognition](https://arxiv.org/abs/1704.08063)

# Train
```
python train.py
```

# Test
```
# lfw.tgz to lfw.zip
tar zxf lfw.tgz; cd lfw; zip -r ../lfw.zip *; cd ..

# lfw evaluation
python lfw_eval.py --model model/sphere20a_20171020.pth
```

# Pre-trained models
| Model name      | LFW accuracy | Training dataset |
|-----------------|--------------|------------------|
| [20171020](model/sphere20a_20171020.7z) | 0.9922 | CASIA-WebFace |

# φ
![equation](https://latex.codecogs.com/gif.latex?phi%28x%29%3D%5Cleft%28-1%5Cright%29%5Ek%5Ccdot%20%5Ccos%20%5Cleft%28x%5Cright%29-2%5Ccdot%20k)

![equation](https://latex.codecogs.com/gif.latex?myphi(x)=1-\frac{x^2}{2!}+\frac{x^4}{4!}-\frac{x^6}{6!}+\frac{x^8}{8!}-\frac{x^9}{9!})

![phi](images/phi.png)

# References
[sphereface](https://github.com/wy1iu/sphereface)
