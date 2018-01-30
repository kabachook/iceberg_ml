# Statoil Iceberg Classification challenge

## My solution

It should be noted that there is a data leak in `inc_angle` feature

So inc_angle's with identical values (up to 4 decimal digits) had the same label most of the time

![Angle plot](https://github.com/kabachook/iceberg_ml/raw/master/angle_plot.jpg)

Also we should avoid extreme values as `log_loss` penalty for these values is high

The solution to that is to clip values from 0.001 to 0.01 and from 0.999 to 0.99

Leak usage is implemented in `leak.py`

As for the model, I used best CNN that I tested, which I discovered using `ice.py` file

My model:
```python

    model=Sequential()
    
    # Using 3xConv in a row to reduce load 3x(3,3) = 1x(7,7)

    # Conv block 1
    model.add(Conv2D(64, kernel_size=(3, 3),activation='relu', input_shape=(75, 75, 3)))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
   
    # Conv block 2
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu' ))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Conv block 3
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    #Conv block 4
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
   
    # Flatten before dense
    model.add(Flatten())

    #Dense 1
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.4))

    #Dense 2
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))

    # Output 
    model.add(Dense(1, activation="sigmoid"))

    optimizer = Adam(lr=0.0001, decay=0.0)
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    
    return model

```

Data is normalized and augmented(flip,flop)

Ensembling using K-fold is a good idea, so in final submission 10-fold is used

After getting submission file, it is processed using `leak.py` script, which is mentioned earlier

My stats:

![Stats](https://github.com/kabachook/iceberg_ml/raw/master/stats.png)

Complete log of CNN training available in `ice.ipynb`

## Possible improvements

* Use different augmentations(shift, rotations)
* XGBoost or LightGBM adding additional metadata
* Pseudolabeling
* KNN/SVM after CNN
* Use diffrerent models for icebergs in group1 and group2 (Plot 1)
* Train many weak CNNs and ensemble, i.e. top100 of them
* Use pretrained VGG16, ResNet, etc.
* Use diffrerent data preparation, i.e. FFT or sqrmean
