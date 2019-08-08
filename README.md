# cnn-facial-landmark

Facial landmark detection based on convolution neural network.

The model is build with TensorFlow, the training code is provided so you can train your own model with your own datasets.

A sample gif extracted from video file showing the detection result.

![](https://github.com/yinguobing/cnn-facial-landmark/blob/master/demo01.gif)

This is the companion code for the tutorial on deep learning [here](https://yinguobing.com/facial-landmark-localization-by-deep-learning-background/), which includes background, dataset, preprocessing, model architecture, training and deployment. I tried my best to make them simple and easy to understand for beginners. Feel free to open issue when you are stuck or have some wonderful ideas to share.


## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

TensorFlow

```bash
# For CPU
python3 -m pip install tensorflow

# or, if you have a CUDA compatible GPU
python3 -m pip install tensorflow-gpu

```

### Installing

Just git clone this repo and you are good to go.

```bash
# From your favorite development directory
git clone https://github.com/yinguobing/cnn-facial-landmark.git
```

## Train, evaluate and export.

Before training started, make sure the following requirements are met.
- training and evaluation tf-record file.
- a directory to store the check point files.
- a directory to export the trained model.
- hyper parameters like training steps, batch size, number of epochs.

The following example shows how to train the model for 500 steps, evaluate it and export it in a `saved_model` format.

```bash
# From the repo's root directory
python3 landmark.py \
    --train_record train.record \
    --val_record validation.record \
    --model_dir train \
    --export_dir saved_model \
    --train_steps 500 \
    --batch_size 32
```

## Contributing

Please read [CONTRIBUTING.md]() for details on our code of conduct, and the process for submitting pull requests to us.


## Authors

Yin Guobing (尹国冰) - [yinguobing](https://github.com/yinguobing/)


## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The TensorFlow official tutorial.

## Changelog

### Update 2019-08-08
A new input function is added to export the model to take raw tensor input. Use the `--raw_input` argument in the exporting command. This is useful if you want to "freeze" the model later.

For those who are interested in inference with frozen model on image/video/webcam, there is a lightweight module here:https://github.com/yinguobing/butterfly, check it out.


### Update 2019-06-24
Good news! The code is updated. Issue #11 #13 #38 #45 and many others have been resolved. No more `key error x` in training, and exporting model looks fine now.

### Update 2019-05-22
Thanks for your patience. I have managed to updated the repo that is used to extract face annotations and generate TFRecord file. Some bugs have been fixed and some minimal sample files have been added. Check it out [here](https://github.com/yinguobing/image_utility) and [here](https://github.com/yinguobing/tfrecord_utility).

The training part(this repo) is about to be updated. I'm working on it.

### Update 2019-04-22
This repository now has 199 github stars that is totally beyond my expectation. Whoever you are, wherever you are from and whichever language you speak, I want to say "Thank you!" to you 199 github friends for your interest.

Human facial landmark detection is easy to get hands on but also hard enough to demonstrates the power of deep neural networks, that is the reason I chose for my learning project. Even I had tried my best to keep a exhaustive record that turned into this repository and the companion tutorial, they are still sloppy and confusing in some parts.

The code is published a year ago and during this time a lot things have changed. TensorFlow 2.0 is coming and the exported model seems not working in the latest release of tf1.13. I think it's better to make this project up to date and keep being beneficial to the community.

I've got a full time job which costs nearly 12 hours(including traffic time) in my daily life, but I will try my best to keep the pace.

Feel free to open issues so that we can discuss in detail.