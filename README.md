# ResearchLog

The following is a log of research done under the Cat's grant 2024.

## 01.07.2024 - 15.07.2024

In the past weeks I've developed an approach to build prototypes with arbitrarily flexible parts. The approach looked promising on CIFAR10. I've done initial experiments on the [CUB_200_2011](https://data.caltech.edu/records/65de6-vp158) dataset (fine-grained bird classification) with ResNet34 taken as the encoder network. The initial accuracy 75% is worse than the baseline 82% accuracy of fine-tuned ResNet and then the 80% accuracy of the original ProtoPNet but this is just a starting point (I trained with just 20 epochs and with a single augmentation - RandomHorizontalFlip)

The original paper (and many following papers) use a fixed augmentation set which is 30x bigger then the base training dataset. The augmentations consist of small rotations, small zoom-ins and random horizontal flips. I have an intuition that those augmentations are redundant and the only relevant one is the horizontal flip. My experiments on ResNet34 confirm this intuition. The full augmentation set might help to boost the metrics but for rapid prototyping RandomHorizontalFlip seems good enough.

The former papers on ProtoPNets use ResNets (among others) as encoders. But ResNets have [large receptive fields](https://gist.github.com/samson-wang/a6073c18f2adf16e0ab5fb95b53db3e6). This means that the 7x7 feature map produced by ResNets can hardly be interpreted as a map of local features - it's more likely a map of different views of the entire image. In that sense the ResNet should rather be interpreted as an ensemble of 49 different classifiers. I think that the [BagNet](https://arxiv.org/abs/1904.00760) is a way better candidate for the encoder network for ProtoPNets as its features are designed to have small receptive fields. Actually there is [a recent paper](https://arxiv.org/abs/2406.15168) that does just that but on a different dataset (apparently simpler). My initial experiments (10 epochs, RandomHorizontalFlip): baseline fine-tuned BagNet achieves 78% accuracy on birds and my version of the prototype network - 70%.

## 15.07.2024 - 22.07.2024

This week's work has been primarily conceptual and came from reflecting upon the results of the experiments done earlier. I will describe the ideas shortly.

**The general remark** - there is a strong analogy between prototypes and convolutional/linear layers. In fact the output neurons of a linear layer can be treated as prototypes (with dot product as the similarity function). Similarly the output neurons of a convolutional layer (followed by AdaptiveMaxPool2d(1, 1)). In fact, the prototypical layer is implemented as a convolutional layer with some tweaks (different similarity function). In my mind the end goal of the prototypes is simply a convolutional network but with some architectural improvements which better utilise the spatial bias and improve interpretability.

That being said, imagine a prototype (or convolutional kernel) consisting of several parts - for example 9 parts. The presence of each part in the image can be measured in a standard way (by computing some kind of similarity function with every location of the image and then max pooling). The presence of the prototype (and the corresponding class) is measured by some form of integration of the scores of its parts (e.g. weighted sum or soft-max pooling). So there is a bird on the image if there is a wing, a beak, a feather and so on. But what should be the relations between the parts?

Even though it's tempting to assume no relation during the evaluation phase, the lack of inter-part constraints seem to hinder the learning process. For example the unrestricted parts tend to cluster around the most distinctive part (e.g. a beak). The generic solutions like dissimilarity loss have been tried in the literature but my belief is this is not enough. Imo the parts should be spatially constrained, e.g. the 9 parts should be arranged in a 3x3 grid. But this grid is too rigid. The answer to this seems to be [Deformable Prototypes](https://arxiv.org/abs/2111.15000) or [Deformable Convolutions](https://arxiv.org/abs/1703.06211) but they are based on the offset field network which kills the entire idea of spatial structure adaptability of parts.

My initial idea was to try to define a general directions within the prototype, e.g. the idea of the center part, the top-left part, top-right part and so on. The top-left part should be approximately above and to the left of the center part and should be matched with the best matching region to the top-left of the best matching region of the center part. And so on.

However this looks like a standard MaxPool2d operation. This means that this mild form of directional invariance is already implemented in standard convolutional networks. But the convolutional filters consider the entire input channels in all of the spatial locations, which allows for inter-channel information mixing but also drastically lowers interpretability. My idea is therefore to put an interesting constraint to the convolutional weight - require the convolutional kernel to attend to only one input channel per spatial location. This would allow for easy interpretation of a filter - the filter is active if this is (approximately) in the center, that is (approximately) to the top-left, that2 is (approximately) to the top-right and so on. The interpretation could be then broadcasted in top-down fashion to the input space.

I can use the Gumbel-Softmax trick to force the convolution to attend to only one input channel per spatial location. This looks quite challenging implementation-wise and I'd like to spend the next week trying to do just that and see what the results are. Specifically I'll focus on the accuracy and whether it drops significantly. As the number of network parameters will drop dramatically (at least order of magnitude) the accuracy should drop. However this constraint arguably can introduce a helpful bias (standard conv-nets might be stuck in local minima due to over-parameterisation) and the idea that the performance will increase is not unreasonable.

## 22.07.2024 - 29.07.2024

I've implemented and tested the Gumbel-softmax version of the convolutional layer (GumbelConv2d) - every output feature selects just one channel per spatial dimension. I tested it on CIFAR10 for some simple architectures. Apart from looking at clean accuracy I visualized the input-space gradients towards labels. I hoped to see more visually aligned gradients for GumbelConv2d. Observations:

- it's non-trivial to select the scale of the logits used to compute the gumbels; also it's non-trivial to select proper tau parameter (see [F.gumbel_softmax](https://pytorch.org/docs/stable/generated/torch.nn.functional.gumbel_softmax.html). More experiments are needed to make sure I've chosen the right ones
- contrary to what I thought GumbelConv2d has more parameters then the corresponding Convd2 as it needs to parameterise the distribution along the input channels
- the performance is worse both in terms of accuracy and interpretability - the filters learned by network with GumbelConv2d are way less expressive than the standard Convd2; also the input-level gradients are less aligned
- similarly to depth-wise separable convolutions I added intermediate 1x1 convolutions to mix the channels. The performance came closer to that of standard Conv2d and showed less overfitting than the Conv2d with such 1x1 layers. GumbelConv2d with 1x1 might actually be equivalent to Conv2d when I think of it now.

Therefore the usefulness of GumbelConv2d seems pretty limited.

I decided to implement the directions of the prototype parts in a different way, more straightforward. It's a simple two-layer convolutional network with a large MaxPool2d intermediate layer and my SoftMaxPool layer that implements differentiable disjunction. Both the accuracy and gradient interpretability are surprisingly high after the initial experiments. I'm excited to explore this direction.

## 29.07.2024 - 05.08.2024

This weeks work was all about clarification - of both ideas and the code. In particular I think that Prototypical Parts Networks have done their part - they inspired an interesting convolutional architectures. But the reliance of ProtoPNets on black-box feature extractors (which are hardly invertible) feels like too much of a stretch - in particular the produced explanations are [not faithful enough](https://arxiv.org/pdf/2302.08508) in the sense they don't reflect the model's actual decision process.

I defined a new research goal I'd like to pursue in the following weeks: design an **accurate** neural network model with **interpretable gradients** on CIFAR10.

### Accurate:
At least 94% clean test accuracy (human-level performance).

### Interpretable gradients:
Input-space gradients of class logits are (after renormalization to RGB space):
1. *perceptually aligned*, i.e. they resemble features meaningful of humans (e.g. objects with masked background)
1. *accurate*, i.e. they represent features relevant to the given class (in particular they are different for different classes)
1. *high-quality*, i.e. they capture fine-grained details of the features

I've created a new repo specifically for CIFAR10. I've adapted the code from [this](https://github.com/KellerJordan/cifar10-airbench) repository to speed up experiments as much as possible. I've invited my friend as a collaborator.

### Motivation

Input level gradients are the most natural model explanation methods - they are theoretically motivated, easy to compute and reflect accurately the training process (backpropagation). However, gradients of SOTA models are [noisy](https://arxiv.org/abs/1706.03825) and hardly interpretable. Alternative feature attribution methods like [guided backpropagation](https://arxiv.org/abs/1412.6806) produce high-quality visaulizations but lack theoretical justification and raise serious concerns about their faithfullness (i.e. by not differentiating between classes), see [here](https://arxiv.org/abs/2104.06629).

But is there really a need for better explanation tool than the good old class logit gradient? The existence of [adversarial examples](https://arxiv.org/abs/1312.6199) suggests that noisy gradients might in fact be faithful representation of model's unreliable decision process and not a mere artifact of oversensitivity.

### Initial motivating results

> NOTE: The following results are for two first classes of CIFAR10 (plane and car).

Consider a simple model defined as follows (achieves 90% accuracy):

```python
nn.Sequential(
    OrderedDict(
        [
            ("conv1", nn.Conv2d(cfg["n_channels"], 120, 10, stride=2, padding=0)),
            ("amp1", nn.AdaptiveMaxPool2d((1, 1))),
            ("flatten", nn.Flatten(1)),
            ("bn1", nn.BatchNorm1d(120)),
            ("mean", Mean(24)),
            ("smpool", SoftMaxPool(cfg["n_classes"])),
            ("scale", Scale()),
        ]
    )
)
```

Intuitively this finds the maximal group (prototype) of `5 = 120 / 24` local features (5 prototype parts) that are all present on the image (hence it defines a disjunction over conjunctions of base features). This was directly inspired by my work on Prototype Networks.

The learned filters are as follows (the groups of 5 consequtive features are clearly visible, the first 60 features are for plane, and the next 60 are for car).

![image info](./docs/assets/disj_conj_features.png)


For the 60 initial test images the gradients look like this:

Plane:
![image info](./docs/assets/disj_conj_plane.png)

Clean images:
![image info](./docs/assets/test.png)

Car:
![image info](./docs/assets/disj_conj_car.png)

### Simple directional bias

The Conjunction-Disjunction model looks promising but there is a clearly visible problem - we loose the spatial relations between parts. This inspires the following improved architecture which roughtly implements directional invariance (achieving 94% accuracy):

```python
nn.Sequential(
    OrderedDict(
        [
            ("conv1", nn.Conv2d(cfg["n_channels"], 120, 10, stride=2, padding=0)),
            ("mp1", nn.MaxPool2d(3)),
            ("bn1", nn.BatchNorm2d(120)),
            ("act1", nn.GELU()),
            ("conv2", nn.Conv2d(120, 120, 4, stride=1)),
            ("amp1", nn.AdaptiveMaxPool2d((1, 1))),
            ("flatten", nn.Flatten(1)),
            ("smpool", SoftMaxPool(cfg["n_classes"])),
            ("scale", Scale()),
        ]
    )
)
```

Let's take a look at the gradients:

Plane:
![image info](./docs/assets/directions_simple_plane.png)

Clean images:
![image info](./docs/assets/test.png)

Car:
![image info](./docs/assets/directions_simple_car.png)

This looks much better but still there is a lot work to do. I think the next layer should be the GumbelConvolution.

### GumbelConvolution

The GumbelConvolution might be usefull here after all as it encodes an intuitive reasoning process. To recap, this is a Convolution with every kernel (output feature) attending to only one input channel per spatial dimension. GumbelConvolution requires more refined training procedure than I initially thought - we have to force the gumbel distribution to approximate more and more closely the one-hot vector as the training  progresses. This seems crucial for allowing the model to actually choose the one channel and may provide benefits - if not for accuracy, then probably for (gradient) interpretability.