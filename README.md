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

## 05.08.2024 - 12.08.2024

Mostly technical week both in terms of code and defining proper learning strategy. I think I'm starting to make the GumbelConvolutions work but I'm not there yet. There are glimpses of interesting level of interpretability.

## 12.08.2024 - 19.08.2024

I'm convinced that GumbelConvolution + SoftMaxPool are the way to go. They allow for designing fully interpretable neural network architectures. A simple example is this:

```python
            ("conv1", nn.Conv2d(cfg["n_channels"], 120, 10, stride=1, padding=0)),
            ("mp1", nn.MaxPool2d(5, padding=2, stride=10)),
            ("bn1", nn.BatchNorm2d(120)),
            ("act1", nn.GELU()),
            ("conv3", GumbelConv(120, 60, 3, stride=1, hard=True)),
            ("amp1", nn.AdaptiveMaxPool2d((1, 1))),
            ("flatten", nn.Flatten(1)),
            ("smpool", SoftMaxPool(cfg["n_classes"])),
            ("scale", Scale()),
```

Every gumbel unit (output unit of the 3x3 GumbelConv) selects one input filter per spatial location. Every input filter is easily visualizable as it corresponds to a 10x10 image patch. The MaxPooling layer is designed so that patches don't overlap too much (to avoid obscure interactions of filters). To visualize the gumbel unit we compute its gradient on a few samples of Gaussian noise input images. Here is such a sample for gumbel unit no. 59:

![image info](./docs/assets/gumbel_59.png)

I think that gradient alignment on Gaussian noise is a way better tool for assessing model interpretability then gradient alignment of natural images. It has two benefits:
- global interpretation of unit activation patterns (not dependent on particular input)
- sanity check - as even simple edge detectors have very nice gradients on natural images

Further work includes designing architectures with higher-resolution Gaussian-noise gradients, probably by stacking more GumbelConvolutions together.

## 19.08.2024 - 26.08.2024

I experimented with 3 layer GumbelConvolutions to make the gradients more detailed. The initial results prompted me to implement generalised version of GumbelConv - one that can look at more than 1 input channel per spatial dimension of the kernel. After some reading and thought I decided to implement it in a pretty straightforward way, sligthly changing the implementation of torch.nn.functional.gumbel_softmax to take topk elements instead of max. I have yet to test if this can help with improving the results.

It's quite apparent that we need to find a proper bias-variance trade-off and the topk parameter seems to be a good candidate here. Small value of topk corresponds to a large inductive bias whereas large values to larger sensitivity of the filters. If the filters are oversensitive (high variance) then the gradients will be noisy; if they are biased then the gradients will be stable but not very expressive. Therefore it seems that we should aim to strike the balance between the stability and sensitity of network filters. Probably the bias should increase with the depth of the layer - lower layers should be more sensitive while deeper layers - more biased.

The networks I'm considering are quite simple but the interpretability of 2 layer network suggests that there is something interesting going on (it's not trivial to make 2-layer network fully interpretable). If I can figure out how to increase the resolution and variance of gradients by adding more layers then I think I will essentially solve computer vision, which is quite exciting. On the other hand this observation alone might indicate that it can be a difficult step.

## 26.08.2024 - 02.09.2024

It seems that mixing more channels in the GumbelConv is NOT the way to go. I failed to make the networks more interpretable that way. The gradients are too noisy and don't encode enough inductive bias. These experiments suggests that standard ConvNets (which are a corner case of GumbelConvs) are just glorified edge detectors that do classification based on counting low-level structures (e.g. class-specific edges or textures) and hardly incorporate any high-level spatial correlations.

I think I should experiment more with the ideas I developed in my recent paper, i.e. semantic features. I'm starting to see a general way of building white box neural network architectures. Basically an output unit of a layer should be a disjunction over the variants of the unit. Disjunction is implemented by SoftMaxPooling. The variants of the unit could be different types or poses of on object or a part of the object (i.e. a wheel of a car). To avoid just a few variants dominating the others we add a BatchNorm layer. Such a network basically performs clustering (every unit is an average of the training examples in its cluster). Now, every variant is itself a semantic feature as defined in my paper, i.e. it's MaxPooled over it's local variations (e.g. small translations of affine/projective transformations). Every variation of the variant is just a linear combination of lower-level features. In particular this can be a GumbelConvolution with just one input channel per spatial dimension. The lower-level layer is build in the same way. 

For example a convolutional classifier consisting of 2 such layers could be interpreted as a disjunction over different variants of objects with every variant defining a spatial relations between parts; the parts are themselves disjunctions of their variants. The tricky part is to make the parts indeed encode the variants of the same semantic entity, e.g. the car should have wheels and the wheel should be a disjunction over various types of wheel.

The parts of the objects in the CIFAR10 dataset should be invariant to small projective transformations, small scale changes and the change of colour palette (optionally different colors could be represented by different units). However I have an idea how to implement the semantic consistency of (variants of) parts in a more general way using standard augmentations and a form of self-supervised learning. Basically the activation of the object-level variants should be invariant over identity-preserving augmentations.

The approach I'm taking here may seem complicated as I introduce several novel notions; however it's a pretty straightforward approach to neural networks and I can almost see a proof that it must work - I build shallow models but take extreme care for the interpretability. I'm almost afraid this approach is too straightforward to work well on harder problems and perhaps is a well-known folklore in the AI community but it's probably not - if one takes into account the disappointing state of academic research marred by the short-term optimisation for citations. The time and further experiments will tell. Anyway it's a fascinating adventure.

## 02.09.2024 - 09.09.2024

This week I've been figuring out the implementation details as there are many moving parts in what I want to do and without proper hyperparameter alignment I could waste a lot of time on failed experiments. I think I got all of the details figured out and I'm ready to put it all together.

Following up on the last weeks sketch of the proposed architecture here are the details:

Object-part local biases (invariance over small local transformations of object parts):

- fixed translations (this is for free in Conv2d + MaxPool2d)
- fixed scale (we will scale the convolutional kernels to encode object-part scale invariance)
- learnable perspective transforms (use [kornia.geometry.transform.warp_perspective](https://kornia.readthedocs.io/en/stable/geometry.transform.html#kornia.geometry.transform.warp_perspective) with learnable matrix; however only the M_{11} and M_{12} matrix coordinates should be learnable as they encode the rigid 3D perspective change; also the matrix M should be multiplied by translation matrices to center the perspective warp around the center of the image and not around the upper left corner)

Object-part variants (non-local variants of object-parts) learned by self-supervision - keeping the maximal values of groups in the penultimate layer constant during training for the augmented variants of the input; we will use the following augmentations from [kornia.augmentation](https://kornia.readthedocs.io/en/latest/augmentation.module.html):
- ColorJiggle
- RandomChannelShuffle

This will allow us to make most of the natural variants of parts activating the single unit in the first layer. Therefore the following layer can be a GumbelConv layer with just one input channel per spatial dimension. This will naturally boost interpretability and the expressive power while keeping the network architecture simple and entirery white box.

The implementation will be based on the [sf_layers](https://github.com/314-Foundation/white-box-nn/blob/main/lib/modules/sf_layers.py) from my last paper, in particular on the ConvLayer. This is very exciting as it turns out that this work incidentally will be a direct continuation of the ideas developed in my paper but for CIFAR10 (which should be harder to dismiss than in the case of MNIST). The approach has matured a lot in the past months and a lot of technical machinery has been developed.

## 09.09.2024 - 16.09.2024

I've implemented the perspective and scale invariant convolutions. However the first results are disappointing - there is no substantial boost neither in interpretability nor in the accuracy.

One day I've spontaneously revisited the past experiments and I've noticed that restraining weights of intermediate GumeblConvs to positive values considerably boosts interpretability (but slightly lowers accuracy). Today I also realised that I don't need GumbelConvs for that and standard Conv2d layers restricted to positive values also show this level of interpretability (provided that the penultimate layer is SoftMaxPool as before). My initial thoughts are that such training regime forces the positive decision process as well as favours weight sparsity - both seem to be relevant for interpretability.

The gradients on test data look nice (gradients on the same inputs - towards cars and planes respectively):

![image info](./docs/assets/positive_lenet_cars.png)

![image info](./docs/assets/positive_lenet_planes.png)

Gradients of units in the penultimate layer on noise look worse but are still intelligible:

![image info](./docs/assets/positive_lenet_units.png)

This looks to be the most promising direction for now.

## 16.09.2024 - 23.09.2024

This week I've been experimenting with positively-constrained convolutions. The constraint boosts gradient alignment of hidden units regardless of network depth which gives full insight into network actions (even if the gradients are computed on noise). This resembles the Guided Backpropagation but is more theoretically sound and faithful to the network's decision process. In particular I can experiment with the network architecture and get immediate insight into the effects of particular architectural choices. I can confirm that intermediate layers indeed learn consistent lower-level features.

The general idea is this - use a positively constrained convolutional backbone to encode bias (local variations of objects) and the SoftMaxPool layer to pool over class variance (different objects within a class). The challenge is to boost the sensitivity to class variance and therefore to boost the accuracy. With interpretable gradients this should be just a matter of time. For example I discovered that it's crucial to disable the learned affine rescaling in BatchNormalization layers to effectively use all the available hidden units. The next idea I'm excited about is to (SoftMax)pool over different convolutional backbones as a single architecture seems to be bound to a particular object scale. The less exciting but more obvious one is to use more data augmentations but I will wait until I feel that I've run out of architectural ideas.

I also need to do some refactoring as the approach has simplified over the last two weeks and I can cut off some parts of code while streamlining the others, in particular the ones responsible for gradient visualization. Current approach also allows me to easily scale to deeper architectures and therefore it's more and more important to move to GPU as the training time on my local machine becomes inconvenient (I try to avoid the overhead connected with working on remote machines but that becomes less and less effective). The refactoring will make the transition easier. Although the https://lightning.ai/ environment allows me to move to remote pretty seamlessly it used to be laggy in the past (but that might have improved in the meantime).

## 23.09.2024 - 30.09.2024

I noticed that the networks I train don't fully utilise the capacity of the first convolutional layer - there are always some dead units. I tried to fix that by adding some noise during training but to no avail. However this seems to be a minor problem and I think I've spent too much time on it.

I started experiments on 3 classes and realised that setting affine=False in the last BatchNorm layer was important for two classes but for 3 classes it doesn't make a big difference - the results are even better with affine=True. This is a reminder that I should be wary not to overfit my findings to a particular oversimplified setting and test on harder problems sooner.

I visualized activation patterns of all hidden layer units for a 4-layer VGG-like network with positively constrained convolutional weights and the results are underwhelming - the features are of really bad quality. I think this is not the problem with the visualization but rather with the architecture - the initial 3x3 convs produce a lot of noise and the next layers focus on low-level local signals instead of global patterns. Going deeper doesn't seem to be a good idea in terms of interpretability.

The insights I made while inspecting gradients suggest that the architectures should be designed in a top-down manner - the last layer is the main organiser of the activation patterns. Lower layers just follow what the top one dictates and adding more lower layers doesn't bring too much value - 3 or even 2 layers should be enough. It's more important to focus on the design of the last layer, e.g. it can represent the entire image or just a smaller sliding window - the sliding window is more adequate for capturing smaller-scale objects. Thus it seems that the ensemble of different shallow models is the way to go.

### Mechanistic interpretability

"Mechanistic interpretability" (of computer vision models) is definitely a fitting keyword here. Basically I try to design a mechanistically interpretable classifier for CIFAR10. I think my approach additionally puts some "simplicity" constraints on the mechanistic explanation as the decision process is required to be "positive", i.e. the network should not use negation, just a mix of "spatially organised" conjunctions and disjunctions. Recently I've been thinking more and more in terms of a locality-sensitive hashing, because what I do is basically assigning a group of buckets per class and using the backbone shallow convnet as a hashing function that maps visually similar inputs to the same bucket. My recent idea to do the ensemble of different models can be reformulated as a simple observation - in principle every bucket could have a unique hashing function, i.e. a unique backbone. This actually could boost the otherwise restricted expressive power of the positive decision process.