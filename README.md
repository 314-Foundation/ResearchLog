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

## 30.09.2024 - 07.10.2024

I experimented with various "parallel" convolutional architectures as described in the last update (i.e. a "parallel layer" consists of a list of standard layers with matching outputs; their outputs are simply concatenated into the output of the whole parallel layer). I tried different weight-sharing strategies with the extreme case being the SoftMaxPool over N copies of a base neural network (with different initializations). Other designs included sharing the first layers but computing convolutions of different scales in the following layers. The interpretations were consistent with the designs (provided that the convolutions were positively-constrained, as before; otherwise the gradients were noisy). However, both the accuracy and the gradient quality seemed to be worse than those of the baseline architecture (I tried to keep the rough number of learnable params the same across different designs). Sharing network layers across different output units as done in the standard architecture seems to be beneficial in comparison to the parallelised designs - as the lower-level features become more sparse, more general and of higher resolution.

The experiments I made were far from exhaustive. In particular I think that it will be worth revisiting the design with different convolutional scales in the future but for now I concluded that the potential benefits would be marginal. I'm looking for a solution that would present a significant improvement to the gradient quality and the accuracy and I failed to discover such a direction while experimenting with parallel network layers.

That made me rethink the basics - what allows the positively-constrained convolutional architecture to generalise to unseen examples? It seems that the MaxPool2d layers must be the answer. As the gradients show, the network learns to combine spatially-arranged features into objects in particular poses. The MaxPool2d layers allow for some "wiggle room" for the locations of the lower-level parts. In particular in the 2-layer conv-net the second layer defines the rough spatial structure of the local features and the first layer defines the local features themselves. How to make this design generalise better? I think that I could try to somehow expand the wiggle-room of the parts, but preferably in a general and/or learnable way so that I wouldn't have to design this "wiggle space" by hand. This is actually a revisit of the ideas I had before, i.e. the semantic features. But those experiments were conducted for the GumbelConvolutions, which were significantly harder to train than the PositiveConvolutions (positively-constrained) I arrived at in the meantime. I think this is a good moment to revisit those ideas and this work will be done in the following week.

## 07.10.2024 - 14.10.2024

As mentioned last week I tried to increase the expressive power of representation-layer units by increasing the inductive bias of parts, i.e. while every unit in the representation layer represents some configuration of parts-at-locations, I tried to increase the broadly understood "wiggle room" of parts. I failed to increase the accuracy and the quality of gradients. The learnable variants of parts turn out to be mostly the copies of the base variant.

Further experimentation led to me believe that boosting the inductive bias of representation units might actually be a wrong direction here. It looks like the problem is exactly the opposite, i.e. the representation-layer units are matching too broadly to many different variants of the object. The network underutilises it's weights and the representation units are too similar to each other. It seems I should rather be increasing the diversity of representation units now - instead of their expressive power. Only after I'm able to optimally use the network's width (unit diversity) should I increase the individual unit's expressive power (increase network's depth).

This led me to re-examine the BatchNorm layer which is the main factor in making the representation units more diverse - without any form of normalization the representation layer tends to be dominated by just a few most strongly activating units. I researched some work that tried to improve the BatchNorm layer. In particular the [Decorrelated Batch Normalization](https://arxiv.org/abs/1804.08450) and [Iterative Normalization](https://arxiv.org/abs/1904.03441) papers seem especially interesting for my setting, i.e. apart from just batch-normalizing the representation units I should decorrelate them to increase their diversity. In other words the idea is to apply whitening to the layer output in the hope of utilising the set of output units fully and making them as diverse as possible. This can also be applied in the former layers to force the variants of parts to be distinct and not just copies of a base variant. I will experiment with the whitening transformation next week.

## 14.10.2024 - 21.10.2024:

As mentioned last week I tested the Iterative Normalization (which is a more refined form of Decorrelated Batch Normalisation) applied to the representation layer. This made both representation quality and accuracy worse. Thus, it seems that decorrelating representation units does not help the representation quality. I tried various ways to incorporate the Iterative Normalization into the architecture but saw no promising improvements.

Then I hypothesised that I could improve Batch Normalization in another way, i.e. force units to focus only on the top percentile of activations inside the batch (I called it Percentile Normalization). In this way there would be no unit that would dominate the representation as the batch activations would be spread evenly across the representation layer - every unit could be matched to at most k% of examples. This would ensure higher variation across the representation units.

As a result, the unit representations were indeed more "spread out" but that did not improve their quality nor the network accuracy - the representation units were more diverse but those additional variants were essentially duplicates. There was no discernible increase in gradient quality. Combining this with Iterative Normalisation did not help either.

Overall, the standard Batch Normalisation performed the best among the normalization approaches I tested. I suspect that the bottleneck I'm facing is actually poor inductive bias after all - even if I can normalise the activations perfectly the filters might still be too rigid to be matched efficiently across possible variations of inputs and matching too broadly to undesired examples.

Eventually this week's experiments have led me to design a surprisingly straightforward way to add substantial inductive bias into my architecture - simply augment the inputs and then take maximum match across all augmentations of the particular input. This is the first time for weeks that I can clearly see the improvement in the gradient quality! Additionally, the method seems to be easily scalable to more augmentations, bearing promise for improving the gradient quality even further. In particular, I've got an idea how to incorporate color-space invariance into the first layer - this has been elusive for me for the last few months. I will see if proper color-invariance can indeed boost gradient quality next week.

I'm also crystallizing my approach, understanding more and more clearly how every design choice fits in the larger picture - the working name for the proposed architecture is Interpretable Neural Hashing. I've got that outlined in more detail but in short the idea is to memoise the training dataset into a collection of slots (representation units) using a neural network as a hashing function; up to identity-preserving augmentations. Additionally, the gradients of the slots have to be interpretable (perceptually aligned) and stable, i.e. have similar visual properties regardless of the input (even on noise). Intuitively, every slot should correspond to exactly one variant of the input (exactly one type of object) - up to augmentations.

Here is a visualisation of 180 slots learned for 3 first CIFAR classes (plane, car, bird), the gradients towards every slot are computed on gaussian noise:

![image info](./docs/assets/aug_module.png)

## 21.10.2024 - 28.10.2024:

I’ve researched various input augmentations for slot-invariance - both colour and spatial (perspective) augmentations. The improvements seem very modest compared to the required computational overhead.

I realised that one has to be very cautious with the input-augmentations as they can boost irrelevant input features (e.g. the background), making it easier to match low-quality slots and not promoting slot quality enough during training. For example, warping perspective of an image of a bird can boost the green background, which in turn matches with low quality slot that recognises some low-resolution arrangement of green squares. However, I might not have chosen the set of augmentations optimally.

One of my working hypotheses is that I should stick to (approximate) local isometries. In particular, I tried a set of small rotations (by +/-15 degrees followed by optional horizontal flip). This does not reduce the accuracy (but neither increases it) and seems to regularize the slot interpretations moderately - the latter is subjective but the slot interpretations seem to be more "horizontally aligned" as one might have expected. However, the difference is indeed negligible if any. Here are the slot visualizations to compare with last week's:

![image info](./docs/assets/aug_mod_flip_rot15.png)

Another interesting idea here would be to consider 3D local isometries of objects (camera changes) as a better alternative to 2D perspective warps of object images (thus reducing the impact of the background). This would require encoding the learnable depth-channel in the slot representations and also a set of differentiable projections on 2D space - so that the learned internal representations are actually "2.5 D" entities that can be viewed from different angles (and those angles can later be matched against the 2D inputs). This seems feasible but potentially tricky to implement. Another way to do this would be to switch to a different dataset that actually allows for 3D input augmentations. However, in my current estimation, the latter option rises too many methodological problems (e.g. I'd have to prove to the community that the problem is not "too easy").

As a sidenote here, I've checked many ideas and different directions in the recent time. Despite most of them not showing the expected improvements I actually think that it does not necessarily mean that those are not "the right" directions - there are many hyperparameters that describe the proposed solutions. It might be that the various improvements won't work in separation but have to "click" together to present a meaningful advancement. For example, it might be that one has to choose an entire set of input augmentations (both perspective, color and some other types) together with the appropriate architecture for different components to work together and unleash their holistic potential. I will try to find such compatible sets of architectural improvements in the near future.

### The yin and yang of learning: point separation vs point identification (expressive power vs inductive bias)

Even more formally, I have to balance the two complementary aspects of the learning procedure. The model obviously has to be expressive enough to separate points from different classes. On the other hand, it has to generalise i.e. glue together certain sets of points (with the help of the inductive bias). In particular, the diminishing returns I get from boosting the inductive bias in my architectures suggest that I should work more on their separation capabilities. I tried to do this before by playing with different variants of batch normalisation but now I think that this was unlikely to work - if a slot scores one input lower then the other incorrectly, then no form of normalisation will help (as this is a monotonic transformation). Instead of normalising the scores I should do something that can swap the incorrect ordering of scores. One of the most natural ideas is to do (input-weighted) subtraction.

Indeed, restricting the convolutional weights to positive values boosted gradient alignment considerably but also reduced the expressive power. The aligned gradients are due to the resulting positive decision process and the induced sparsity of learned weights - both of them increase the stability of gradients across different inputs (including random noise). It's interesting to try to introduce more expressive power with the help of the negative decision process, while still retaining the gradient alignment. In other words, I could try to disentangle the positive and negative reasoning into separate branches and glue them together in a controlled way. Throwing in a non-linearity could probably boost the expressive power even when compared to the unrestricted convolutional networks, as the positive and negative branch won't be simply added together. I shall play around with this idea and see what happens.

## 28.10.2024 - 04.11.2024:

I added the negative decision process to the architecture. There is a slight improvement in accuracy but nothing substantial. Moreover, the gradients toward negative units are rather trivial - features that lower the score of a given class are simply the features that boost either of the remaining classes. Thus, I can see no substantial benefits to the interpretability; the negative component makes the network more complex and the benefits to the accuracy seem marginal.

I might revisit this in the future but right now I'd like to explore the other thing I was excited about last week, i.e. to explore a 3-dimensional inductive bias. This might be a more appropriate approach after all as boosting the network's ability to separate points in a generic way generally contributes rather to better train memoization than to better generalisation.

The objects in the CIFAR datasets are 3-dimensional therefore I think that the good inductive bias has to account for that. I plan to add a depth dimension to the learnable convolutional kernels, treating them as 3-dimensional template features - and simply project the 3D kernel to 2D kernel in a few different ways to achieve various views of the 3D kernel. Hence, the 3D kernel defines a set of 2D kernels but with intricately connected weights in a way that mirrors the connection of different views of a 3D object, which seems to be an appropriate bias here. I've already made sure that there are tools to do just that in the kornia library and I will do the first experiments soon.

## 04.10.2024 - 11.11.2024:

I don't have much to report this week. I've implemented the ProjectiveConv2d which encodes the 3D projective bias - as mentioned last week. The implementation was tricky in some places and took longer then expected but it should work ok now. I should be able to start the experiments tomorrow.

## 11.11.2024 - 18.11.2024:

I tested the ProjectiveConv2d. This approach offers many possibilities and I definitely haven't tested all of them - there are many ways to project a 3D representation on the 2D space. I decided to rotate the 3D representation along the x and y axis by a certain acute angle in 8 directions - this corresponds to shifting the viewpoint slightly to see the sides of the object.

The results were inconsistent even across similar sets of parameters. Some offered slight improvements while others decreased the metrics. This approach also introduces significant overhead as it makes the training 8x slower. Inspecting the representations revealed that the 3D representation was mostly redundant as usually only one 2D view was relevant. Perhaps there are ways to make the 3D representation useful but I started to rethink my approach.

From a certain perspective tweaking the inductive bias is just a technical detail. I realised that I need to focus on the larger picture. What I did so far is to design an interpretable neural hashing procedure. This is in it of itself a novel design that deserves better scrutiny. In particular, it naturally lends itself to a continual learning paradigm and inspires a different training procedure.

Even if my method won't achieve good test time accuracy on CIFAR it can still excel in different areas. Doing interpretable continual compression of inflowing data with no catastrophic forgetting is actually a pretty exciting thing to do and I realised that this can be achieved by neural hashing. After the model encounters the new data it either assigns it to the matching slot or, if there is no good match, creates a new slot and expands its representation space.

Instead of evaluating test-time accuracy I can propose a different metric - "accuracy after slot update", i.e. I will do one pass through the (unlabelled) test set and filter examples that aren't matched to any of the slots. I will then check the labels of those examples and add new corresponding slots to the network. Then I can compute the accuracy in the next pass. This setting seems better-suited for real-world scenarios than the standard train-test evaluation paradigm. Ideally, the amount of new slots should be low and the new metric I propose is this: how many new slots need to be added to achieve a desired accuracy.

In essence, this is an assignment problem - every example in a batch of data has to be assigned to exactly one slot (existing or new one). Every slot has its assigned label. The goal is to make every slot a high-precision detector of the label. Therefore the entire layer becomes an ensemble of high-precision label detectors. If some examples are left unassigned then I create an appropriate new slot. We'll see how the implementation will go but I think I've figured out most details.

Thus want I think I can do is roughly this:

- Plug & Play module - can be applied to representation space of any backbone architecture (in place of the linear probe).
- Easily detects out-of-distribution data (if no slot is matched)
- Naturally robust to adversarial attacks (adversaries should be out-of-distribution)
- Learns continually (slots are added incrementally)
- No catastrophic forgetting - by design (previous slots are not removed)
- Works well even for very simple backbone architectures (we'll see but the "accuracy after slot update" can be high)
- Can be made interpretable with the right backbone (as I've shown so far)
- Ensemble of high-precision label detectors (slots)
- Slots can be made invariant to arbitrary fixed input augmentations (similar to TTA)
- Backbone should encode local invariance of objects (inductive bias)
- Slots help the backbone to separate classes

## 18.11.2024 - 25.11.2024

Some good old programming this week - nothing too difficult but quite detailed. I've implemented the core functions for the incremental slot-matching mechanism. Have to wrap them up into procedures and do some code cleaning before I can run experiments.

## 25.11.2024 - 02.12.2024

The implementation turns out to be quite demanding. No part is particularly difficult but orchestrating them together requires a lot of care and foresight so that the resulting module is designed well enough to allow for flexible and robust experimentation. It even inspired some nice refactoring of the existing codebase. I think I'm almost there but certainly this takes longer than expected.

## 02.12.2024 - 09.12.2024

I've implemented most of the IncrementalNeuralHashing layer

I've been really humbled by this task. Often I felt that my brain could use more working memory to grasp all the relevant details together so that the code is correct and concise and not an unmanageable mess. Proper understanding of what I really want to do at every step was crucial as it allowed me to simplify the implementation enough to actually carry it out. Even then every non-trivial line of code required careful scrutiny and longish variable names such as `thresholded_bs_scores_normalized` to keep track of things.

In particular, the layer had to account for various backbones and input augmentations to allow for robust experimentation. Tensors had to be L2 normalised and unnormalised appropriately for proper threshold computation and efficient gradient propagation. The optimiser had to be re-initialised after every expansion of the weight matrix and therefore preferably stateless (so I chose the classic SGD without momentum - for this layer only). The weight matrix should be pruned after every epoch to eliminate dead slots. The slot-threshold estimation was a complex tensor-manipulation task on its own and now has to be fitted appropriately in the training loop. There were many other minor architectural/technical decisions I had to evaluate to make the resulting implementation reasonably straightforward and easy to work with.

Things left to do, they seem quite basic:
- select the unmatched examples (actually their variants with maximum norm), normalize them and treat them as potential new slots
- compute their 100% precision thresholds on the current batch (treating them as class detectors) - this is tricky but I already have a function for this so it should be easy to fit in
- select the minimal set of those slots as new slots
- add the new slots to their respective classes
- concat new slots with the old slots, replace the weight and the optimizer
- check if the gradients are preserved; if not, redo the forward pass (don't optimise for performance just yet)
- update other buffers: threshold, n_slots_per_class, n_slot_activations_per_epoch

Plan for later:
- test for bugs
- test on MNIST and see how many slots are added after training
- test the accuracy
- add the minimum set of unmatched test examples as new slots and test the accuracy again (continual learnig)
- test on CIFAR with the interpretable backbone
- check hyperparams such as: 
    - use (batch thresholds) vs (running averages of thresholds) during training (batch thresholds mean positive gradients only)
    - MaxPool vs SoftMaxPool for pooling slot values along the classes
    - initial values of layer params (such as scale)

## 09.12.2024 - 17.12.2024

I've finished implementing the IncrementalNeuralHashing layer, the class has ~300 lines of non-trivial tensor manipulation code and I'm really glad that this technical work is done. It should work fine as I tested relevant parts of the code along the way and the architecture is carefully thought out. Perhaps I should write some unit tests, but I don't want to postpone the experiments too much and such tests are non-that-trivial for tensors. I want to refactor the code a little by extracting a SlotMatcher class that would encapsulate some repetitive computations and make the code easier to use but that is quite straightforward now. It was really hard to wrap my mind around all the details and different scenarios (i.e. training, testing, detecting and adding new slots during training, detecting unmatched examples during eval etc.). I'm excited to start tests now.

The layer should have a nice property - in the worst case it should memoize all the training examples. This would imply poor inductive bias of the backbone. The better the backbone inductive bias, the better the data compression done by the hashing layer. This would be the opposite of a linear layer that just computes an average of class representations in the worst case scenario (where one output unit dominates the others).

## 17.12.2024 - 23.12.2024

I've spent a few days refactoring but it was well worth it. The code got much cleaner, I understood the problem deeper and made some functional improvements. There were some minor bugs in the code but I think I fixed them all. The layer works as expected on MNIST but I'm not very far into experiments just yet (I use MNIST before CIFAR to get a better understanding of the behaviour of the proposed layer).

More importantly I realised that the slot selection I'm performing is an instance of the classic "set cover problem". This allowed me to select much better sets of slots during training. I've re-implemented the well-known greedy approximation as matrix operation in Pytorch for even better performance.

Somehow I feel it's a very good sign to have an NP-hard problem in the core of the proposed learning algorithm - intuitively speaking, learning *should* be computationally hard. In fact, learning can be indeed considered an instance of the set cover problem - you cover observations with the least possible amount of ideas. The fact that this rather philosophical observation is a consequence of technical work is pretty encouraging and it really feels I'm on to something interesting.

## 23.12.2024 - 30.12.2024

In the spare time I had during the holiday season I made more experiments, tests, thinking and some crucial technical improvements to the algorithm. This is the best piece of code I've ever written - I'm really proud to have captured all that complexity in an elegant and manageable way - the hard work has paid off. There's also a compelling theory behind all the stuff which I started to describe formally. The Preliminary experiments on MNIST suggest that it works just like I hoped it would - even the simplest architecture turns out to be accurate and interpretable despite minimal hyperparameter tweaking. I can also see that it can easily detect errors in data and got possible new insights to adversarial vulnerability. We'll see if this transfers to CIFAR. But even if it doesn't (which I find unlikely) I think I have enough material for a very interesting paper.

I've also done some reading on [neural cirtuits](https://distill.pub/2020/circuits/) and sketched the introductory chapters of the paper (Introduction + Related Work) - this might be a little premature but I felt it was a good moment to take a little pause and look at a broader perspective before I dive back into technical work.

## 30.12.2024 - 06.01.2025

The training of the IncrementalNeuralHashing layer looks stable and robust to (sensible) hyperparameter choices. A simple 1-layer architecture "memoises" the train dataset in 998 units (~50x compression rate) and achieves 97.8% test accuracy on MNIST - while being fully interpretable. One can play around with various inductive biases and input-augmentations to achieve even better compression rate and test accuracy.

One of such inductive biases is adding an appropriate convolutional layer (which will be the go-to idea on CIFAR). The interpretability can be preserved if the layer is designed appropriately (in particular, the stride of the MaxPool2d layer has to match the kernel size).

I'm testing various ways to organize 2-layer neural hashing, optimising for the compression rate and interpretability of gradients. One of such ideas is making deeper slots (currently slots are just single layer weights, but in principle they can be arbitrarily deep parallel subnetworks, possibly of different architectures).

## 06.01.2025 - 13.01.2025

I figured out how to train the INH layer without supervision. Initially I thought that labels are essential to train INH properly (e.g. to estimate the thresholds) but I'm pretty sure this can be done without them - the rough idea is to combine INH with something similar to BatchNormalisation. I'm working out the details now; initial results are promising and seem to confirm my intuition.

Unsupervised INH would be useful to train deeper models layer by layer while retaining the mechanistic interpretability of INH in every layer; in general it would be great for interpretable unsupervised clustering.

## 13.01.2025 - 20.01.2025

Did more experiments to figure out how to do the unsupervised INH exactly. BatchNormalisation indeed casts an interesting light here - adding normalisation changes the dynamics of learning so that the layer learns distinctive and more abstract features instead of simply memoising inputs. This improves expressive power and adds another avenue for generalisability (apart from architectural inductive bias).

I also made sure that I can visualize RGB filters in a way that is scale-invariant but also sensitive to translation so that the explanations are well-defined (as my method normalises the inputs).

Unsupervised slots learned without BatchNorm (averages of closely-matching examples):

![image info](./docs/assets/inh.png)

Unsupervised slots learned with BatchNorm (more abstract features):

![image info](./docs/assets/inh_batch_norm.png)

(The target architecture will operate on smaller image patches, these are just to help working out the details of the method)

One of the most important takeaways seems to be the realisation that in the unsupervised case every input (a specific image patch from the sliding window) should be matched with exactly one slot. This isn't immediately obvious as examples generally consist of many distinctive features. But this should be addressed by stacking layers, i.e. the first convolutional layer captures single features-at-locations by sliding windows and the second captures specific arrangements of those features. Assigning more than one feature to one input in one layer seems difficult to do as it usually results in duplicated features.

## 20.01.2025 - 27.01.2025

It's a good moment to recap my work in the past weeks as I encountered many branching paths for further work and I need to make an informed decision on what to focus on next.

My goal is to create an accurate and mechanistically interpretable [neural cirtuits](https://distill.pub/2020/circuits/) for CIFAR and MNIST (simple datasets to reduce the problem space). My definition for mechanistic interpretability is quite strong - I want the input-level gradients towards classes to be perceptually aligned and faithful, in particular I want them to be different for different classes on the same inputs. 

Why gradients? Because they are straightforward to compute, visualise and motivate theoretically. Additionally, gradients of convolutions are computed by transposed convolutions which is a natural way to visualise the convolutional filters in deeper layers.

I’ve been switching between the following aspects of neural network design, deepening my understanding and developing new ideas:

- Intra-layer - lateral interactions of neurons in a single layer, i.e. pooling mechanisms, matching function (computing input-filter similarity, i.e. dot product or inverse L2 distance);
- Batch-wise - various statistics of single neuron activation across many examples, i.e. batch normalization, selection of top k best matches;
- Inter-layer - how to stack layers on each other to preserve interpretability of gradients, i.e. positively-constrained convolutions or Gumbel convolutions;

My core idea are slots. They are the neurons in the last layer of the network (representation space). One slot should capture one variant of the input up to small local perturbations. Classes are (continuous) disjunctions over slots, i.e. input belongs to a class if its representation is similar to one of the slots in the class. Slots are similar to prototypes in ProtoPNets and are one of the tools to make the network more interpretable.

Slots combine intra-layer (adequate class-wise slot pooling) and batch-wise (batch norm) interactions. By restricting slot input weights to positive values (inter-layer interaction) I managed to train on CIFAR some 2-layer and 3-layer networks enjoying interpretable gradients on every layer. The accuracy wasn't satisfactory and also I felt that the quality of gradients could be improved. Also, the same technique didn't seem to produce interpretable gradients on MNIST which was worrying.

I tried many approaches, mostly different inductive biases or batch-wise regularisation. In particular, I felt that I need more control over the batch-wise matching of the slots and therefore I developed the Incremental Neural Hashing mechanism (INH). Basically, it computes batch-level thresholds and assignes inputs that are over this threshold to the slot. This gives me more control than Batch Normalisation, which simply normalises the slots over the batch.

INH required a lot of technical work but turned out to work very well for 1-layer networks. However, training a 2-layer network with INH as the last layer was underwhelming; in particular, I couldn't achieve interpretability of the first layer, contrary to the case with BatchNorm as batch-wise regularizer. I think this is because INH tends to memoise inputs quickly, but the output of the previous layer is noisy at the beginning of training.

That's why I looked at the possibility of training INH without supervision. This would allow me to train network layer-by-layer bottom-up and thus solve the problem of noisy initial inputs. Instead of computing slot thresholds based on label distribution I tried to rely on the assumption that only X% of examples in the batch should be matched with the slot.

This works well, the method learns meaningful features in the slots. I used it to learn most representative local patches: 

MNIST patches using cosine similarity:

![image info](./docs/assets/mnist_patches_cosine.png)

MNIST patches using L2 similarity:

![image info](./docs/assets/mnist_patches.png)

CIFAR patches using cosine similarity:

![image info](./docs/assets/cifar_patches_cosine.png)

CIFAR patches using L2 similarity:

![image info](./docs/assets/cifar_patches.png)

I also realised that my method of visualization of the kernel filters relies on the assumption that the input patches have approximately the same norm. While this is true for large image patches, it's incorrect for smaller ones. That's why I tested L2 convolutions which compute L2 similarity with the input instead of the dot product. Training L2 convolutions requires different initialization, larger learning rate and overall seems more difficult but is doable; L2 filters have more faithful interpretations for small RGB patches and probably are preferable for the initial layer of the network. Deeper layers can be forced to satisfy the similar norm assumption and therefore standard convolutions should be applicable there.

Therefore INH can learn the catalogue of local features in an entirely unsupervised way. The idea is to use that catalogue of such low-level features to learn higher-level features in a similar way.

There are, however, potential problems with that. The next layer may explode, memoising too many features (I expect this to be the case for L2 convolutions). It also may not learn larger features at all and just repeat what the previous layer has learned (this might be the case for standard convolutions, especially with Batch Norm). I also rely on the assumption that there is just one feature in one patch which is not true for larger patches. This assumption can be dropped but it requires further engineering. Most importantly - I'm not sure what the ideal weights of the following layer should look like to preserve interpretability of gradients. It seems that I need to focus on the inter-layer aspect again.

The problem space grows and I don't have the team to pursue all the branches at once. I'll try to choose the most promising one. It would be ideal to reduce the problem space again.

Now, by pure accident, I trained a simple 2-layer network on MNIST - almost a 2-layer MLP - that learned these features in the first layer:

![image info](./docs/assets/mnist_mlp_features.png)

They are particularly good-looking. Unfortunately, the gradients of the second layer units are not interpretable, even if I use slots as the representation space. But it feels like it should be possible to somehow constrain the weights of the second layer to make those interpretable. I mean - the first layer has learned clear parts, the slots in the second layer should just encode small sets of just a few parts, right? Perhaps some sparsity constraint would be enough?

Anyway, this feels like a good problem to study for now - how to make gradients of second-layer neurons interpretable provided that the gradients of first layer neurons are interpretable?

## 27.01.2025 - 03.02.2025

It finally clicked, I've got a great idea, working out the implementation details now but the overall vision is clear. It looks like we've been using convolutions wrong all along.

## 03.02.2025 - 10.02.2025

Mostly pen and paper work this week, I've written down the rough formula for my architecture. I've clarified it enough to implement it efficiently. Contrary to the standard feedforward layer stacking my approach is inherently recursive (not to be confused with recurrent networks) and actually a top-down computation. The implementation itself will help me clarify it further.

I've lost some working time because of flu but it's ok now.

## 10.02.2025 - 17.02.2025

I've simplified the initial implementation of my idea. It can be thought of as nested input unfolding and then folding it again while applying convolutional filters along the way. This is a multi-layer architecture but every layer operates on the input-level, making it straightforward to interpret. It's a little tricky to implement in full generality with many spatial locations of nested sliding windows - so for now I'm assuming that the windows of every layer cover most of the input image - and hope that these large filters will naturally learn to focus on parts of the image. Even if this is too optimistic, coding the simpler architecture first will greatly help to implement the go-to version.

The idea looks very promising in theory and should have been tested empirically already but my mind seems to work slower recently. I've had a big headache for few a days this week - a lot of stress unrelated to the research and apparently I haven't fully recovered from the flu.

## 17.02.2025 - 24.02.2025

I've finally implemented the architecture and can start experimenting. The entire multi-layer network can be interpreted as a set of adaptive filters applied to the input. This looks very optimistic and fits well with my previous research directions; in particular, with slots/neural hashing/ensemble of detectors - which allows for straightforward test-time training. The novelty is in the way the spatial bias is exploited by the subsequent layers but the slot mechanism remains the same.

My recent headaches are probably due to elevated blood pressure. I may have overdone the workload in the past months. I need to take it easier in the coming weeks. But I feel that the research is wrapping up nicely and I can already see the outlines of the coming paper - assuming that the experiments will go as the theory seems to indicate.

## 24.02.2025 - 03.03.2025

The simplified implementation turns out to be rather expensive computationally. This is due to constructing very large tensors during nested (un)folding. The go-to version should be faster. Despite that the experiments are promising and align well with the theory and my expectations.

The main takeaway is that the nested filters should not overlap spatially. This can be approximated by additional regularisation loss or restricting certain weights to positive values. This approach works for MNIST (after some hyperparameter tweaking), the network indeed learns to factorise the data into parts (layer 1) ...

![image info](./docs/assets/mnist_nested_layer1.png)

... that are later assembled into structures (layer 2):

![image info](./docs/assets/mnist_nested_layer2.png)

The network learns objects as different assemblies of parts; this makes the model fully interpretable globally (in input-independent fashion).

On CIFAR the mentioned regularisations don't work well and therefore I need to employ the architectural constraint, i.e. implement the go-to version of my architecture, which has to be done anyway for performance reasons. It's almost done now.

Overall it looks like I have all the tools required to build accurate and mechanistically interpretable deep neural networks for computer vision, I just need to put them together.

## 03.03.2025 - 10.03.2025

This week I took time to contemplate the essence of my solution and what exactly makes it different from the standard feed-forward convolutional layer stacking. On the first glance the nested input unfolding is just a more expensive way to scan for hierarchical spatial features as it makes the same computations many times in different contexts (nested windows) and intuitively can be viewed as just changing the order of operations. Therefore I suspected that my solution might be equivalent to standard layer stacking with some additional constraints. I tried some natural ideas but I failed, it seems that the training dynamics are different after all, which would be good.

My working theory is that the hidden units in feed-forward networks tend to learn superpositions of features, i.e. one output unit can respond to many loosely connected patterns in the previous layer (e.g. imagine two features: one activates the first half of lower-layer neurons, and the other activates the other half. Then a single higher-level unit can learn to activate on both of those features, mostly thanks to the activation function that squashes the information from the previous layer). This boosts the expressive power of such a unit but also obfuscates its activation pattern as the input-space gradients of the recognised features may interfere destructively, making the features difficult to recover after training. In general, the pure feed-forward architecture seems to have no good mechanism to enforce meaningful lateral feature interaction, in particular to discourage random feature overlap (destructive interference) and encourage feature specialisation.

My solution is designed to avoid this destructive feature interference and ecnourage feature specialisation. The features are embedded in the (unfolded) input-space, which forces the feature gradients to interfere constructively to match the input pattern. I'm not sure how to recreate this mechanism using standard layer stacking, it might be possible using some trick from linear algebra but I'm not aware of such. Therefore I'm sticking to the nested layer unfolding.

These considerations have led me to delay the implementation of the final version of my architecture. Even though I have all the details written down on paper, the actual implementation turns out to be quite tedious. However, I should be able to start new experiments soon.

## 10.03.2025 - 17.03.2025

My intuition from last week turned out to be correct as I figured out how to translate my nested architecture to standard feedforward layer stacking. This is all based on a simple, yet seemingly underrated Lemma:

> Let N be a neural network in which every layer is either linear, ReLU or MaxPool. Additionally, suppose there are no bias terms in linear layers. Let u_n(x) be the value of some unit u_n in the n-th layer on input x. Then the u_n(x) is equal to the scalar product of x with the gradient of u_n(x) in x (wherever it's defined).

It's easily proven by induction on n (layer depth).

This is magnificent in the context of interpretability. It allows me to implement all the recent insights efficiently and stay relatively close to the existing architectures. In particular, it will be interesting to see how ReLU layers affect interpretability (as they are arguably the main source of feature superposition).

## 17.03.2025 - 24.03.2025

Experiments are going very well. I easily recreated the results for nested filters, training is much faster and the architecture more natural and modifiable. It's interesting to see how many details are needed to achieve the interpretability, changing a single design decision results in messy neuron activation patterns. However, those design decisions are well-motivated, so there seems to be no accident here.

My idea is essentially a self-supervised regularisation loss that enforces interpretability. I train the backbone network using this loss and the linear classification head on top of it. I can optionally detach the classification head during training to see how well the self-supervised training separates the classes. I observed the following fascinating phenomenon:

> The more interpretable the backbone neurons are, the less the classification head overfits. This connection seems to be very clear, if the backbone neurons have messy gradients, then the head behaves randomly on test data (while fitting well to train). If the gradients are slightly interpretable, then the test performance of the head is somehow better. The best test performance is achieved for the backbone with clearly interpretable neurons. This phenomenon disappears if the classification gradients are allowed to flow through the backbone during training (i.e. the classification head is not detached from the gradient graph).

Other observations include:
- the ReLU activation indeed introduces superposition, i.e. ReLU-units complicate the gradients of the deeper units; in particular, next-layer units resemble a disjunction over features. I'm not sure how to inspect the activation patterns of such units faithfully. Initial experiments (and theory) suggest that ReLU units may not be that important after all and could be used more sparingly without damaging netowrk performance;
- by default, I activate only one neuron in the backbone representation space - this essentially makes the backbone a hashing function that memoises inputs up to the architectural inductive bias. I realised I can loosen this assumption without damaging interpretability - to boost the accuracy and generalisation power. Namely, I can activate more than one neuron in the last layer. Then the interpretations of units become more and more similar to more general features rather than particular data examples.
- The regularisation I use for interpretability recreates the input. It's important to standarise the similarity to allow the network (to learn) to ignore the background.

## 24.03.2025 - 31.03.2025

Recent insights lead me back to the Gumbel/Positive/Disj_Conj Convolutions I explored a few months ago. I understand them much better now. In particular:

- methods that resulted in more interpretable gradients are essentially doing the same thing - preventing gradient interference. Standard networks tend to focus on directions (linear combinations of hidden unit activations) instead of particular units. To achieve unit interpretability, activations have to be adequately constricted, i.e. spatially for vision. Roughly speaking, there should be only one feature per location (per particular layer on particular input). More precisely - convolutions should encode conjunction of disjunctions-at-locations, which is something in-between Gumbel and Positive Convolutions I explored before.
- thanks to the Lemma I mentioned in 10.03.-17.03 update I view the standard network layers as adaptive spatial filters; it's refreshing to examine different layers in this context, for example the BatchNorm layer can be understood as performing two essential regularisations:
    1. spreading the activations across all the units in a layer (preventing the layer to collapse just to few dominant units);
    2. modifying the gradient flow so that it encourages the features to distinguish between examples (instead of just resembling parts of the examples);
- I don’t need the additional reconstruction loss as I understand better how standard loss flows through the network and how high-level features are learned. In particular, I walk away from the hashing paradigm and don’t try to memoize the inputs in the representation layer. Instead, this layer should learn high-level features that separate the classes. In particular, one input can activate more than one feature in the representation space, increasing network capacity.

The preliminary experiments seem to confirm these intuitions, i.e. disjunctions-at-locations indeed boost the interpretability significantly. I checked it in the scenarios that were easy to implement in a non-differentiable manner, i.e. first two layers and the representation layer. In principle the approach should work for any layer but it requires differentiable implementation to allow learning. I already have an idea how to do this, it's a pretty low-level concept as it modifies the scalar product itself - but makes a lot of sense theoretically. There might be some difficulty with proper choice of hyperparameters but I'm quite optimistic.

## 31.03.2025 - 7.04.2025

I'm getting closer and closer to building robust mechanistically interpretable neural networks. My core idea is replacing activation function with differentiable disjunction. This has the following core benefits:

- activation function is layer-global while differentiable disjunction is neuron-specific, i.e. different output neurons can "activate" the input (output of the previous layer) in different ways; This boosts the expressive power of the network;
- activation function kills some gradients (e.g. for negative values in case of ReLU) which makes it problematic to interpret neuron activation patterns faithfully; Neural disjunction doesn't kill any gradients, which allows to faithfully inspect every neuron activation on every input.

Overall I'm leaning toward the following definition of mechanistic interpretability of neural networks:

Definition:
> Network is mechanistically interpretable if input level gradients of every unit are perceptually aligned for every input.

Neural disjunction is straightforward to interpret mechanistically. Notice that standard networks perform two basic logical operations: AND and NOT. Taking a scalar product with a vector of weights (of an output unit) is maximised along the direction of this vector, therefore the unit can be considered a (weighted) conjunction of lower-level features. The NOT gate is implemented simply by weighting the input coordinate by a negative value. However, there is no natural way to encode OR gates in neural networks. Theoretically it could be implemented by the De Morgan Law: OR(A, B) = NOT(AND(NOT A, NOT B)). Although AND(NOT A, NOT B) can be fulfilled by a single layer, the topmost negation requires an additional layer. Also AND(NOT A, NOT B) is negative if A and B are positive, so we need an activation function that preserves high negative values, which generally is a problem (e.g. for ReLU networks it is plainly impossible).

This drawback is partially alleviated by the activation functions - if A and B don't occur together and at least one of them is always suppressed, then AND(A, B) is maximised by either A or B and therefore can be interpreted as OR(A, B). This assumption however is pretty strong. Also, we can never really know if such a unit encodes conjunction or disjunction of A and B, at least not by analysing the network weights - we'd need a deeper knowledge about the data distribution, which generally is infeasible.

It's hard to expect neural networks to be mechanistically interpretable if they cannot encode disjunction of features in a natural way. Perhaps this is the reason for noisy gradients, vulnerability to adversarial attacks and overall difficulty in interpreting models.

My solution overcomes those problems by replacing activation functions with neural disjunction. In this week I tested several hyperparameter choices (i.e. how to implement the neural disjunction exactly) to achieve adequate gradient flow during training. I've done it for the Linear layer and it works great. The next step is to do this for Conv2d. In general I want to use neural disjunction (OR) along the channels while using standard neural conjunction (AND) along the spatial locations of the convolutional filter. This would result in a naturally interpretable layer, i.e. "[(this OR this OR ...) at this location] AND [(this OR this OR ...) at this location] AND ...". This would avoid the gradient interference of different units in the same spatial location which should produce perceptually aligned gradients in the visual domain.

This all really makes sense theoretically and the experiments so far are very promising.

## 7.04.2025 - 14.04.2025

The experiments are going well, the only problem is that my implementation of neural disjunction is slow even on GPU. This is because I need to reimplement the scalar product with some twist and I do it in PyTorch. It seems I'd need to implement it in C and CUDA kernel.

## 14.04.2025 - 21.04.2025

I wrote the theoretical part of the paper. My approach is to explain networks by their gradient fields. I adopt the axiomatic approach, introducing the predicate Expl and 3 axioms of gradient field explainability - motivated by intuitive definition of an explainable gradient field. Then, using those axioms, I introduce 2 novel neural operators (XAND and ExplOR) that provably preserve gradient field explainability. The goal is to show that those 2 kinds of neurons suffice to build performant models.

## 21.04.2025 - 28.04.2025

I explored several methods to reduce the computational cost of ExplOR neuron (formerly called neural disjunction) using existing PyTorch interfaces before jumping into writing the custom CUDA kernel. The main bottleneck is the need to copy the input d times, where d is the number of output neurons, i.e. the memory bottleneck. I optimised the architecture to use existing pytorch operators as much as possible. Then I tried using vmap to vectorise the function across the batch of d output neurons; this helped a lot but still is an order of magnitude too slow. Then I realised that ExplOR looks like a special case of attention operator, i.e. `torch.nn.functional.scaled_dot_product_attention`, and this is highly optimised. However, translating ExplOR to attention requires building big diagonal tensors via `torch.diag_embed` and that again is the memory bottleneck. But this means that it will be much easier to write custom CUDA kernels as I can take the existing `scaled_dot_product_attention` implementation and modify it slightly so that the `diag_embed` is not needed.

In the meantime I made some improvements to the theoretical part of the paper.

## 28.04.2025 - 05.05.2025

This week, I started exploring how to write a custom C++/CUDA kernel in PyTorch to speed up ExplOR. While promising, the process turned out to be quite complex. Fortunately, after revisiting the problem, I found a way to implement the function efficiently in pure PyTorch. By slightly adjusting the formulation and leveraging logarithmic properties, I was able to use a standard einsum operation instead. This is a great outcome, as it saves significant time and effort otherwise needed for writing and testing a custom kernel.

## 05.05.2025 - 12.05.2025

This week, I decided to move away from the axiomatic framing of ExplOR and instead present it as a novel activation function with multiple benefits. The revised perspective highlights its expressive power: it generalizes ReLU and MaxPool, allowing for more output neurons than inputs. Functionally, it can be seen as a fusion of ReLU, MaxPool, BatchNorm, and a special case of attention, while remaining differentiable and efficient. At its core, the operation is a dot product in tropical algebra, with certain twists to improve training dynamics. It can optionally preserve the Gradient Representation Property and ensure no input gradients are entirely zeroed out on any input. Importantly, its expressiveness makes it a strong candidate for use with sparse linear layers, enhancing interpretability. I've implemented the necessary PyTorch modules, carefully handling all the required technical details.

## 12.05.2025 - 19.05.2025

Experiments are going very well. I think I will be able to achieve compelling results on much harder datasets than previously expected.

## 19.05.2025 - 26.05.2025

- Added learnable weights to ExplOR, allowing it to represent weighted disjunctions, in contrast to standard linear/convolutional layers that implement weighted conjunctions.
- This structure enables networks composed of alternating weighted OR and weighted AND layers, allowing filters to be interpreted as logical formulas representing combinations of atomic features.
- Due to distributivity, final activations can be interpreted as disjunctions of conjunctions — i.e., logical expressions in disjunctive normal form (DNF). Therefore, every network feature can be interpreted globally as maximum (disjunction) over certain set of conjunctions (sums) of atomic features (low-level filters). The particular compound feature selected for a given input is equal to the input-level gradient of the feature. Therefore it is optimal that the atomic features are separated spatially to avoid interference of co-occurring features.
- Ran tests on simple networks to probe optimal hyperparameter ranges. Notable findings:
    - BatchNorm significantly improves gradient quality and smoothens the loss landscape.
    - Surprisingly, zero initialization gave the best results for logits.
- Integrated MaxPool2d into ExplOR, making it learnable and improving spatial inductive bias.

## 26.05.2025 - 02.06.2025

This week, my main focus remained on finding the optimal way to combine ExplOR layers with sparse linear layers. Continued experimentation helped clarify the underlying theory, which also led to a refactoring of the code.

In particular, I realized that interpreting a linear layer as an AND operation is inaccurate — a more appropriate analogue for AND would be a softmin or a -logsumexp(-x) function. A linear layer is simply a weighted sum, so the analogy to logical formulas does not hold.

Instead, I’m converging on the idea that the theoretical core of my work should be the local exclusion constraint: the assumption that the input is structured as a set of locations, and that at each location only one feature can be active at a time (i.e. for a fixed input). On the philosophical sidenote: This constraint may in fact define the space itself in physics — or at least the "space" understood as the way the brain organises its inputs.

Viewed this way, the ExplOR2d layer becomes a direct implementation of the local exclusion constraint as a neural network layer. This framing not only simplifies the theoretical side of the paper but also narrows down the space of architectures and hyperparameters I need to consider.

## 02.06.2025 - 09.06.2025

Further improvements to both the theory and the implementation technicalities. I found a neat formalisation in terms of (weakly) indpependent variables and therefore I might return to the first principles approach to interpretability. Implementation-wise I'm discovering how to stack ExplOR layers to make the best out of them.

I've noticed that scaling the logits before applying softmax has a crucial impact. I'm computing something similar to attention over input features, and I need to scale both the input features and the weights before calculating the combined attention scores. There's an interesting trade-off here: if the input scale is larger than the weight scale, ExplOR behaves similarly to Top-k activation — it activates only the largest inputs and is relatively insensitive to the output neuron's weights. Conversely, if the weight scale is larger, ExplOR acts like a neuron with sparse weights. When the scales are comparable, we get a behavior somewhere between Top-k and sparse weights — the output neuron attends to a set of input neurons in an input-dependent way, which is more robust than either of the corner-cases.

## 09.06.2025 - 16.06.2025

I'm beginning to understand the core rules of how to build multi-layer networks with high-quality interpretable gradients. Main take-aways:

- Gradient Representation Property is essential, i.e. enforcing f(x) = <grad f(x), x> for every neuron. This is possible thanks to the key-value structure of ExplOR, the inherent tropical bias in EplOR and a novel activation function SiLUBatchNorm. In this way we can preserve the input-level alignment while accounting for the bias in data and different scales of input neurons.
- Local Exclusion Constraint is crucial for deeper layers, i.e. every output filter should locally attend to a sparse number of input filters per input. This regularises the backward pass, forcing layers to combine information in a comprehensible way. This can be implemented by spatial ExplOR2d.
- Features from disjoint locations can be assumed to be mutually independent and therefore no constraint is needed for integrating information from separate locations (i.e. Convolutions with appropriate dilation).
- ExplOR uses an additional set of weights ("logits") that perform soft subset selection. This allows the output neurons to focus on different sets of input features. Those sets can be initialized with a novel "dobble activation" function. The initial sets can be large as they are further refined by multiplying with the neuron's weight.
- The inclusion of relatively constant (but still learnable) logits increases the biological plausibility of the architecture (logits can be interpreted as physical neural connections while weights - as the strengths of those connections).
- Overall, the Location-independence and Local Exclusion seem to be good candidates for axioms for explainability-preserving neurons. This corresponds to how humans tend to represent the data as distributed spatially (features-at-locations).
- To ground the axioms of explainable filters we'll assume that affine filters are explainable. However, they are not the only "primitive" explainable filters - we will be basing on the empirical fact that lower layers of most neural networks are usually explainable. The explainability-preserving neurons (abiding to the Local Exclusion principle) will be defined in deeper layers.
- The ExplOR is indeed an input-dependent attention over the neuron-dependent subset of input features. It selects (in a soft manner) a dominating activation but only among the neuron-dependent subset of activations. Thus it is more robust then mere TopK activation selection or a sparse weight, but functionally, on a given input, this behaves like a TopK activation and a sparse weight at the same time. This makes it effectively sparse but still easy to train
- Therefore, I don't really need a XAND layer (with learnable sparse weights) as ExplOR is already a more robust implementation of such a layer.
- I observed that the best results require logits >> (normed inputs), i.e. that the architectural bias dominates the variance in data (this makes sense for proper discrimination of output neurons via different selected subsets of input neurons).

## 16.06.2025 - 23.06.2025

This week, there has been a significant development in my research. I've done ablations and it turns out I can achieve interpretability in a much simpler way. Specifically, I developed an activation function that can be dropped in most architectures, rendering it way more interpretable while preserving its performance.

For example, these are the visualizations of what a modified VGG13 network looks at when recognizing the particular class of CIFAR10 (for first 3 classes, plane, car and bird):

![image info](./docs/assets/vgg_13.png)

The modified network achieves similar performance as the standard one. I'm refining a compelling theory of why this is indeed a faithful representation of the network's decision process for a given input x and the given label.

My plans for further work include training at least VGG and ResNet networks (possibly also ViT) on CIFAR10 and on some harder dataset, for example Imagenette, to show off that the drop-in module and the corresponding theory work at scale.

## 23.06.2025 - 30.06.2025

This week's results frankly blew my mind. After further ablations I realised I can generate perfectly aligned and coherent counterfactual examples on any input on most PRETRAINED networks. I can use the same method to visualise faithfully network's decision process.

Sample images from ResNet50 pretrained on ImageNet (with simple modification to gradient flow, with no retraining):

Base:

![image info](./docs/assets/base.png)

Gradients:

![image info](./docs/assets/gradients.png)

Counterfactuals toward "church":

![image info](./docs/assets/church.png)

Diff "church" - original:

![image info](./docs/assets/church_diff.png)

Counterfactual toward "ostrich":

![image info](./docs/assets/ostrich.png)

The gradients and counterfactuals of the original network using the same technique looks like noise.

To explain this phenomenon I dived deeper into neural network theory and I can show compelling arguments why neural networks become kernel machines during training. This means that neural nets become linear in the appropriately chosen feature space during training, arguably early on. This explains why these visualisations are possible and even fundamental for the network's decision process.

## 30.06.2025 - 14.07.2025

I've improved my method and done experiments on other ReLU architectures (resnext50_32x4d, inception_v3, vgg11_bn, densenet121) and it works just as well; preliminary experiments on architectures using SiLU and GELU are also promising but I'm leaving those out of the scope of the paper. I've been refining the theoretical part relentlessly to show that my method of aligning gradients is indeed faithful to the network's decision process, as this gives the whole new depth to the explanations I'm producing. I'm happy with the result, it's not a full formal proof but still pretty convincing and concise semi-formal arguments. Thus I've completed the sketch of the paper, i.e. how the formal presentation will go, which visualisations to include etc. and I can proceed to write it all up for the first, lean version of the paper.

## 14.07.2025 - 21.07.2025

It's going well. I should be able to publish the paper by next week.

## 21.07.2025 - 31.07.2025

The paper is publicitly available at [https://arxiv.org/abs/2507.22832](https://arxiv.org/abs/2507.22832)