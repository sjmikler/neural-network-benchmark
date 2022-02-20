import datasets
import models
import training

tds, vds, input_shape, n_classes = datasets.get_cifar(100, 100)
model = models.ResNet(input_shape=input_shape, n_classes=n_classes, version="WRN16-8")

tds = tds.repeat()
training.run(model, tds, None)
