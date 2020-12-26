# Season-Transfer
This is an independent project I undertook, inspired by [1] and Tensorflow's CycleGan tutorial [2]. Whereas the authors of [1] produced winter<->summer generators based on 
photos of Yosemite National Park, I sought a similar model that could transfer the various green hues of spring/summer to the burnt foliage of autumn. In order to complete
this task I downloaded from google images 700 photos each depicting the two seasons. I purposely chose photos of varied settings, urban and suburban parks and streets, rural
pastures, mountaintop vistas, buildings, people, etc., in order to help the network generalize as well as possible.

I have provided several codes using different architectures. SeasonTransfer.py is the implementation closest to [2], using a U-net as the generator and a 30x30 patchgan for the
discriminator. This is the model that the example pictures were generated from. Other implementations that worked well include SeasonTransferSN.py, which provides great
stability due to the use of Spectral normalization [3] (an additional script is provided, SpecNorm.py, with the normalization implementation). Also adaptive instance normalization
led to varied, non deterministic mappings between the seasons (SeasonTransferAdaIN.py). Additionally I have provided a ResNet implementation (SeasonTransferResNet.py), which
is the same implementation from [1], but could not be trained in reasonable time using my laptop (intel I core7). All codes use binary cross entropy as the loss function, and work on photos of 256x256 resolution. Ultimately the quality of the results seemed to be hinded by the limited training set size.

The pictures provided are 3 mappings, both Autumn to Summer and vice-versa, taken from the training set (TrainingSet_[].png). Additionally there are 3 Summer to Autumn mappings (MyPhone_[].png) using various Summer settinsg taken with my phone during Summer 2020.

[1] Zhu, Y., Park, T., Isola, P., Efros A.A. (2018) Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
[2] https://www.tensorflow.org/tutorials/generative/cyclegan
[3] Karras, T., Laine, S., Aila, T. (2019) A Style-Based Generator Archictecture for Generative Adversarial Networks
[4] Miyato, Y.,Kataoka, T., Koyama, M., Yoshida, Y. (2018) Spectral Normalization for Generative Adversarial Networks
