1- We developed a GAN model generator capable of creating DDoS traffic that closely matches the DDoS instances from the dataset. We modified the values present in the DDoS-functional features in the generated DDoS traffic to make them look similar to the benign instances.

2- We built a new dataset based on the combination of the generated and the original dataset, with two classes: real and fake.

3- We trained a new model using the new dataset, to be able to detect the fake or generated data.

4= We trained another model using the original dataset including only the DDoS's functional features, to be able to distinguish between DDoS and normal samples.
