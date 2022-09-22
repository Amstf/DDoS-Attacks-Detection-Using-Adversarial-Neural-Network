# DDoS-Attacks-Detection-Using-Adversarial-Neural-Network
## Abstract
In a Distributed Denial of Service (DDoS) attack, a set of compromised internet-connected devices (distributed  servers, personal computers, Internet of Things devices, etc.) are used to overwhelm a target (server, service, or network) with a huge flood of requests, so that it can no longer satisfy legitimate requests. DDoS detection is a challenging issue in the cybersecurity domain, which was addressed recently through the use of Machine learning (ML) and Deep Learning (DL) algorithms. Although ML/DL can improve the detection accuracy, but they can still be evaded - ironically - through the use of ML/DL techniques in the generation of the attack traffic. In particular, Generative Adversarial Networks (GAN) have proven their efficiency in mimicking legitimate data.
This Project addresses the above aspects of ML/DL-based DDoS detection and anti-detection techniques.
First, we propose a DDoS detection method based on the Long Short-Term Memory (LSTM) model, which is a type of Recurrent Neural Networks (RNNs) capable of learning long-term dependencies.
The detection scheme prove a high accuracy level in detecting DDoS attacks. Second, the same technique is tested against different types of adversarial DDoS attacks generated using GAN. 
The results show the inefficiency of LSTM-based detection scheme. Finally, we demonstrate how to enhance this scheme to detect  adversarial DDoS attacks. Our experimental results show that our detection model is efficient and accurate in identifying GAN-generated adversarial DDoS traffic.

### 👨🏻‍💻 &nbsp;Tasks

&nbsp; We developed a GAN model generator capable of creating DDoS traffic that closely matches the DDoS instances from the dataset. We modified the values present in the DDoS-functional features in the generated DDoS traffic to make them look similar to the benign instances. 

&nbsp; We built a new dataset based on the combination of the generated and the original dataset, with two classes: real and fake.   

&nbsp; We trained a new model using the new dataset, to be able to detect the fake or generated data.  

&nbsp; We trained another model using the original dataset including only the DDoS's functional features, to be able to distinguish between DDoS and normal samples.