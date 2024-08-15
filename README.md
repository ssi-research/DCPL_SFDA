# De-Confusing Pseudo-Labels in Source-Free Domain Adaptation (ECCV 2024)

Our paper "De-Confusing Pseudo-Labels in Source-Free Domain Adaptation" is accepted to ECCV 2024!
[[Paper]](https://arxiv.org/pdf/2401.01650)

Code will be released soon.

## Overview



Source-free domain adaptation (SFDA) aims to adapt a source-trained model to an unlabeled target domain without access to the source data. SFDA has attracted growing attention in recent years, where existing approaches focus on self-training that usually includes pseudo-labeling techniques. In this paper,  we introduce a novel noise-learning approach tailored to address noise distribution in domain adaptation settings and learn to \textbf{de-confuse the pseudo-labels}. More specifically, we learn a noise transition matrix of the pseudo-labels to capture the label corruption of each class and learn the underlying true label distribution. Estimating the noise transition matrix enables a better true class-posterior estimation, resulting in better prediction accuracy. We demonstrate the effectiveness of our approach when combined with several SFDA methods: SHOT, SHOT++, and AaD. We obtain state-of-the-art results on three domain adaptation datasets: VisDA, DomainNet, and OfficeHome.


<div  align="center">    
<img src="images/framework_upd.png"  height="380px"/> 
</div>
