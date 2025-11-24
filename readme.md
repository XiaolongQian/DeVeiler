# Learning Latent Transmission and Glare Maps for Lens Veiling Glare Removal [[arXiv]](https://arxiv.org/abs/2511.17353v1) [[Project Page]](https://xiaolongqian.github.io/DeVeiler-page/)

[Xiaolong Qian](https://github.com/XiaolongQian/)\*<sup>1</sup>, [Qi Jiang](https://github.com/zju-jiangqi/)\*<sup>1</sup>, [Lei Sun](https://ahupujr.github.io/)<sup>2,†</sup>, [Zongxi Yu](https://github.com/ZongxiYu-ZJU/)<sup>1</sup>, [Kailun Yang](https://yangkailun.com/)<sup>3</sup>, [Peixuan Wu](https://xiaolongqian.github.io/DeVeiler-page/)<sup>1</sup>, [Jiacheng Zhou](https://xiaolongqian.github.io/DeVeiler-page/)<sup>1</sup>, [Yao Gao](https://github.com/LiGpy/)<sup>1</sup>, [Yaoguang Ma](https://scholar.google.com/citations?user=UnVrCQ0AAAAJ&amp;hl=zh-CN)<sup>1</sup>, [Ming-Hsuan Yang](https://faculty.ucmerced.edu/mhyang/)<sup>4,5</sup>, [Kaiwei Wang](http://wangkaiwei.org/)<sup>1,†</sup>

**Affiliations**
- <sup>1</sup>Zhejiang University
- <sup>2</sup>INSAIT, Sofia University “St. Kliment Ohridski”
- <sup>3</sup>Hunan University
- <sup>4</sup>University of California, Merced
- <sup>5</sup>Google DeepMind

**Notes:** \* Equal contribution &nbsp;&nbsp; † Corresponding author

## Abstract
Beyond the commonly recognized optical aberrations, the imaging performance of compact optical systems—including single-lens and metalens designs—is often further degraded by veiling glare caused by stray-light scattering from non-ideal optical surfaces and coatings, particularly in complex real-world environments. This compound degradation undermines traditional lens aberration correction yet remains underexplored. A major challenge is that conventional scattering models (e.g., for dehazing) fail to fit veiling glare due to its spatial-varying and depth-independent nature. Consequently, paired high-quality data are difficult to prepare via simulation, hindering application of data-driven veiling glare removal models. To this end, we propose VeilGen, a generative model that learns to simulate veiling glare by estimating its underlying optical transmission and glare maps in an unsupervised manner from target images, regularized by Stable Diffusion (SD)-based priors. VeilGen enables paired dataset generation with realistic compound degradation of optical aberrations and veiling glare, while also providing the estimated latent optical transmission and glare maps to guide the veiling glare removal process. We further introduce DeVeiler, a restoration network trained with a reversibility constraint, which utilizes the predicted latent maps to guide an inverse process of the learned scattering model. Extensive experiments on challenging compact optical systems demonstrate that our approach delivers superior restoration quality and physical fidelity compared with existing methods. These suggest that VeilGen reliably synthesizes realistic veiling glare, and its learned latent maps effectively guide the restoration process in DeVeiler.

