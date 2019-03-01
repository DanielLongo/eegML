## Electroencephalography Machine Learning ##
#### Goals: ####
* Improve data augmentation: synthesize EEGs with a forward model enabled recurrent conditional wGAN
* Improve understanding and diagnostics: Create a siamese network capable of generated a manifold of EEGs
* Imporved singal processing: Remove Artifacts from EEGs

#### TODOS: ####
- [ ] Test g loss with c and d attributes
- [x] Create forward model enabled generator 
- [x] Create conditional generator (concat)
- [ ] Create conditional generator (projection)
- [x] Enable larger continuous EEG generation (Add S<sub>t</sub> as input)
- [x] Create convolutional varient
- [x] Create remove noise network (conv and recurrent) 
- [ ] Test different intermediate representations
- [ ] Get the entire architecture to compile 
- [ ] Train Siamese Network 
- [ ] Use Siamese Network to Generate a manifold 
- [ ] Use cGAN for data augmentation
  

#### Project Plan: ####

<img src="https://github.com/DanielLongo/eegML/blob/master/ProjectPlan/pg1.png"/>
<img src="https://github.com/DanielLongo/eegML/blob/master/ProjectPlan/pg2.png"/>
