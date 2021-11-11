# CoughLIME
## Listenable Explanations for the Predictions of any COVID-19 Cough Classifier

Extends LIME: https://github.com/marcotcr/lime

Run main.py to generate sample explanations. I

CoughLIME generates explanations in listenable in visual form for the predictions of any COVID-19 classifier from audio data and is specifically tailored to cough data. CoughLIME decomposes the input into different interpretable components and generates explanations by determining the weights of each component towards the classifier's prediction. 

To generate the components, five decompositions are provided:  
- "temporal": splitting the audio array into equal components along temporal axis  
- "spectral": splitting the audio array into equal components along spectral axis  
- "loudness": splitting the audio array along temporal axis according to minima in power array  
- "ls": combining loudness and spectral decompositions  
- "nmf": splitting audio array according to non-negative matrix factorization  

Explanations can be generated for the predictions of the DiCOVA baseline (https://arxiv.org/abs/2103.09148) and CidER (https://arxiv.org/abs/2102.08359) models. An evaluation of the explanations can be performing using pixel flipping for audio and the Delta-AUC. 


