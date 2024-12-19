# Diffusion-based turbine wake model

This repository provides an implementation of a diffusion-based turbine wake model based on (Ali et al. 2024): 
```
Ali K, Stallard T, Ouro P. A diffusion-based wind turbine wake model. Journal of Fluid Mechanics. 2024;1001:A13. doi:10.1017/jfm.2024.1077 
```

Bibtex entry:  
```
 @article{Ali_2024, 
 title={A diffusion-based wind turbine wake model},    
 volume={1001},
 DOI={10.1017/jfm.2024.1077},
 journal={Journal of Fluid Mechanics},
 author={Ali, Karim and Stallard, Tim and Ouro, Pablo}, 
 year={2024},
 pages={A13} 
} 
  ``` 

  Please cite the above article in case this model is used.   

  Key points:
- This model is based on the assumption that normal to the streamwise direction, the shape of a turbine's wake behaves similar to the diffusion of a passive scalar.     
- The model naturally evolves from a radially uniform shape in the near wake to a Gaussian shape in the far wake. 
- The wake length scale is adjusted to take into account the near-wake region. 
- The model presents analytical solutions to integrals of the modified Bessel function within the context of enforcing the conservation of linear momentum.
    
For any enquiries, please contact: karim.ali@manchester.ac.uk

## Usage
The Python script **diffusion_wake.py** contains two main functions:
- **diffusion_model**: which is the implementation of the diffusion-based wake model 
- **example**: an example comparison of the diffusion-based wake model to experimental measurements in the wake of a turbine

The **diffusion_model** function is called as
```
diffusion_model(yds, ct, ti, xd, lnw)
```
where
- yds: a list of values for the lateral coordinate y normalised by the diameter of the turbine.
- ct: the thrust coefficient of the turbine.
- ti: the turbulence intensity of the free-stream flow.
- xd: the streamwise distance measured from the turbine location, normalised by the turbine's diameter.
- lnw: the streamwise extent of the near wake region, normalised by the turbine's diameter.

The function **example** can be called as
```
example()
```
which will create an image named **example.png** containing the wake comparison.