# Collective X Adversarial Active Matter simulation

This repository has provided code implementation of the Active Matter simulation written in C++. Code and library can be found in the directory */AMSim/AMSim*. Data Analysis code can be found in */Data Analysis python*. Short movie of the simulation result can be found in */Media*. The thesis report is included in root directory in pdf form.

## Overview of Project

Collective motion including flocking and clustering can often emerge from a system consisting of many active matters. For instance, flocks of bird, micro-organisms or even crowd of peoples. External force can be applied on such active matters, for example, Active Brownian Particles to perform collective behaviour, and such force can be approximated via Reinforcement Learning. 

We proposed an approach that combines model-based learning with temporal different learning that allowed us to train particles species that learn to perform flocking and clustering motion, utilizing the external force. We also demonstrated how by placing both species of particles in the same system can form adversarial behaviour against other species, which resemble the prey-predator model.

Some examples of the interaction between agents within the setting of prey-predator model  
![Interaction of flocking and clustering agents](/Media/ssd.png)

This project is completed by me during my master degree (Machine Learning in Science) as my master thesis, under the supervision of Dr Tom Oakes. Special thanks to Dr Tom Oakes who has provided tons of feedback and guidance, which is essential for completing this project.

## Simulation Setup Guide

**Required Library:**  
GLUT  
Eigen

**Agent Class:**  
Contain logic for Model-based RL agent. Agent contains two main components: Model that store state transition probability and state value function learned. This class also included data storing API (where the model and state value function can be stored in CSV or data file) and regularization tricks (epsilon and learning rate decay).

**Environment Class:**  
Environment of active matter (aka particles). Movement logic, Initialization of active matter, properties of the environment and active matter, and display rendering code can be found in this class. Action list within the MDP formulation can be fed into the step function, which then returns reward list, new environment state, and termination flag (keep in mind that action and reward need to be wrapped and passed by list since environment contains more than a single active matter).

**Main Class:**  
Responsible for running the simulation and training agents. Two separate agents are initiated as flocking agent and clustering agent. Flocking agents are trained to maximize system active work while clustering agent will attempt to achieve the opposite. Iteration of the exchange between agent and environment follow the workflow of OpenAI Gym. This class also allowed multiple simulations to run in parallel. 

## Result

Result of our trained agents is shown in the below videos and images. Note that the videos are speed up.

Formation of Flocking and Clustering agents where both exist in the same system (prey-predator model).  
![Formation of both flocking and clustering agents](/Media/progass-ca.png)  
https://github.com/zfoong/collective-x-adversarial-ams/blob/master/Media/demo-both.mp4

Formation of Clustering agents only  
![Formation of clustering agents](/Media/progass-a.png)  
https://github.com/zfoong/collective-x-adversarial-ams/blob/master/Media/demo-clustering.mp4

Formation of Flocking agents only  
![Formation of both flocking agents](/Media/progass-c.png)  
https://github.com/zfoong/collective-x-adversarial-ams/blob/master/Media/demo-flocking.mp4

Please refers to the [thesis report](https://github.com/zfoong/collective-x-adversarial-ams/blob/master/Master_Thesis_tham_yik_foong.pdf) for other details of experience performed and its result.
