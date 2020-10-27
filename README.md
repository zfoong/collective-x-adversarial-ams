# Collective X Adversarial Active Matter simulation

This repository has provided code implementation of the Active Matter simulation written in C++. Code and library can be found in the directory */AMSim/AMSim*. Data Analysis code can be found in */Data Analysis python*. Short movie of the simulation result can be found in */Media*. Thesis report is included in root directory in pdf form.

## Overview of Project

Collective motion including flocking and clustering can often emerge from a system consisting of many active matters. For instance, flocks of bird, micro-organisms or even crowd of peoples. External force can be applied on such active matters, for example, Active Brownian Particles to perform collective behaviour, and such force can be approximated via Reinforcement Learning. 

We proposed an approach that combines model-based learning with temporal different learning that allowed us to train particles species that learn to perform flocking and clustering motion, utilizing the external force. We also demonstrated how by placing both species of particles in the same system can form adversarial behaviour against other species, which resemble the prey-predator model.

## Simulation Setup Guide

**Required Library:**  
GLUT  
Eigen

**Agent Class:**  
Contain logic for Model-based RL agent. Agent contain two main components: Model that store state transition probability and state value function learned. This class also included data storing API (where model and state value function can be stored in CSV or data file) and regularization tricks (episilon and learning rate decay).

**Environment Class:**  
Environment of active matter (aka particles). Movement logic, Initialization of active matter, properties of environment and active matter, and display rendering code can be found in this class. Action list within the MDP fomulation can be fed into the step function, which then return reward list, new environment state, and termination flag (keep in mind that action and reward need to be wrapped and passed by list since environment contain more than a single active matter).

**Main Class:**  
Responsible for running the simulation and training agents. Two seperate agents are intiated as flocking agent and clustering agent. Flocking agents are trained to maximize system active work while clustering agent will attempt to achieve the opposite. Iteration of the exchange between agent and environment follow the workflow of OpenAI Gym. This class also allowed multiple simulation to run in parallel. 

## Result

Result of our trained agents are shown in below videos. Note that the videos is speed up.

Flocking and Clustering agents exist in the same system (prey and predator model).  
https://github.com/zfoong/collective-x-adversarial-ams/blob/master/Media/demo-both.mp4

Clustering agents only  
https://github.com/zfoong/collective-x-adversarial-ams/blob/master/Media/demo-clustering.mp4

Flocking agents only  
https://github.com/zfoong/collective-x-adversarial-ams/blob/master/Media/demo-flocking.mp4

Please refers to the thesis report for other details of experience performed and its result.
