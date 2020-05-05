# RLProject

This repo contains code for the project for the course CS5500 : Reinforcement Learning at IIT Hyderabad.
The goal of this project was to train a policy gradient model to play Go on a small board (of size 5x5) and transfer the strategies learned to play a 9x9 board

## Requirements
To run, the following must be installed
- The Go environments must be installed from [here](https://github.com/aigagror/GymGo).
- The Pachi agent can be found [here](https://github.com/openai/pachi-py). 
- We used pyTorch and openai gym to set up the networks

## Files
- policy_network.py : contains the classes for the architecture and the PG algorithm for the 5x5 board. 
- project.py : trains the 5x5 model and uses the trained model to play against pachi
- transfer.py : tries to use the trained model to train a PG network to play on th 9x9 board

### Project Contributers
- Gitanjali Mannepalli (CS16BTECH11014)
- Tejas Ananda (CS16BTECH11043)

Report for the project can be found [here](https://docs.google.com/document/d/1bkD_qSP10ZdyemDFiB7bkxRfAU7GeYfU4rS1qs39HQE/edit?usp=sharing)
