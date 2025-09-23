# HiDiffEnergy

Main repository for code and dataset. Coming soon
![Description of image](main_diagram.png)

# HiDiffEnergy: A Hierarchical Diffusion Model for Energy Load Generation

HiDiffEnergy is a deep learning project for generating high-fidelity, long-sequence synthetic residential energy load profiles.  
It utilizes a two-stage hierarchical diffusion model to capture both the long-term structure and the short-term volatile details of electricity consumption and solar generation data from the **Ausgrid dataset**.

---

## Abstract
Generating realistic long-term energy consumption and solar
generation data is challenging because the data contain multi-
scale temporal features, ranging from long-term seasonal
trends to short-term daily and hourly households behaviors.
Existing generative approaches tend to overlook this multi-
scale nature, capturing either large-scale trends or small-scale
variations, but rarely both. We propose HiDiff-Energy, a hier-
archical diffusion framework with two stages. The high-level
model captures global dynamics over weeks and months,
while the low-level adds high-frequency, household-level
variations. To handle long sequences efficiently, the low-level
trains on shorter segments but produces outputs that remain
coherent across the full time span. The model also includes
conditional embeddings of household identity and tempo-
ral context, which help preserve each household’s unique
consumption patterns. With this hierarchical design, HiDiff-
Energy can generate realistic, individualized energy data over
extended periods, accurately modeling both large-scale trends
and fine-scale features 

---

## Features
- **Hierarchical Generation**: Coarse-to-fine approach generates long, coherent time-series data.  
- **Conditional Modeling**: Conditioned on household, temporal, and seasonal factors for context-aware and diverse profiles.  

---

## Setup

Clone the repository:
```bash
git clone https://github.com/your-username/HiDiffEnergy.git
```
Create a Python virtual environment:
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
Install dependencies:
```
pip install -r requirements.txt
```
Then run the train.py for each of of the code.
```
python train.py

```

## Additional Resources for reference

Below are links and articles that might help with **hyperparameter tuning**, **theoretical insights**, and **practical guidelines**

---

### 1.TimeGan
For an in-depth explanation or theoretical background:
- **Official Documentation**: (https://github.com/openai/maddpg](https://github.com/jsyoon0823/TimeGAN))

---

### 2. TimeVAE
Helpful resources for **hyperparameter tuning** and practical DQN insights:
- **Official Documentation**: [https://github.com/openai/maddpg](https://github.com/jsyoon0823/TimeGAN](https://github.com/wangyz1999/timeVAE-pytorch))


### 3. Diffusion model
For **implementation details** and **hyperparameter selection** in PPO:
- **PyTorch Tutorial**: [pytorch.org/tutorials/intermediate/reinforcement_ppo.html](https://pytorch.org/tutorials/intermediate/reinforcement_ppo.html)
- **Beginner-Friendly Repo**: [github.com/ericyangyu/PPO-for-Beginners](https://github.com/ericyangyu/PPO-for-Beginners)
- - **Official Documentation**: (https://github.com/amazon-science/unconditional-time-series-diffusion)

---

### 4. Policy Gradient Methods
Introductory materials on **policy gradient** (and how you might adapt it for multi-agent setups):
- **OpenAI Spinning Up**: [spinningup.openai.com VPG](https://spinningup.openai.com/en/latest/algorithms/vpg.html)
- **PyTorch Notebook**: [github.com/tsmatz RL tutorials (Policy Gradient)](https://github.com/tsmatz/reinforcement-learning-tutorials/blob/master/02-policy-gradient.ipynb)

---

### Other Relevant Sources
- **Gymnasium (OpenAI Gym) Docs**: [gymnasium.farama.org](https://gymnasium.farama.org/)

## Conclusion

Thank you for exploring **SolarSharer** and its multi-agent reinforcement learning framework. We hope this environment and the accompanying training scripts help accelerate your research and development. If you have any questions, suggestions, or encounter any issues, please feel free to **open an issue** or **contact us directly** anany.p.gupta-1@ou.edu . We value your feedback. Thankyou!





