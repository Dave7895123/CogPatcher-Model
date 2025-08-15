# The Cost of Frugal Learning: A Process Model and Proof-of-concept simulation about the Origin of Cognitive Bias
 This repository contains the official source code for the paper "The Cost of Frugal Learning", a process model on the origin of cognitive bias.
While mainstream computational models equate rational belief with complex optimization, the persistence of human cognitive biases suggests a different story. Here, we challenge the view that such biases are simply flaws, proposing instead that they are a systemic cost of a computationally frugal adaptive strategy. Our process model introduces an agent that, when faced with cognitive conflict, sidesteps demanding global updates by applying a simple, rule-based “belief patch”. This quick fix, however, leaves a “learning scar”: a belief that is computationally cheap but structurally fragile. Through simulations of the classic A-not-B error, we illustrate how these pre-existing scars precipitate systematic errors under cognitive load. Ultimately, we argue that some biases are not failures of reasoning, but the inevitable consequence of an adaptive system choosing a local, “good-enough” repair over costly global optimization.
## Getting Started

To get a local copy of the model up and running, please follow these simple steps.

### Prerequisites

Make sure you have the following software installed on your system:
* [Git](https://git-scm.com/)
* [Python](https://www.python.org/downloads/) (version 3.9 or later is recommended)
* `pip` (Python's package installer, which typically comes with Python)

### Installation

1.  **Clone the repository**

    Open your terminal or command prompt and clone this repository to your local machine:
    ```bash
    git clone [https://github.com/](https://github.com/)[Your-Username]/[Your-Repository-Name].git
    cd [Your-Repository-Name]
    ```
    *(Replace `[Your-Username]` and `[Your-Repository-Name]` with your actual GitHub username and repository name.)*

2.  **Create and Activate a Virtual Environment (Highly Recommended)**

    It is a best practice to create a virtual environment to keep the project's dependencies isolated.

    * On **macOS and Linux**:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    * On **Windows**:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```
    After activation, you will see `(venv)` at the beginning of your terminal prompt.

3.  **Install Dependencies**

    All required libraries are listed in the `requirements.txt` file. Install them with a single command:
    ```bash
    pip install -r requirements.txt
    ```

You are now ready to run the simulations! Please see the [Usage](#usage) section for instructions on how to run the model.

## Usage

This project contains two main experiments that should be run sequentially. First, the `CogPatcher` simulation is run to generate the experiential data. Second, the `Associative-Learner` script is run to train the control model on that data and reproduce the paper's key findings.

### Running the Experiment

Follow these steps in your terminal from the project's root directory.

**Step 1: Run the CogPatcher Simulation**

This step runs the main simulation for the `CogPatcher` model. The simulation will run for the number of steps specified in `main.py` (e.g., 100,000 steps) and will generate the `agent_experience_log.csv` file, which records all key events.

````bash
python main.py
Note: This simulation may take several minutes to complete. As the agent_experience_log.csv file is already included in this repository, you may skip this step if you only wish to re-run the analysis on the existing data.

Step 2: Train and Evaluate the Associative-Learner

This script trains the neural network (our Associative-Learner control model) on the data generated in Step 1 and evaluates its performance on the B-trials. It will print the final analysis, including the model's prediction and confidence for the critical A-non-B task, directly to the console.

Bash

python Associative-Learner.py
The output of this script contains the quantitative results used in the paper to demonstrate the mechanistic differences between the two models.

License
This project is licensed under the MIT License. See the accompanying LICENSE file for the full text.
